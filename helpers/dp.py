# --- START OF FILE dp.py ---

import os
import logging
import pyvips
import shutil
import torch
import timm
import numpy as np
from PIL import Image
from timm.data import create_transform
from torch.amp import autocast
from typing import List, Tuple, Callable, Optional, Dict, Any
import natsort # For natural sorting of filenames like 1.jpg, 10.jpg, 2.jpg


# --- Logging Setup ---
log_file = "image_processing.log"
# Clear previous handlers to avoid duplicate logs if re-run in the same session
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    level=logging.INFO, # Changed to INFO for more detailed logs during execution
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='a'), # Append mode
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

for name in logging.root.manager.loggerDict:
    if 'vips' in name.lower():
        logging.getLogger(name).setLevel(logging.WARNING)



# --- Constants ---
CLASS_SP = 0  
CLASS_DP = 1
SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp', '.tif', '.tiff')

# --- Model Configuration ---
DP_MODEL_NAME = "caformer_b36.sail_in22k_ft_in1k_384"
DP_MODEL_FILENAME = "dp-99.11-caformer_b36.sail_in22k_ft_in1k_384.pth"
CURRENT_FILE_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DP_MODEL_PATH = os.path.join(CURRENT_FILE_DIRECTORY, "models", DP_MODEL_FILENAME)
INPUT_SIZE = 384
NUM_CLASSES = 2
# --- The Importable Function ---

def process_dp(
    folderpath: str,
    dp_output_folder: Optional[str] = None,
    *, # Make subsequent arguments keyword-only for clarity
    skip_first: bool = True,
    jpeg_quality: int = 92,
    sp_threshold: float = 0.5 # NEW: Threshold for SP classification
) -> Dict[str, Any]:
    """
    Processes image files in a folder: loads model/transform, merges pairs,
    classifies based on a threshold, saves merged DP pairs, and moves originals.

    **Warning:** This function loads the model and creates the transform internally
    on each call, which can be inefficient for processing multiple folders.

    Args:
        folderpath: Path to the directory containing image files.
        dp_output_folder: Path where original DP image pairs should be moved.
                          If None, originals are moved to a subfolder named
                          `dp_subfolder_name` within `folderpath`.
        skip_first: (Keyword only) Whether to skip the first image file found
                    (sorted naturally). Defaults to True.
        jpeg_quality: (Keyword only) Quality setting (1-100) for saving merged
                      JPEG images. Defaults to 92.
        dp_subfolder_name: (Keyword only) Name of the subfolder created within
                           `folderpath` to store original DP pairs if
                           `dp_output_folder` is None. Defaults to "DP_Originals".
        sp_threshold: (Keyword only) The probability threshold above which a pair
                      is classified as SP (Single Page, class 1). If the SP
                      probability is <= this threshold, it's classified as DP
                      (Double Page, class 0). Defaults to 0.5 (equivalent to
                      choosing the class with the highest probability).
                      Setting to 0.45 makes it *easier* to classify as SP.
                      Setting to 0.55 makes it *harder* to classify as SP.

    Returns:
        A dictionary containing processing statistics:
        {
            'processed_pairs': int,
            'dp_saved': int,
            'dp_originals_moved_to': str  # The absolute path where originals were moved
        }
    """
    log.debug("process_dp called: Model and transform will be loaded internally for this call.")
    log.debug(f"--- Starting DP processing for folder: {folderpath} ---")
    log.debug(f"Using SP classification threshold: {sp_threshold}") # Log the threshold

    # --- Initial Checks ---
    if not os.path.isdir(folderpath):
        log.error(f"Input folder not found or is not a directory: {folderpath}")
        return {'processed_pairs': 0, 'dp_saved': 0, 'dp_originals_moved_to': ''}
    if not 0.0 < sp_threshold < 1.0:
        log.warning(f"sp_threshold ({sp_threshold}) is outside the typical (0, 1) range. Using it anyway.")
        # Or raise ValueError("sp_threshold must be between 0 and 1")


    # --- Setup Device, Model, and Transform (INSIDE THE FUNCTION) ---
    model = None
    transform = None
    device = None

    try:
        # 1. Device Setup
        if torch.cuda.is_available():
            device = torch.device("cuda")
            log.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            log.info("CUDA not available. Using CPU.")

        # 2. Load Model
        if not os.path.exists(DP_MODEL_PATH):
            log.error(f"Model file not found at: {DP_MODEL_PATH}")
            raise FileNotFoundError(f"Model file not found: {DP_MODEL_PATH}")

        log.debug(f"Loading model structure '{DP_MODEL_NAME}'...")
        model = timm.create_model(DP_MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
        log.debug(f"Loading state dict from: {DP_MODEL_PATH}")
        # Use weights_only=True for security if loading untrusted checkpoints
        checkpoint = torch.load(DP_MODEL_PATH, map_location=device, weights_only=True)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict) and 'model' in checkpoint: # Handle timm save format
             state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # Remove 'module.' prefix if saved with DataParallel
        if all(key.startswith('module.') for key in state_dict):
            log.info("Removing 'module.' prefix from state dict keys.")
            state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        model.to(device)
        model.eval() # Set model to evaluation mode
        log.debug("Model loaded successfully and set to evaluation mode.")

        # 3. Create Transform
        # Use model's default_cfg if available
        if hasattr(model, 'default_cfg'):
            cfg = model.default_cfg
            current_input_size = cfg['input_size'][-1] # Get the image size
            mean=cfg['mean']
            std=cfg['std']
            interpolation=cfg.get('interpolation', 'bicubic')
            # Adjust crop_pct logic slightly for common cases
            crop_pct=cfg.get('crop_pct', 1.0 if current_input_size == 224 else 0.9)
            log.info(f"Using model default_cfg for transform: input_size={current_input_size}, mean={mean}, std={std}, interpolation={interpolation}, crop_pct={crop_pct}")
        else:
             log.warning("Model has no default_cfg. Using hardcoded transform parameters.")
             current_input_size=INPUT_SIZE
             mean=(0.485, 0.456, 0.406)
             std=(0.229, 0.224, 0.225)
             interpolation='bicubic'
             crop_pct = 1.0 # Assume center crop for inference

        transform = create_transform(
            input_size=current_input_size,
            is_training=False,
            mean=mean,
            std=std,
            interpolation=interpolation,
            crop_mode='center', # Use 'center' for consistent inference cropping
            crop_pct=crop_pct
        )
        log.debug(f"Created model input transform with input_size={current_input_size}.")

    except FileNotFoundError as e:
        log.error(f"Setup failed: {e}")
        return {'processed_pairs': 0, 'dp_saved': 0, 'dp_originals_moved_to': ''}
    except Exception as e:
        log.error(f"Error during internal setup (model/transform/device): {e}", exc_info=True)
        return {'processed_pairs': 0, 'dp_saved': 0, 'dp_originals_moved_to': ''}
    # --- End of Internal Setup ---


    # --- Determine and Create DP Output Directory ---
    if dp_output_folder is None:
        dp_dir = os.path.join(CURRENT_FILE_DIRECTORY,"DP_Originales",os.path.basename(folderpath))
        log.debug(f"No explicit DP output folder provided. Using default subfolder: {dp_dir}")
    else:
        dp_dir = dp_output_folder
        log.debug(f"Using specified DP output folder for originals: {dp_dir}")
 
    try:
        os.makedirs(dp_dir, exist_ok=True)
        log.debug(f"Ensured DP originals directory exists: {dp_dir}")
    except OSError as e:
        log.error(f"Failed to create DP originals directory '{dp_dir}': {e}")
        return {'processed_pairs': 0, 'dp_saved': 0, 'dp_originals_moved_to': dp_dir}

    # --- List and Sort Image Files ---
    try:
        all_files = [f for f in os.listdir(folderpath)
                     if os.path.isfile(os.path.join(folderpath, f)) and
                        f.lower().endswith(SUPPORTED_EXTENSIONS)]
        image_files = natsort.natsorted(all_files)

        if not image_files:
            log.warning(f"No image files with supported extensions found in {folderpath}")
            return {'processed_pairs': 0, 'dp_saved': 0, 'dp_originals_moved_to': os.path.abspath(dp_dir)}
        log.info(f"Found {len(image_files)} potential image files.")
    except OSError as e:
        log.error(f"Error listing files in {folderpath}: {e}")
        return {'processed_pairs': 0, 'dp_saved': 0, 'dp_originals_moved_to': os.path.abspath(dp_dir)}

    # --- Skip First Image (Optional) ---
    if skip_first and image_files:
        if len(image_files) > 0:
            skipped_file = image_files.pop(0)
            log.debug(f"Skipped first image file: {skipped_file}")
        else:
             log.warning("Skip first requested, but no image files were found to skip.")

    if len(image_files) < 2:
        log.warning(f"Not enough images left ({len(image_files)}) to form pairs after potential skipping.")
        if image_files:
             log.warning(f"Remaining file not processed: {image_files[0]}")
        log.info(f"--- DP processing completed for folder: {folderpath} (No pairs processed) ---")
        return {'processed_pairs': 0, 'dp_saved': 0, 'dp_originals_moved_to': os.path.abspath(dp_dir)}

    # --- Process Images in Pairs ---
    processed_count = 0
    dp_saved_count = 0
    files_to_process = image_files

    for i in range(0, len(files_to_process) - 1, 2):
        img_file1 = files_to_process[i]     # Right side
        img_file2 = files_to_process[i+1]   # Left side
        img_path1 = os.path.join(folderpath, img_file1)
        img_path2 = os.path.join(folderpath, img_file2)

        # Check existence again inside the loop in case files were moved/deleted
        if not os.path.exists(img_path1) or not os.path.exists(img_path2):
            log.warning(f"One or both files in pair ('{img_file1}', '{img_file2}') no longer exist in source folder. Skipping pair.")
            continue

        log.debug(f"Processing pair: Left='{img_file2}', Right='{img_file1}'")

        try:
            # --- Image Loading and Merging ---
            vips_img1 = pyvips.Image.new_from_file(img_path1)
            vips_img2 = pyvips.Image.new_from_file(img_path2)

            # Handle potential height mismatch
            if vips_img1.height != vips_img2.height:
                log.warning(f"Height mismatch: '{img_file1}' ({vips_img1.height}px) vs '{img_file2}' ({vips_img2.height}px). Resizing '{img_file2}' to match height of '{img_file1}'.")
                scale_factor = vips_img1.height / vips_img2.height
                vips_img2_resized = vips_img2.resize(scale_factor, kernel='lanczos3')
                log.info(f"Resized '{img_file2}' to {vips_img2_resized.width}x{vips_img2_resized.height}")
                vips_img2 = vips_img2_resized
                # Optional: Log width difference after resize if significant
                if vips_img1.width != vips_img2.width:
                     log.warning(f"Widths still differ after height-based resize: '{img_file1}' ({vips_img1.width}px) vs '{img_file2}' ({vips_img2.width}px).")

            # Join images (Left = img2, Right = img1)
            merged_vips_img = vips_img2.join(vips_img1, 'horizontal', expand=True, align='centre')

            # Ensure 3 bands (RGB)
            if merged_vips_img.bands == 4: # RGBA
                log.debug(f"Merged image pair '{img_file2}'-'{img_file1}' has 4 bands. Extracting RGB.")
                merged_vips_img = merged_vips_img.extract_band(0, n=3)
            elif merged_vips_img.bands == 1: # Grayscale
                 log.debug(f"Merged image pair '{img_file2}'-'{img_file1}' has 1 band. Converting to RGB.")
                 merged_vips_img = merged_vips_img.bandjoin([merged_vips_img, merged_vips_img])
            elif merged_vips_img.bands != 3:
                 log.error(f"Merged image pair '{img_file2}'-'{img_file1}' has unexpected bands: {merged_vips_img.bands}. Skipping.")
                 continue

            # --- Prepare for Model ---
            # Convert VIPS image to NumPy array -> PIL Image -> Tensor
            np_image = merged_vips_img.numpy()

            # Ensure correct dtype (uint8) for PIL
            if np_image.dtype != np.uint8:
                 log.debug(f"NumPy array dtype is {np_image.dtype}. Attempting conversion to uint8.")
                 if np_image.max() <= 1.0 and np_image.min() >= 0.0 : # Float 0-1 range
                     np_image = (np_image * 255).astype(np.uint8)
                 elif np_image.max() <= 255 and np_image.min() >= 0 : # Int/Float 0-255 range
                     np_image = np_image.astype(np.uint8)
                 else: # Other ranges, clip and convert
                      log.warning(f"NumPy array value range ({np_image.min()}-{np_image.max()}) is unusual. Clipping to 0-255 and converting to uint8.")
                      np_image = np.clip(np_image, 0, 255).astype(np.uint8)

            pil_image = Image.fromarray(np_image)
            input_tensor = transform(pil_image).unsqueeze(0).to(device)

            # --- Model Inference and Classification with Threshold ---
            with torch.no_grad():
                use_amp = device.type == 'cuda' # Enable AMP only on CUDA
                with autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(input_tensor)
                    probabilities = torch.softmax(outputs, dim=1)

                    # Get probabilities for each class
                    dp_prob = probabilities[0, CLASS_SP].item()
                    sp_prob = probabilities[0, CLASS_DP].item()

                    # Apply the custom threshold logic
                    if sp_prob > sp_threshold:
                        prediction = CLASS_DP
                        confidence_score = sp_prob # Confidence is the probability of the chosen class (SP)
                        result_label = "SP"
                    else:
                        prediction = CLASS_SP
                        confidence_score = dp_prob # Confidence is the probability of the chosen class (DP)
                        result_label = "DP"

            processed_count += 1
            log.debug(f"Prediction for '{img_file2}' + '{img_file1}': {result_label} (DP Prob: {dp_prob:.4f}, SP Prob: {sp_prob:.4f}). Threshold for SP: >{sp_threshold}. Confidence in chosen class: {confidence_score:.4f}")

            # --- Action based on Classification ---
            if prediction == CLASS_SP:
                base1 = os.path.splitext(img_file1)[0]
                base2 = os.path.splitext(img_file2)[0]
                # Use a consistent naming convention, e.g., left-right
                save_filename = f"{base1}-{base2}.jpg"
                save_path = os.path.join(folderpath, save_filename)

                try:
                    log.info(f"Saving merged DP image to: {save_path} with quality {jpeg_quality}")
                    merged_vips_img.write_to_file(save_path, Q=jpeg_quality)
                    dp_saved_count += 1

                    # Move original files AFTER successful save
                    move_success = True
                    files_to_move = [(img_file1, img_path1), (img_file2, img_path2)]
                    moved_files_temp = [] # Keep track of successfully moved files in case of partial failure

                    for src_file, src_path in files_to_move:
                        dest_path = os.path.join(dp_dir, src_file)
                        try:
                            if os.path.exists(dest_path):
                                log.warning(f"Destination file already exists: '{dest_path}'. Overwriting.")
                                # Optional: Implement renaming logic here if overwrite is not desired
                                # e.g., add a timestamp or counter to the filename
                            log.debug(f"Moving original '{src_file}' from '{folderpath}' to '{dp_dir}'")
                            shutil.move(src_path, dest_path)
                            moved_files_temp.append((dest_path, src_path)) # Store dest and original src
                        except (IOError, OSError, shutil.Error) as move_e:
                            log.error(f"Failed to move source file '{src_file}' to '{dest_path}': {move_e}")
                            move_success = False
                            # Attempt to move back already moved files from this pair
                            log.warning("Attempting to roll back move operation for this pair...")
                            for moved_dest, original_src in moved_files_temp:
                                try:
                                    log.info(f"Moving '{moved_dest}' back to '{original_src}'")
                                    shutil.move(moved_dest, original_src)
                                except Exception as rollback_e:
                                    log.error(f"Failed to move file back during rollback: {rollback_e}. File '{os.path.basename(moved_dest)}' might be left in '{dp_dir}'.")
                            break # Stop moving files for this pair

                    if not move_success and os.path.exists(save_path):
                         log.warning(f"Originals for saved DP '{save_filename}' could not be fully moved (rollback attempted). Merged file remains in source folder.")
                         # Optionally delete the saved merged file if originals couldn't be moved:
                         # try:
                         #     os.remove(save_path)
                         #     log.info(f"Removed merged file '{save_path}' due to move failure.")
                         #     dp_saved_count -= 1 # Decrement counter
                         # except OSError as del_e:
                         #     log.error(f"Failed to remove merged file '{save_path}' after move failure: {del_e}")


                except pyvips.Error as save_e:
                    log.error(f"Failed to save merged image '{save_path}': {save_e}")
                except Exception as e:
                    log.error(f"Unexpected error during DP saving/moving for pair '{img_file1}/{img_file2}': {e}", exc_info=True)
            else: # Classified as SP
                log.debug(f"Pair '{img_file2}' + '{img_file1}' classified as SP based on threshold {sp_threshold}. Originals remain in place.")

        except pyvips.Error as vips_e:
            log.error(f"PyVIPS error processing pair '{img_file1}', '{img_file2}': {vips_e}. Skipping pair.")
            continue
        except FileNotFoundError as fnf_e:
             log.error(f"File not found during processing pair: {fnf_e}. Skipping pair.")
             continue
        except MemoryError as mem_e:
             log.error(f"Memory error processing pair '{img_file1}', '{img_file2}'. System might be low on RAM. Error: {mem_e}. Skipping pair.")
             continue
        except Exception as e:
            log.error(f"Unexpected error processing pair '{img_file1}', '{img_file2}': {e}. Skipping pair.", exc_info=True)
            continue

    # --- Handle Leftover Image ---
    if len(files_to_process) % 2 != 0:
        leftover_file = files_to_process[-1]
        leftover_path = os.path.join(folderpath, leftover_file)
        # Check existence in case it was part of a failed move operation earlier
        if os.path.exists(leftover_path):
            log.info(f"One image file left over (not processed in a pair): {leftover_file}")

    log.info(f"Processed {processed_count} pairs in '{folderpath}'.")
    log.info(f"Saved {dp_saved_count} merged DP images in '{folderpath}'.")
    log.info(f"Attempted to move {dp_saved_count * 2} original DP source images to '{os.path.abspath(dp_dir)}'.")
    log.info(f"--- DP processing completed for folder: {folderpath} ---")

    # --- Clean up model from memory ---
    del model
    del transform
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        log.info("CUDA cache cleared.")

    return {
        'processed_pairs': processed_count,
        'dp_saved': dp_saved_count,
        'dp_originals_moved_to': os.path.abspath(dp_dir)
    }


# --- Example Usage ---
#if __name__ == '__main__':
#    log.info("--- Starting Example Usage of process_dp (Internal Model Loading) ---")
#
#    # --- Configuration (Adjust these paths) ---
#    # Use a raw string (r"...") or double backslashes (\\) for Windows paths
#    EXAMPLE_INPUT_FOLDER = r"C:\Paprika2\Git3\PaprikAI\output\Sanctuary - Perfect Edition - T03" # <--- CHANGE THIS
#    # Optional: Specify a base directory where DP_Originals subfolders will be created
#    EXAMPLE_DP_OUTPUT_BASE = r"C:\Paprika2\Git3\PaprikAI\output"
#
#    # --- Call the Function ---
#    if os.path.isdir(EXAMPLE_INPUT_FOLDER):
#        log.info(f"Calling process_dp on folder: {EXAMPLE_INPUT_FOLDER}")
#
#        # Example 3: Custom threshold (0.55), specific DP output base folder
#        log.info("--- Running Example 3: Custom Threshold (0.55), Specific Output Base ---")
#        results3 = process_dp(
#            folderpath=EXAMPLE_INPUT_FOLDER,
#            sp_threshold=0.1 # Specify base output dir
#        )
#        print(f"Processing results (Example 3 - Threshold 0.55): {results3}")
#        print("-" * 30)
#
#    else:
#        log.error(f"Example input folder not found or not a directory: {EXAMPLE_INPUT_FOLDER}. Skipping process_dp execution.")
#
#    log.info("--- Example Usage Finished ---")

# --- END OF FILE dp.py ---