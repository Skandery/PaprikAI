# --- START OF FILE DP.py ---

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
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='a'), # Append mode
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# --- Constants (Potentially moved inside or made configurable if needed) ---
CLASS_DP = 0
CLASS_SP = 1
SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp', '.tif', '.tiff')

# --- Model Configuration (Defined here, loaded inside the function) ---
# NOTE: Hardcoding these here means changing the model requires editing the script.
# Consider passing these as arguments or using a config file for more flexibility.
DP_MODEL_NAME = "caformer_b36.sail_in22k_ft_in1k_384"
DP_MODEL_FILENAME = "99.11-DP-caformer_b36.sail_in22k_ft_in1k_384.pth"
# Assume model is in a 'models' subfolder relative to *this* script's location
# This path is resolved *when the function is called*.
CURRENT_FILE_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DP_MODEL_PATH = os.path.join(CURRENT_FILE_DIRECTORY, "./models", DP_MODEL_FILENAME)
INPUT_SIZE = 384 # Should match model's expected input size
NUM_CLASSES = 2  # DP=0, SP=1

# --- The Importable Function ---

def process_dp(
    folderpath: str,
    dp_output_folder: Optional[str] = None,
    *, # Make subsequent arguments keyword-only for clarity
    skip_first: bool = True,
    jpeg_quality: int = 92,
    dp_subfolder_name: str = "DP_Originals" # Name if dp_output_folder is None
) -> Dict[str, Any]:
    """
    Processes image files in a folder: loads model/transform, merges pairs,
    classifies, saves merged DP pairs in the input folder, and moves original
    DP pairs to a specified output folder or a default subfolder.

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

    Returns:
        A dictionary containing processing statistics:
        {
            'processed_pairs': int,
            'dp_saved': int,
            'dp_originals_moved_to': str  # The absolute path where originals were moved
        }
    """
    log.info("process_dp called: Model and transform will be loaded internally for this call.")
    log.info(f"--- Starting DP processing for folder: {folderpath} ---")

    # --- Initial Checks ---
    if not os.path.isdir(folderpath):
        log.error(f"Input folder not found or is not a directory: {folderpath}")
        return {'processed_pairs': 0, 'dp_saved': 0, 'dp_originals_moved_to': ''}

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

        log.info(f"Loading model structure '{DP_MODEL_NAME}'...")
        model = timm.create_model(DP_MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
        log.info(f"Loading state dict from: {DP_MODEL_PATH}")
        checkpoint = torch.load(DP_MODEL_PATH, map_location=device,weights_only=True)

        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        if all(key.startswith('module.') for key in state_dict):
            state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        model.to(device)
        model.eval() # Set model to evaluation mode
        log.info("Model loaded successfully and set to evaluation mode.")

        # 3. Create Transform
        if hasattr(model, 'default_cfg'):
            cfg = model.default_cfg
            current_input_size = cfg['input_size'][-1]
            mean=cfg['mean']
            std=cfg['std']
            interpolation=cfg.get('interpolation', 'bicubic')
            crop_pct=cfg.get('crop_pct', 1.0 if current_input_size == 224 else 0.9) # Adjusted default logic slightly
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
            crop_mode='center',
            crop_pct=crop_pct
        )
        log.info(f"Created model input transform with input_size={current_input_size}.")

    except FileNotFoundError as e:
        log.error(f"Setup failed: {e}")
        return {'processed_pairs': 0, 'dp_saved': 0, 'dp_originals_moved_to': ''}
    except Exception as e:
        log.error(f"Error during internal setup (model/transform/device): {e}", exc_info=True)
        # Return an empty result if setup fails
        return {'processed_pairs': 0, 'dp_saved': 0, 'dp_originals_moved_to': ''}
    # --- End of Internal Setup ---


    # --- Determine and Create DP Output Directory ---
    if dp_output_folder is None:
        dp_dir = os.path.join(CURRENT_FILE_DIRECTORY,"DP_Originales",os.path.basename(folderpath))
        log.info(f"No explicit DP output folder provided. Using default subfolder: {dp_dir}")
    else:
        dp_dir = dp_output_folder
        log.info(f"Using specified DP output folder for originals: {dp_dir}")

    try:
        os.makedirs(dp_dir, exist_ok=True)
        log.info(f"Ensured DP originals directory exists: {dp_dir}")
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
            return {'processed_pairs': 0, 'dp_saved': 0, 'dp_originals_moved_to': dp_dir}
        log.info(f"Found {len(image_files)} potential image files.")
    except OSError as e:
        log.error(f"Error listing files in {folderpath}: {e}")
        return {'processed_pairs': 0, 'dp_saved': 0, 'dp_originals_moved_to': dp_dir}

    # --- Skip First Image (Optional) ---
    if skip_first and image_files:
        if len(image_files) > 0:
            skipped_file = image_files.pop(0)
            log.info(f"Skipped first image file: {skipped_file}")
        else:
             log.warning("Skip first requested, but no image files were found to skip.")

    if len(image_files) < 2:
        log.warning(f"Not enough images left ({len(image_files)}) to form pairs after potential skipping.")
        if image_files:
             log.warning(f"Remaining file not processed: {image_files[0]}")
        log.info(f"--- DP processing completed for folder: {folderpath} (No pairs processed) ---")
        return {'processed_pairs': 0, 'dp_saved': 0, 'dp_originals_moved_to': dp_dir}

    # --- Process Images in Pairs ---
    processed_count = 0
    dp_saved_count = 0
    files_to_process = image_files

    for i in range(0, len(files_to_process) - 1, 2):
        img_file1 = files_to_process[i]     # Right side
        img_file2 = files_to_process[i+1]   # Left side
        img_path1 = os.path.join(folderpath, img_file1)
        img_path2 = os.path.join(folderpath, img_file2)

        if not os.path.exists(img_path1) or not os.path.exists(img_path2):
            log.warning(f"One or both files in pair ('{img_file1}', '{img_file2}') no longer exist in source folder. Skipping pair.")
            continue

        log.info(f"Processing pair: Left='{img_file2}', Right='{img_file1}'")

        try:
            vips_img1 = pyvips.Image.new_from_file(img_path1)
            vips_img2 = pyvips.Image.new_from_file(img_path2)

            if vips_img1.height != vips_img2.height:
                log.warning(f"Height mismatch: '{img_file1}' ({vips_img1.height}px) vs '{img_file2}' ({vips_img2.height}px). Resizing '{img_file2}'.")
                scale_factor = vips_img1.height / vips_img2.height
                vips_img2_resized = vips_img2.resize(scale_factor, kernel='lanczos3')
                log.info(f"Resized '{img_file2}' to {vips_img2_resized.width}x{vips_img2_resized.height}")
                vips_img2 = vips_img2_resized
                if vips_img1.width != vips_img2.width:
                     log.warning(f"Widths still differ after height-based resize: '{img_file1}' ({vips_img1.width}px) vs '{img_file2}' ({vips_img2.width}px).")

            merged_vips_img = vips_img2.join(vips_img1, 'horizontal', expand=True, align='centre')

            if merged_vips_img.bands == 4:
                merged_vips_img = merged_vips_img.extract_band(0, n=3)
            elif merged_vips_img.bands == 1:
                 merged_vips_img = merged_vips_img.bandjoin([merged_vips_img, merged_vips_img])
            elif merged_vips_img.bands != 3:
                 log.error(f"Merged image pair '{img_file2}'-'{img_file1}' has unexpected bands: {merged_vips_img.bands}. Skipping.")
                 continue

            np_image = merged_vips_img.numpy()

            if np_image.dtype != np.uint8:
                 if np_image.max() <= 1.0 and np_image.min() >= 0.0 :
                     np_image = (np_image * 255).astype(np.uint8)
                 elif np_image.max() <= 255 and np_image.min() >= 0 :
                     np_image = np_image.astype(np.uint8)
                 else:
                      np_image = np.clip(np_image, 0, 255).astype(np.uint8)

            pil_image = Image.fromarray(np_image)
            input_tensor = transform(pil_image).unsqueeze(0).to(device)

            with torch.no_grad():
                use_amp = device.type == 'cuda'
                with autocast(device_type='cuda', enabled=use_amp):
                    outputs = model(input_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    confidence, predicted_class = torch.max(probabilities, 1)
                    prediction = predicted_class.item()
                    confidence_score = confidence.item()

            processed_count += 1
            result_label = "DP" if prediction == CLASS_DP else "SP"
            log.info(f"Model prediction for '{img_file2}' + '{img_file1}': {prediction} ({result_label}) with confidence {confidence_score:.4f}")

            if prediction == CLASS_DP:
                base1 = os.path.splitext(img_file1)[0]
                base2 = os.path.splitext(img_file2)[0]
                save_filename = f"{base1}-{base2}.jpg"
                save_path = os.path.join(folderpath, save_filename)

                try:
                    log.info(f"Saving merged DP image to: {save_path} with quality {jpeg_quality}")
                    merged_vips_img.write_to_file(save_path, Q=jpeg_quality)
                    dp_saved_count += 1

                    move_success = True
                    for src_file, src_path in [(img_file1, img_path1), (img_file2, img_path2)]:
                        dest_path = os.path.join(dp_dir, src_file)
                        try:
                            # Simple overwrite warning, could add renaming logic if needed
                            if os.path.exists(dest_path):
                                log.warning(f"Destination file already exists: '{dest_path}'. Overwriting.")
                            log.info(f"Moving original '{src_file}' from '{folderpath}' to '{dp_dir}'")
                            shutil.move(src_path, dest_path)
                        except (IOError, OSError, shutil.Error) as move_e:
                            log.error(f"Failed to move source file '{src_file}' to '{dest_path}': {move_e}")
                            move_success = False
                            break
                    if not move_success and os.path.exists(save_path):
                         log.warning(f"Originals for saved DP '{save_filename}' could not be fully moved. Merged file remains.")

                except pyvips.Error as save_e:
                    log.error(f"Failed to save merged image '{save_path}': {save_e}")
                except Exception as e:
                    log.error(f"Unexpected error during DP saving/moving for pair '{img_file1}/{img_file2}': {e}")
            else:
                log.info(f"Pair '{img_file2}' + '{img_file1}' classified as SP. No action taken.")

        except pyvips.Error as vips_e:
            log.error(f"PyVIPS error processing pair '{img_file1}', '{img_file2}': {vips_e}. Skipping.")
            continue
        except FileNotFoundError as fnf_e:
             log.error(f"File not found during processing pair: {fnf_e}. Skipping.")
             continue
        except MemoryError as mem_e:
             log.error(f"Memory error processing pair '{img_file1}', '{img_file2}'. Error: {mem_e}. Skipping.")
             continue
        except Exception as e:
            log.error(f"Unexpected error processing pair '{img_file1}', '{img_file2}': {e}. Skipping.", exc_info=True)
            continue

    # --- Handle Leftover Image ---
    if len(files_to_process) % 2 != 0:
        leftover_file = files_to_process[-1]
        leftover_path = os.path.join(folderpath, leftover_file)
        if os.path.exists(leftover_path):
            log.info(f"One image file left over (not processed in a pair): {leftover_file}")
        # No need for 'else' as the file might have been moved in a previous step if pair processing was interrupted

    log.info(f"Processed {processed_count} pairs in '{folderpath}'.")
    log.info(f"Saved {dp_saved_count} merged DP images in '{folderpath}'.")
    log.info(f"Attempted to move {dp_saved_count * 2} original DP source images to '{dp_dir}'.")
    log.info(f"--- DP processing completed for folder: {folderpath} ---")

    # --- Clean up model from memory (optional, might help if calling repeatedly) ---
    del model
    del transform
    if 'cuda' in str(device):
        torch.cuda.empty_cache()
        log.info("CUDA cache cleared.")

    return {
        'processed_pairs': processed_count,
        'dp_saved': dp_saved_count,
        'dp_originals_moved_to': os.path.abspath(dp_dir)
    }


# --- Example Usage ---
if __name__ == '__main__':
    log.info("--- Starting Example Usage of process_dp (Internal Model Loading) ---")

    # --- Configuration (Adjust these paths) ---
    EXAMPLE_INPUT_FOLDER = r"C:\Paprika2\Projet 2\Brigades immunitaires (Les) - Black T07" # <--- CHANGE THIS
    # Optional: Specify where to move the original DP files

    # --- Call the Function ---
    # Note: No model loading happens here. It happens inside process_dp.

    if os.path.isdir(EXAMPLE_INPUT_FOLDER):
        log.info(f"Calling process_dp on folder: {EXAMPLE_INPUT_FOLDER}")

        # Example 1: Using a specific output folder for originals
        log.info("--- Running Example 1: Specific DP Output Folder ---")
        results1 = process_dp(
            folderpath=EXAMPLE_INPUT_FOLDER,
            skip_first=True, # Example: Use keyword argument
            jpeg_quality=92  # Example: Use keyword argument
        )
        print(f"Processing results (Example 1): {results1}")


    else:
        log.error(f"Example input folder not found or not a directory: {EXAMPLE_INPUT_FOLDER}. Skipping process_dp execution.")

    log.info("--- Example Usage Finished ---")

# --- END OF FILE DP.py ---