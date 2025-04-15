# --- START OF FILE renomage_logic.py ---

import os
from PIL import Image, ImageStat
import numpy as np
import torch
import timm
from timm.data import create_transform, resolve_data_config
from natsort import natsorted
import shutil
import time
import logging # Use logging for cleaner output control

# --- Configuration (Module Level) ---
# These can be adjusted here before calling the function, or potentially passed as arguments if more flexibility is needed later.
MODEL_PATH = r"F:\Projet-DP\Numerotation\Numerotation-regnety_320.swag_ft_in1k.pth" # Use raw string
MODEL_NAME = "regnety_320.swag_ft_in1k"
NUM_CLASSES = 2
NUM_PAIRS = 15 # Number of pairs for inference batches
WHITE_THRESHOLD = 250 # Pixel value threshold for white page check (0-255)
WHITE_STDDEV_THRESHOLD = 5 # Max standard deviation for white page check
ALLOWED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp') # Lowercase extensions

# --- Logging Setup ---
# Configure logging to control output level easily
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
# You could change level=logging.INFO to logging.WARNING to reduce output further

# --- Helper Functions (Internal to this module) ---

def _is_image_white(image_path, threshold=WHITE_THRESHOLD, stddev_threshold=WHITE_STDDEV_THRESHOLD):
    """Checks if an image is predominantly white."""
    try:
        img = Image.open(image_path).convert('L') # Convert to grayscale
        stat = ImageStat.Stat(img)
        is_white = stat.mean[0] >= threshold and stat.stddev[0] < stddev_threshold
        # logging.debug(f"White check: {os.path.basename(image_path)} - Mean: {stat.mean[0]:.2f}, StdDev: {stat.stddev[0]:.2f} -> {is_white}")
        return is_white
    except FileNotFoundError:
        logging.warning(f"File not found while checking if white: {image_path}")
        return False
    except Exception as e:
        logging.error(f"Error checking if image is white ({image_path}): {e}")
        return False

def _merge_images_side_by_side(img_path1, img_path2):
    """Merges two images side-by-side."""
    try:
        img1 = Image.open(img_path1).convert('RGB')
        img2 = Image.open(img_path2).convert('RGB')
    except FileNotFoundError as e:
        logging.error(f"Error opening image for merging: {e}")
        return None
    except Exception as e:
        logging.error(f"Error processing image {img_path1} or {img_path2} for merging: {e}")
        return None

    width1, height1 = img1.size
    width2, height2 = img2.size
    total_width = width1 + width2
    max_height = max(height1, height2)
    merged_image = Image.new('RGB', (total_width, max_height), color='white')
    merged_image.paste(img1, (0, 0))
    merged_image.paste(img2, (width1, 0))
    return merged_image

def _run_inference(model, pil_images, transform, device):
    """Runs inference on a list of PIL images."""
    model.eval()
    batch_tensors = []
    valid_image_count = 0
    for img in pil_images: # No tqdm here for library use
        if img is None:
            logging.warning("Skipping a None image encountered during preprocessing.")
            continue
        try:
            tensor_img = transform(img)
            batch_tensors.append(tensor_img)
            valid_image_count += 1
        except Exception as e:
            logging.error(f"Error transforming image for inference: {e}")
            # Decide if you want to skip or halt; skipping seems reasonable
            continue

    if not batch_tensors:
        logging.error("No valid images to process in the batch.")
        return np.array([]) # Return empty array, caller must handle

    logging.info(f"Running inference on batch of {len(batch_tensors)} images.")
    batch = torch.stack(batch_tensors).to(device)
    with torch.no_grad():
        try:
            outputs = model(batch)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            return predictions.cpu().numpy()
        except Exception as e:
            logging.error(f"Error during model inference: {e}")
            return np.array([]) # Return empty array on error

def _create_white_page_if_needed(folder_path):
    """
    Creates a white image placeholder based on specific conditions of the second image file
    AFTER potential renaming. Uses natsorted for reliable sorting.
    Conditions: Second file is 001.jpg, 003.jpg, etc. (lowercase check)
    Creates: 000a.jpg, 002.jpg, etc. respectively.
    Returns: True if created or already exists, False on error or if not needed.
    """
    logging.info(f"Checking for white page creation in: {folder_path}")
    try:
        # Get all jpg/jpeg files, sorted naturally, lowercase for matching
        image_files = natsorted([f for f in os.listdir(folder_path)
                                 if f.lower().endswith(('.jpg', '.jpeg'))])

        if len(image_files) < 2:
            logging.info("Folder has less than 2 images. Skipping white page creation.")
            return True # Not an error state

        second_image_name = image_files[1]
        second_image_path = os.path.join(folder_path, second_image_name)

        # Define the mapping using lowercase
        name_mapping = {
            '001.jpg': '000a.jpg', '001.jpeg': '000a.jpeg',
            '003.jpg': '002.jpg',  '003.jpeg': '002.jpeg',
            '005.jpg': '004.jpg',  '005.jpeg': '004.jpeg',
            '007.jpg': '006.jpg',  '007.jpeg': '006.jpeg',
            '009.jpg': '008.jpg',  '009.jpeg': '008.jpeg',
            # Add more if needed, ensure consistency with jpeg
        }

        second_name_lower = second_image_name.lower()
        if second_name_lower in name_mapping:
            white_image_name = name_mapping[second_name_lower]
            white_image_path = os.path.join(folder_path, white_image_name)

            if os.path.exists(white_image_path):
                logging.info(f"White page '{white_image_name}' already exists. Skipping creation.")
                return True

            try:
                # Use the second image's dimensions
                with Image.open(second_image_path) as img:
                    width, height = img.size
                # Create and save the white image
                white_image = Image.new('RGB', (width, height), color='white')
                white_image.save(white_image_path)
                logging.info(f"SUCCESS: Created white page '{white_image_name}' (size {width}x{height}).")
                return True

            except FileNotFoundError:
                 logging.error(f"Could not find image '{second_image_path}' to get dimensions for white page.")
                 return False
            except Exception as e:
                logging.error(f"Error creating white page '{white_image_name}': {e}")
                return False
        else:
            logging.info(f"Second image '{second_image_name}' does not match criteria for white page creation.")
            return True # Not an error state

    except Exception as e:
        logging.error(f"An unexpected error occurred during white page check: {e}")
        return False

def _rename_files_safely(folder_path, original_files_full_paths, new_basenames, dry_run=True):
    """
    Renames files safely using temporary names.
    Returns: True if successful (or dry run), False otherwise.
    """
    if len(original_files_full_paths) != len(new_basenames):
        logging.error("Mismatch between number of original files and new filenames.")
        return False

    action = "(DRY RUN)" if dry_run else "(EXECUTING)"
    logging.info(f"--- Renaming Files {action} in {folder_path} ---")

    rename_plan = []
    temp_filenames = []
    final_filenames = []
    timestamp = int(time.time()) # Unique identifier for temp files

    # Create plan and temporary names
    for i, old_full_path in enumerate(original_files_full_paths):
        if not os.path.exists(old_full_path):
             logging.warning(f"Original file not found, skipping rename planning for: {old_full_path}")
             # This introduces a mismatch, maybe better to abort? Or adjust new_basenames?
             # For now, let's abort if a file is missing during planning.
             logging.error("Aborting rename due to missing source file during planning.")
             return False # Abort if a source file is missing

        old_basename = os.path.basename(old_full_path)
        new_basename = new_basenames[i]
        _, ext = os.path.splitext(old_basename)
        temp_basename = f"__temp_{timestamp}_{i}{ext}" # Unique temporary name
        temp_full_path = os.path.join(folder_path, temp_basename)
        final_full_path = os.path.join(folder_path, new_basename)

        logging.info(f"Plan: '{old_basename}' -> '{new_basename}'")

        rename_plan.append({
            "old_full": old_full_path,
            "temp_full": temp_full_path,
            "final_full": final_full_path,
            "old_base": old_basename,
            "new_base": new_basename
        })
        temp_filenames.append(temp_full_path)
        final_filenames.append(final_full_path)

    if dry_run:
        logging.info("DRY RUN: No files will be renamed.")
        return True # Dry run is considered successful in terms of planning

    # --- Stage 1: Rename original to temporary ---
    logging.info("Stage 1: Renaming to temporary names...")
    success_stage1 = True
    renamed_to_temp = [] # Keep track of files renamed to temp for potential rollback
    try:
        for item in reversed(rename_plan): # Reverse might help avoid some collisions if not using unique temps
            shutil.move(item['old_full'], item['temp_full'])
            renamed_to_temp.append(item) # Record successful temp rename
    except Exception as e:
        logging.error(f"ERROR renaming '{item['old_base']}' to temp: {e}")
        success_stage1 = False
        logging.error("Attempting to rollback Stage 1 renames...")
        # Rollback: Rename temp files back to original
        for rolled_back_item in reversed(renamed_to_temp):
            try:
                shutil.move(rolled_back_item['temp_full'], rolled_back_item['old_full'])
                logging.info(f"Rolled back '{os.path.basename(rolled_back_item['temp_full'])}' to '{rolled_back_item['old_base']}'")
            except Exception as rb_e:
                logging.critical(f"CRITICAL ERROR during rollback! Failed to rename '{os.path.basename(rolled_back_item['temp_full'])}' back to '{rolled_back_item['old_base']}': {rb_e}")
                # At this point, the state is inconsistent. Manual intervention likely needed.
        return False # Indicate failure

    if not success_stage1: return False # Should be caught above

    # --- Stage 2: Rename temporary to final ---
    logging.info("Stage 2: Renaming temporary to final names...")
    success_stage2 = True
    renamed_to_final = [] # Keep track for potential rollback
    try:
        for item in rename_plan:
            shutil.move(item['temp_full'], item['final_full'])
            renamed_to_final.append(item)
    except Exception as e:
        logging.error(f"ERROR renaming temp '{os.path.basename(item['temp_full'])}' to final '{item['new_base']}': {e}")
        success_stage2 = False
        logging.error("Attempting to rollback Stage 2 renames (back to temporary)...")
        # Rollback: Rename final files back to temp. This leaves temp files.
        for rolled_back_item in reversed(renamed_to_final):
             try:
                 shutil.move(rolled_back_item['final_full'], rolled_back_item['temp_full'])
                 logging.info(f"Rolled back '{rolled_back_item['new_base']}' to temporary '{os.path.basename(rolled_back_item['temp_full'])}'")
             except Exception as rb_e:
                 logging.critical(f"CRITICAL ERROR during Stage 2 rollback! Failed to rename '{rolled_back_item['new_base']}' back to temp: {rb_e}")
        # Now, try to rollback Stage 1 as well (temp back to original)
        logging.error("Attempting to rollback Stage 1 renames (temporary back to original)...")
        for item in reversed(rename_plan): # All items were renamed to temp in stage 1 if we got here
             try:
                 # Check if the temp file still exists (it should if stage 2 rollback worked)
                 if os.path.exists(item['temp_full']):
                     shutil.move(item['temp_full'], item['old_full'])
                     logging.info(f"Rolled back temp '{os.path.basename(item['temp_full'])}' to original '{item['old_base']}'")
                 else:
                      logging.warning(f"Temp file '{os.path.basename(item['temp_full'])}' missing during full rollback.")
             except Exception as rb_e:
                 logging.critical(f"CRITICAL ERROR during full rollback! Failed to rename temp '{os.path.basename(item['temp_full'])}' back to original '{item['old_base']}': {rb_e}")
        return False # Indicate failure

    if success_stage1 and success_stage2:
        logging.info("Renaming completed successfully.")
        return True
    else:
        # This path should theoretically not be reached due to returns in except blocks, but as a safeguard:
        logging.error("Renaming process encountered errors and may have failed rollback.")
        return False


# --- Main Function to be Imported ---

def bypass_numerot(input_folder: str, dry_run: bool = False) -> str:
    """
    Analyzes images in a folder, potentially bypasses inference if the second image
    is white, otherwise runs inference to decide on a renaming scheme.
    Renames files according to the decision (000, 000a, 001... or 000, 001, 002...).
    Optionally creates a white page after negative renaming under specific conditions.

    Args:
        input_folder (str): Path to the folder containing images.
        dry_run (bool): If True, only logs planned actions without renaming.

    Returns:
        str: A status message indicating success ("success: positive_rename",
             "success: negative_rename", "success: dry_run_positive", etc.)
             or failure ("error: folder_not_found", "error: model_load",
             "error: inference_failed", "error: rename_failed", etc.).
    """
    if not os.path.isdir(input_folder):
        logging.error(f"Input folder not found: {input_folder}")
        return "error: folder_not_found"

    logging.info(f"--- Starting bypass_numerot for folder: {input_folder} ---")
    if dry_run:
        logging.info("*** DRY RUN MODE ENABLED ***")

    # --- Find, Sort, and Filter Image Files ---
    original_image_files_full_paths = []
    try:
        all_files = os.listdir(input_folder)
        for f in all_files:
            if f.lower().endswith(ALLOWED_EXTENSIONS):
                 original_image_files_full_paths.append(os.path.join(input_folder, f))
        original_image_files_full_paths = natsorted(original_image_files_full_paths)
    except Exception as e:
        logging.error(f"Error listing or sorting files in {input_folder}: {e}")
        return "error: file_listing_failed"


    if not original_image_files_full_paths:
        logging.warning(f"No supported image files found in {input_folder}. No action taken.")
        return "success: no_images_found"

    logging.info(f"Found {len(original_image_files_full_paths)} supported image files.")

    # --- White Page Bypass Check ---
    bypass_inference = False
    if len(original_image_files_full_paths) >= 2:
        second_image_path = original_image_files_full_paths[1]
        logging.info(f"Checking if second image '{os.path.basename(second_image_path)}' is white...")
        if _is_image_white(second_image_path):
            logging.info("Second image detected as white. Bypassing inference.")
            bypass_inference = True
        else:
            logging.info("Second image is not white. Proceeding with inference.")
    else:
        logging.info("Less than 2 images found, cannot perform white page bypass check. Proceeding.")


    final_result = 0 # Default to negative/zero result if inference is skipped/fails but not bypassed
    model = None
    transform = None
    device = None

    if not bypass_inference:
        # --- Load Model (only if inference is needed) ---
        logging.info("--- Loading Model ---")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")
        try:
            model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
            if not os.path.exists(MODEL_PATH):
                 logging.error(f"Model weights file not found at {MODEL_PATH}")
                 return "error: model_file_not_found"
            logging.info(f"Loading weights from: {MODEL_PATH}")
            checkpoint = torch.load(MODEL_PATH, map_location='cpu') # Load to CPU first
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif isinstance(checkpoint, dict) and 'model' in checkpoint: # Handle timm save format
                 state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            # Remove 'module.' prefix if present (from DataParallel saving)
            new_state_dict = { (k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items() }
            model.load_state_dict(new_state_dict)
            logging.info("Model weights loaded successfully.")
            model.to(device) # Move model to target device
            model.eval()
            config = resolve_data_config({}, model=model)
            inference_input_size = 384
            transform = create_transform(
                input_size=inference_input_size,
                is_training=False,
                mean=config['mean'],         # Use resolved mean
                std=config['std'],           # Use resolved std
                interpolation=config.get('interpolation', 'bicubic'), # Use resolved or default interpolation
                crop_mode='center',          # Explicitly match training validation
                crop_pct=1.0                 # Explicitly match training validation
            )

        except Exception as e:
            logging.exception(f"Error loading model or creating transform: {e}") # Log traceback
            return "error: model_load_failed"

        # --- Prepare for Inference ---
        required_images = 1 + 2 * NUM_PAIRS + 1 # Index 2*NUM_PAIRS+1 needed for shifted batch
        if len(original_image_files_full_paths) < required_images:
             logging.warning(f"Not enough images for full inference. Found {len(original_image_files_full_paths)}, need {required_images} for {NUM_PAIRS} shifted pairs.")
             # Decide how to handle this: proceed with fewer pairs, or default to negative?
             # Let's default to negative renaming if inference can't run fully.
             logging.warning("Proceeding with default negative renaming due to insufficient images for inference.")
             final_result = 0 # Ensure negative path is taken
             bypass_inference = True # Skip the actual inference run
        else:
            # --- Prepare Batch 1 (Normal Pairs: 1-2, 3-4, ...) ---
            logging.info("Preparing Batch 1 (Normal Pairs)")
            batch1_merged = []
            for i in range(NUM_PAIRS):
                idx1 = 1 + i * 2
                idx2 = 2 + i * 2
                # No need to check idx2 >= len, loop range handles it implicitly if required_images check passed
                file1 = original_image_files_full_paths[idx1]
                file2 = original_image_files_full_paths[idx2]
                merged = _merge_images_side_by_side(file1, file2)
                if merged:
                    batch1_merged.append(merged)
                else:
                    logging.warning(f"Failed to merge pair {i+1} ({os.path.basename(file1)}, {os.path.basename(file2)}). Skipping.")

            # --- Prepare Batch 2 (Shifted Pairs: 2-3, 4-5, ...) ---
            logging.info("Preparing Batch 2 (Shifted Pairs)")
            batch2_merged = []
            for i in range(NUM_PAIRS):
                idx1 = 2 + i * 2
                idx2 = 3 + i * 2
                # No need to check idx2 >= len here either
                file1 = original_image_files_full_paths[idx1]
                file2 = original_image_files_full_paths[idx2]
                merged = _merge_images_side_by_side(file1, file2)
                if merged:
                    batch2_merged.append(merged)
                else:
                    logging.warning(f"Failed to merge shifted pair {i+1} ({os.path.basename(file1)}, {os.path.basename(file2)}). Skipping.")

            if not batch1_merged or not batch2_merged:
                logging.error("Cannot proceed with inference as at least one batch is empty after merging attempts.")
                # Defaulting to negative renaming seems safest if inference prep fails
                final_result = 0
                bypass_inference = True # Skip inference run
            else:
                # --- Run Inference ---
                logging.info("--- Running Inference ---")
                preds1 = _run_inference(model, batch1_merged, transform, device)
                preds2 = _run_inference(model, batch2_merged, transform, device)

                if preds1.size == 0 or preds2.size == 0:
                    logging.error("Inference did not produce results for one or both batches. Defaulting to negative renaming.")
                    final_result = 0 # Default to negative
                    # No need to set bypass_inference=True here, we are past that stage
                else:
                    # --- Calculate Scores ---
                    logging.info("--- Calculating Scores ---")
                    scores1 = np.where(preds1 == 0, -1, 1)
                    scores2 = np.where(preds2 == 0, -1, 1)
                    sum1 = np.sum(scores1)
                    sum2 = np.sum(scores2)
                    final_result = sum1 - sum2

                    logging.info(f"Batch 1 Sum: {sum1}, Batch 2 Sum: {sum2}, Final Result (Sum1 - Sum2): {final_result}")
                    # --- End of Inference Calculation ---

    # --- Renaming Logic ---
    new_basenames = []
    rename_decision = ""

    if bypass_inference or final_result > 0:
        rename_decision = "positive"
        logging.info("Decision: Positive result or white page bypass. Applying 000, 000a, 001, 002... renaming.")
        if not original_image_files_full_paths:
             logging.warning("No files to rename.")
        else:
            _, ext = os.path.splitext(original_image_files_full_paths[0])
            ext = ext.lower()
            new_basenames.append(f"000{ext}")
            if len(original_image_files_full_paths) > 1:
                new_basenames.append(f"000a{ext}")
            for i in range(2, len(original_image_files_full_paths)):
                new_basenames.append(f"{i-1:03d}{ext}")

    else: # final_result <= 0
        rename_decision = "negative"
        logging.info("Decision: Negative or zero result. Applying 000, 001, 002... renaming.")
        if not original_image_files_full_paths:
             logging.warning("No files to rename.")
        else:
            _, ext = os.path.splitext(original_image_files_full_paths[0])
            ext = ext.lower()
            for i in range(len(original_image_files_full_paths)):
                new_basenames.append(f"{i:03d}{ext}")

    # --- Execute Renaming ---
    rename_successful = False
    if new_basenames: # Only attempt rename if there's a plan
        rename_successful = _rename_files_safely(input_folder, original_image_files_full_paths, new_basenames, dry_run=dry_run)
    elif original_image_files_full_paths:
         logging.warning("Rename plan is empty, but files exist. Skipping rename execution.")
         # This case shouldn't happen with current logic, but good to note.
         rename_successful = True # Consider it "successful" as no rename was needed/possible
    else:
         # No files found initially, so no renaming needed.
         rename_successful = True # Successful in the sense that nothing needed doing.


    # --- Post-Renaming Actions & Return Status ---
    if not rename_successful and not dry_run:
        # Renaming failed during execution (rollback might have also failed)
        return f"error: rename_failed_{rename_decision}"

    final_status_suffix = f"_{rename_decision}" if rename_decision else ""
    if dry_run:
        return f"success: dry_run{final_status_suffix}"

    # If rename was successful (or skipped because no files) and not dry run:
    if rename_decision == "negative":
        # Only call white page creation after successful negative rename
        logging.info("Negative rename complete. Checking if white page creation is needed.")
        white_page_ok = _create_white_page_if_needed(input_folder)
        if not white_page_ok:
            # Log the error, but maybe don't make the whole process fail?
            # Or return a specific status?
            logging.error("White page creation failed after negative rename.")
            # Decide: return "warning: white_page_failed" or stick with success?
            # Let's stick with success but log the error. The primary task (rename) succeeded.

    logging.info(f"--- bypass_numerot finished for folder: {input_folder} ---")
    return f"success:{final_status_suffix}" # e.g., "success:_positive", "success:_negative"

# --- END OF FILE renomage_logic.py ---