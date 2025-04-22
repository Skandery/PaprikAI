import os
import re
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError # Added UnidentifiedImageError
from collections import Counter
import shutil # Using shutil for safer directory operations
import logging
import natsort

logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')


# --- Configuration ---
# Percentage of image height from the bottom to scan for page number
OCR_BOTTOM_CROP_PERCENT = 0.10 # 10% like original script
# Grayscale threshold value (adjust if needed for different lighting/scan quality)
OCR_THRESHOLD_VALUE = 150
# Minimum number of identical gaps needed to stop early and confirm the gap
MIN_GAP_CONFIRMATIONS = 5
# Minimum number of files needed to reliably determine the gap
MIN_FILES_FOR_GAP = 3
# Maximum allowed gap value (prevents unreasonable shifts)
MAX_ALLOWED_GAP = 10 # Increased slightly from original 5, adjust as needed
# Aspect ratio threshold for detecting double pages (Width / Height)
DP_ASPECT_RATIO_THRESHOLD = 1.5 # Adjust if needed (e.g., 1.4 for less wide DPs)
# Image file extensions to process
SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp') # Added webp

# --- Helper Functions ---

def _initialize_easyocr_reader():
    """Initializes and returns the EasyOCR reader."""
    try:
        import easyocr # Import here to make it optional if only using other parts
        logging.debug("Initializing EasyOCR Reader (this may take a moment)...")
        # Try GPU first
        try:
            reader = easyocr.Reader(['en'], gpu=True)
            logging.info("EasyOCR Reader initialized with GPU.")
            return reader
        except Exception as e_gpu:
            logging.error(f"Could not initialize EasyOCR with GPU: {e_gpu}")
            logging.error("Attempting to initialize EasyOCR Reader with CPU...")
            try:
                reader = easyocr.Reader(['en'], gpu=False)
                logging.warning("EasyOCR Reader initialized with CPU.")
                return reader
            except Exception as e_cpu:
                logging.error(f"Error initializing EasyOCR reader with CPU: {e_cpu}")
                return None
    except ImportError:
        logging.error("Error: easyocr library is not installed. Please install it: pip install easyocr")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during EasyOCR initialization: {e}")
        return None

def _read_image_robust(image_path):
    """Reads an image file, handling potential path encoding issues."""
    try:
        # Read file as bytes
        with open(image_path, 'rb') as f:
            file_bytes = f.read()
        # Decode using OpenCV
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            logging.warning(f"Warning: OpenCV could not decode image: {os.path.basename(image_path)}")
            return None
        return image
    except FileNotFoundError:
        logging.error(f"Warning: File not found: {os.path.basename(image_path)}")
        return None
    except Exception as e:
        logging.error(f"Error reading image {os.path.basename(image_path)}: {e}")
        return None

def _extract_page_number_easyocr(image_path, reader):
    """Extract the page number from the bottom region using EasyOCR."""
    image = _read_image_robust(image_path)
    if image is None:
        return None

    try:
        height, width, _ = image.shape
        # Define the bottom region based on percentage
        top_crop = int((1.0 - OCR_BOTTOM_CROP_PERCENT) * height)
        bottom_region_cv = image[top_crop:, :]

        # --- Preprocessing for EasyOCR ---
        gray_cv = cv2.cvtColor(bottom_region_cv, cv2.COLOR_BGR2GRAY)
        threshold_np = np.where(gray_cv > OCR_THRESHOLD_VALUE, 255, 0).astype(np.uint8)

        # --- OCR ---
        result = reader.readtext(threshold_np, detail=0, paragraph=False)

        # --- Number Extraction ---
        numbers_found = []
        number_pattern = re.compile(r'\d+')
        for text_line in result:
            # Normalize text slightly (remove common OCR noise like '|')
            cleaned_line = text_line.replace('|', '').strip()
            numbers_in_line = number_pattern.findall(cleaned_line)
            for num_str in numbers_in_line:
                try:
                    num = int(num_str)
                    # Filter out potentially huge numbers or single digits likely noise
                    if 0 < num < 5000: # Adjusted range, consider context
                        numbers_found.append(num)
                except ValueError:
                    continue

        if numbers_found:
            # Prioritize smaller numbers if multiple are found? Or most frequent?
            # For now, return the first valid number found.
            # logging.info(f"Debug OCR: {os.path.basename(image_path)} -> Found numbers: {numbers_found} -> Using: {numbers_found[0]}")
            return numbers_found[0]
        else:
            # logging.info(f"Debug: No numbers found by EasyOCR in {os.path.basename(image_path)}")
            return None

    except Exception as e:
        logging.error(f"Error extracting page number from {os.path.basename(image_path)}: {e}")
        return None

def _extract_filename_index(filename):
    """Extract the leading numeric index from the filename (including negative)."""
    try:
        name_without_ext = os.path.splitext(filename)[0]
        # Match optional hyphen followed by digits at the start
        match = re.match(r'^(-?\d+)', name_without_ext)
        if match:
            return int(match.group(1))
        # Handle the '000a', '000b' case from correction step
        match_alpha = re.match(r'^(0+)([a-z])$', name_without_ext, re.IGNORECASE)
        if match_alpha:
             # Treat '000a' as index 0 for sorting/comparison purposes,
             # but it's special. Maybe return 0.1, 0.2? Or handle separately.
             # Let's return None for now to avoid renaming these further unless needed.
             return None # Or a special value if needed later
        return None
    except Exception as e:
        logging.error(f"Error extracting index from {filename}: {e}")
        return None

def _get_image_files(folder_path):
    """Gets a naturally sorted list of image files in the folder."""
    try:
        all_files = os.listdir(folder_path)
        image_files_unsorted = [
            f for f in all_files
            if f.lower().endswith(SUPPORTED_EXTENSIONS)
            and not f.startswith('.') # Ignore hidden files
            and not f.startswith('temp_rename_') # Ignore temp files
            # Optional: Normalize Unicode filenames (NFC form is common)
            # f = unicodedata.normalize('NFC', f)
        ]
        # Use natsort if available for better sorting (e.g., 1, 2, 10)
        image_files = natsort.natsorted(image_files_unsorted)
        return image_files
    except Exception as e:
        logging.error(f"Error listing files in {folder_path}: {e}")
        return []

def _perform_rename(folder_path, rename_map, operation_name="main"):
    """
    Safely renames files using a temporary directory.
    Handles potential source/target conflicts by moving all involved files.
    """
    if not rename_map:
        logging.warning(f"No files to rename for '{operation_name}' operation.")
        return True

    logging.info(f"Performing '{operation_name}' renaming...")
    # logging.info(f"Rename map ({len(rename_map)}): {rename_map}") # Debug

    # --- Pre-check for direct conflicts (multiple sources mapping to the same target) ---
    target_counts = Counter(rename_map.values())
    duplicates = [name for name, count in target_counts.items() if count > 1]
    if duplicates:
        logging.error(f"Error: Renaming would cause direct conflicts. Multiple files target: {duplicates}")
        # Find which source files map to the duplicates
        conflicting_sources = {target: [] for target in duplicates}
        for old, new in rename_map.items():
            if new in conflicting_sources:
                conflicting_sources[new].append(old)
        for target, sources in conflicting_sources.items():
             logging.info(f"  - Target '{target}' is targeted by: {sources}")
        logging.error(f"Aborting '{operation_name}' rename operation.")
        return False

    temp_dir = os.path.join(folder_path, f"temp_rename_dir_{operation_name}")
    moved_to_temp = {} # Store mapping from original name to temp path

    try:
        os.makedirs(temp_dir, exist_ok=True)

        # --- Identify all files involved ---
        # Files being renamed (sources)
        source_files = set(rename_map.keys())
        # Files that are targets of the rename
        target_files = set(rename_map.values())
        # Existing files in the folder
        current_files = set(_get_image_files(folder_path))

        # Files to move: sources + any existing file whose name is a target
        files_to_move = source_files.union(target_files.intersection(current_files))

        # logging.info(f"Debug: Files to move to temp: {files_to_move}") # Debug

        # --- Move involved files to temporary directory ---
        for filename in files_to_move:
            old_path = os.path.join(folder_path, filename)
            # Use a unique temp name in case of case-insensitivity issues or future complex maps
            # temp_filename = f"{uuid.uuid4()}_{filename}" # Safer but less debuggable
            temp_path = os.path.join(temp_dir, filename) # Keep original name in temp dir

            if os.path.exists(old_path):
                try:
                    shutil.move(old_path, temp_path)
                    moved_to_temp[filename] = temp_path
                    # logging.info(f"Moved to temp: {filename}") # Debug
                except Exception as move_err:
                    logging.error(f"Error moving {filename} to temp: {move_err}. Attempting rollback.")
                    raise # Re-raise to trigger rollback in outer try/except
            else:
                # This can happen if a file is listed as a target but doesn't actually exist
                # Or if a source file disappeared between listing and moving
                if filename in source_files:
                    logging.warning(f"Warning: Source file {filename} not found during move to temp.")
                # else: # It was a target file that didn't exist, which is fine
                #    pass

        # --- Move files back with new names ---
        success_count = 0
        rename_errors = False
        for old_name, new_name in rename_map.items():
            temp_path = moved_to_temp.get(old_name) # Get temp path using original name

            if temp_path and os.path.exists(temp_path):
                new_path = os.path.join(folder_path, new_name)

                # Double-check for conflict *before* moving back (shouldn't happen with temp dir logic, but safety)
                if os.path.exists(new_path):
                    logging.error(f"CRITICAL WARNING: Target path {new_path} exists unexpectedly before moving back {old_name} as {new_name}. Skipping this rename.")
                    rename_errors = True
                    continue # Skip this specific rename

                try:
                    shutil.move(temp_path, new_path)
                    logging.debug(f"Renamed ({operation_name}): {old_name} -> {new_name}")
                    success_count += 1
                    # Mark as moved back by removing from moved_to_temp
                    del moved_to_temp[old_name]
                except Exception as move_back_err:
                    logging.error(f"Error moving {temp_path} back to {new_path}: {move_back_err}")
                    rename_errors = True
                    # Leave the file in the temp dir for manual recovery

            elif old_name in source_files: # Only warn if it was a source file we expected to move
                logging.error(f"Warning: Temp file for {old_name} not found for renaming to {new_name}.")
                rename_errors = True


        # --- Move back any remaining files that were moved but not renamed ---
        # These are files that were targets but not sources, moved for safety.
        for original_name, temp_path in list(moved_to_temp.items()):
             if os.path.exists(temp_path):
                 original_path = os.path.join(folder_path, original_name)
                 if not os.path.exists(original_path):
                     try:
                         shutil.move(temp_path, original_path)
                         # logging.info(f"Restored non-renamed file: {original_name}") # Debug
                         del moved_to_temp[original_name] # Mark as restored
                     except Exception as restore_err:
                         logging.error(f"Error restoring non-renamed file {original_name} from temp: {restore_err}")
                         rename_errors = True
                 else:
                     logging.error(f"Warning: Original path {original_path} exists. Cannot restore {original_name} from temp automatically.")
                     rename_errors = True


        if rename_errors:
             logging.error(f"Completed '{operation_name}' rename with {success_count} successes and errors.")
             # If errors occurred, do not automatically delete the temp dir
             logging.error(f"Warning: Errors occurred. Please check the temporary directory: {temp_dir}")
             return False
        else:
             logging.info(f"Successfully renamed {success_count} files for '{operation_name}'.")
             return True

    except Exception as e:
        logging.error(f"Error during '{operation_name}' renaming process: {e}")
        # Attempt to move files back from temp dir if error occurred during the process
        logging.error("Attempting to restore files from temporary directory...")
        restore_errors = False
        try:
            if os.path.exists(temp_dir):
                for filename in os.listdir(temp_dir):
                    temp_path = os.path.join(temp_dir, filename)
                    # Attempt to restore to original name (best guess)
                    original_path = os.path.join(folder_path, filename)
                    if os.path.isfile(temp_path): # Ensure it's a file
                        if not os.path.exists(original_path): # Avoid overwriting
                            try:
                                shutil.move(temp_path, original_path)
                                logging.warning(f"Restored {filename}")
                            except Exception as move_err:
                                logging.error(f"Error restoring {filename}: {move_err}")
                                restore_errors = True
                        else:
                            logging.info(f"Warning: Could not restore {filename}, target exists.")
                            restore_errors = True
        except Exception as restore_e:
            logging.error(f"Error during restore attempt: {restore_e}")
            restore_errors = True

        if restore_errors:
            logging.error(f"Restore incomplete. Please check the temporary directory: {temp_dir}")
        return False # Indicate failure

    finally:
        # Clean up temporary directory ONLY if the process was successful and no files remain
        if 'rename_errors' in locals() and not rename_errors and not moved_to_temp and os.path.exists(temp_dir):
             try:
                 # Check if empty before removing
                 if not os.listdir(temp_dir):
                     os.rmdir(temp_dir)
                 else:
                     logging.warning(f"Warning: Temporary directory {temp_dir} is not empty after successful operation. Manual cleanup needed.")
             except OSError as rm_err:
                 logging.warning(f"Warning: Could not remove temporary directory {temp_dir}: {rm_err}")
        elif os.path.exists(temp_dir):
             # If rename failed or files are left, leave the temp dir
             logging.warning(f"Leaving temporary directory for inspection: {temp_dir}")


def _correct_negative_and_zero_indices(folder_path):
    """
    Corrects file numbering for files indexed < 1 after main rename.
    First file (numerically, including negatives) becomes 000.jpg.
    Subsequent files originally < 1 become 000a.jpg, 000b.jpg, etc.
    Handles potential conflicts with existing files.
    """
    logging.debug(f"Correcting indices < 1 in folder: {folder_path}")
    image_files = _get_image_files(folder_path)

    file_numbers = []
    files_without_index = []
    for filename in image_files:
        number = _extract_filename_index(filename)
        if number is not None:
            file_numbers.append({'name': filename, 'index': number})
        else:
            # Keep track of files like 'cover.jpg' or '000a.jpg'
            files_without_index.append(filename)

    if not file_numbers:
        logging.warning("No valid numbered files found for correction.")
        return True # Nothing to do

    # Sort by the numeric index
    file_numbers.sort(key=lambda x: x['index'])

    # Find files with index < 1
    files_to_correct = [f for f in file_numbers if f['index'] < 1]

    if not files_to_correct:
        logging.debug("No files with index < 1 found. Correction step skipped.")
        return True

    logging.error(f"Found {len(files_to_correct)} files with index < 1 to correct.")

    # Determine target names
    rename_map_correct = {}
    # Get names of files that *won't* be corrected (index >= 1) + non-indexed files
    existing_target_names = set(f['name'] for f in file_numbers if f['index'] >= 1)
    existing_target_names.update(files_without_index)

    # --- Determine new names for files < 1 ---
    # The numerically first file (most negative or zero) becomes 000.jpg
    first_file_info = files_to_correct[0]
    target_000 = "000.jpg"
    rename_map_correct[first_file_info['name']] = target_000
    potential_targets = {target_000} # Track targets generated in this step

    # Subsequent files become 000a, 000b, ...
    for i, file_info in enumerate(files_to_correct[1:]):
        suffix = chr(ord('a') + i)
        target_name = f"000{suffix}.jpg" # Assuming jpg, might need to preserve original ext
        # Preserve original extension
        _, ext = os.path.splitext(file_info['name'])
        target_name = f"000{suffix}{ext.lower()}" # Use lowercase extension
        rename_map_correct[file_info['name']] = target_name
        potential_targets.add(target_name)

    # --- Check for conflicts ---
    conflicts = False
    # Check if generated targets conflict with each other (shouldn't happen with a,b,c logic)
    # Check if generated targets conflict with existing files (index >= 1 or non-indexed)
    for old_name, new_name in rename_map_correct.items():
        if new_name in existing_target_names:
            logging.error(f"Error: Correction conflict. Target name '{new_name}' (from '{old_name}') already exists as a file not being corrected.")
            conflicts = True
        # Check if a file being corrected targets the *original* name of another file being corrected
        # (e.g. -1 -> 000.jpg, but 0.jpg exists and is being corrected to 000a.jpg)
        # The improved _perform_rename should handle this via the temp dir.

    if conflicts:
        logging.error("Aborting correction due to naming conflicts.")
        return False

    logging.info("Correction rename map:", rename_map_correct)

    # Perform the renaming using the safe rename helper
    return _perform_rename(folder_path, rename_map_correct, operation_name="correction")


def _handle_double_pages(folder_path):
    """
    Detects double pages (landscape aspect ratio) and renames them.
    Example: 002.jpg (exists), 003.jpg (is DP) -> renames 003.jpg to 002-003.jpg, deletes 002.jpg
    Runs *after* main rename and correction steps.
    """
    logging.debug(f"Checking for double pages (DP) in: {folder_path}")
    image_files = _get_image_files(folder_path)

    files_data = []
    for filename in image_files:
        number = _extract_filename_index(filename)
        # We only care about positively numbered files for DP logic
        if number is not None and number >= 0:
            files_data.append({'name': filename, 'index': number})
        # Silently ignore files like 000a.jpg or cover.jpg for DP processing

    if not files_data:
        logging.error("No suitable numbered files found for DP check.")
        return True

    # Sort by index
    files_data.sort(key=lambda x: x['index'])

    dp_rename_map = {}
    files_to_delete = set()
    potential_targets = set() # Track generated DP names

    # Create a quick lookup map from index to filename
    index_to_name = {f['index']: f['name'] for f in files_data}

    logging.debug(f"Analyzing {len(files_data)} numbered files for DP...")
    for i, current_file in enumerate(files_data):
        current_name = current_file['name']
        current_index = current_file['index']

        # Skip files already marked for deletion (e.g., the first page of a DP)
        if current_name in files_to_delete:
            continue

        file_path = os.path.join(folder_path, current_name)
        try:
            with Image.open(file_path) as img:
                width, height = img.size
        except FileNotFoundError:
            logging.error(f"Warning: File not found during DP check: {current_name}")
            continue
        except UnidentifiedImageError:
            logging.error(f"Warning: Cannot identify image file (corrupt?): {current_name}")
            continue
        except Exception as e:
            logging.error(f"Error reading image dimensions for {current_name}: {e}")
            continue

        if height == 0: continue # Avoid division by zero

        aspect_ratio = width / height

        # Check if it's a double page
        if aspect_ratio > DP_ASPECT_RATIO_THRESHOLD:
            logging.info(f"  - Detected DP: {current_name} (Ratio: {aspect_ratio:.2f})")

            # Assume DP represents pages index-1 and index
            # Requires the *previous* page (index-1) to exist
            prev_index = current_index - 1
            if prev_index < 0:
                logging.warning(f"    Warning: DP '{current_name}' seems to be the first page (index {current_index}). Cannot form 'prev-current' name.")
                continue # Skip renaming this DP if it's page 0

            prev_filename = index_to_name.get(prev_index)

            if prev_filename is None:
                logging.warning(f"    Warning: Previous page file (index {prev_index}) not found for DP '{current_name}'. Cannot rename or delete previous.")
                # Decide: Rename anyway? Or skip? Let's skip for safety.
                continue
            elif prev_filename in files_to_delete:
                 logging.info(f"    Info: Previous page '{prev_filename}' already marked for deletion by another DP. Skipping DP rename for '{current_name}'.")
                 continue


            # Format new DP name
            _, ext = os.path.splitext(current_name)
            dp_name = f"{prev_index:03d}-{current_index:03d}{ext.lower()}"

            # Check for conflicts before adding to map
            if dp_name in potential_targets:
                 logging.error(f"    Error: DP target name '{dp_name}' conflicts with another generated DP name. Skipping rename for '{current_name}'.")
                 continue
            # Check if target DP name conflicts with an existing file *not* being deleted/renamed
            # (Unlikely if numbering is sequential, but possible)
            existing_files_check = set(index_to_name.values()) - files_to_delete - set(dp_rename_map.keys())
            if dp_name in existing_files_check:
                 logging.error(f"    Error: DP target name '{dp_name}' conflicts with existing file. Skipping rename for '{current_name}'.")
                 continue


            # Add rename operation for the DP file
            dp_rename_map[current_name] = dp_name
            potential_targets.add(dp_name)
            # Mark the previous page file for deletion
            files_to_delete.add(prev_filename)
            logging.info(f"    Action: Will rename '{current_name}' to '{dp_name}' and delete '{prev_filename}'.")


    # --- Perform DP Renames ---
    rename_success = True
    if dp_rename_map:
        rename_success = _perform_rename(folder_path, dp_rename_map, operation_name="dp_rename")
    else:
        logging.debug("No DP renames required.")

    if not rename_success:
        logging.error("DP renaming failed. Skipping deletion step.")
        return False # Indicate failure

    # --- Perform Deletions ---
    delete_success_count = 0
    delete_errors = False
    if files_to_delete:
        logging.warning(f"Deleting {len(files_to_delete)} previous pages for DPs...")
        for filename_to_delete in files_to_delete:
            file_path_to_delete = os.path.join(folder_path, filename_to_delete)
            if os.path.exists(file_path_to_delete):
                try:
                    os.remove(file_path_to_delete)
                    logging.warning(f"  - Deleted: {filename_to_delete}")
                    delete_success_count += 1
                except Exception as e:
                    logging.warning(f"  - Error deleting {filename_to_delete}: {e}")
                    delete_errors = True
            else:
                logging.warning(f"  - Warning: File marked for deletion not found: {filename_to_delete}")
                # This might happen if it was already deleted or involved in a failed rename

    if delete_errors:
        logging.error("Errors occurred during DP previous page deletion.")
        return False # Indicate partial failure
    elif files_to_delete:
         logging.warning(f"Successfully deleted {delete_success_count} files.")

    return True # DP handling completed


def _create_white_page_if_needed(folder_path):
    """
    Creates a white image placeholder based on specific conditions of the *second* image file
    found *after* all renaming and DP handling.
    Also ensures the first image is named '000.jpg'.

    Conditions: Second file is 001.jpg, 003.jpg, etc. (odd numbers)
    Creates: 000a.jpg, 002.jpg, etc. respectively.
    NOTE: This logic might be fragile if DP handling significantly changes numbering.
    """
    logging.info(f"Checking for white page creation in: {folder_path}")
    try:
        # Get image files *after* all processing
        image_files = _get_image_files(folder_path)
        
        try:
            import natsort  # Optional dependency
            image_files = natsort.natsorted(image_files)
            logging.debug("Using natural sort for white page check.")
        except ImportError:
            image_files.sort()
            logging.info("Using basic sort for white page check (install natsort for better ordering).")

        if len(image_files) < 2:
            logging.info("Folder has less than 2 images after processing. Skipping white page creation.")
            return True

        first_image_name = image_files[0]
        second_image_name = image_files[1]
        logging.debug(f"First image: {first_image_name}, Second image: {second_image_name}")

        # Rename the first image to '000.jpg' if it’s not already
        if first_image_name.lower() != '000.jpg':
            original_path = os.path.join(folder_path, first_image_name)
            new_path = os.path.join(folder_path, '000.jpg')
            
            # If '000.jpg' already exists, warn and skip renaming
            if os.path.exists(new_path):
                logging.warning(f"'000.jpg' already exists. Skipping renaming of '{first_image_name}'.")
            else:
                os.rename(original_path, new_path)
                logging.debug(f"Renamed '{first_image_name}' to '000.jpg'.")
                first_image_name = '000.jpg'  # Update variable to reflect rename

        # Now continue with white page logic
        target_white_page = None
        required_white_page_name = None

        second_name_lower = second_image_name.lower()
        if second_name_lower == '001.jpg':
            required_white_page_name = '000a.jpg'
        else:
            match = re.match(r'(\d+)\.(jpg|jpeg|png|webp)$', second_name_lower)
            if match:
                num = int(match.group(1))
                ext = match.group(2)
                if num > 0 and num % 2 != 0:
                    required_white_page_name = f"{num-1:03d}.{ext}"

        if required_white_page_name:
            white_image_path = os.path.join(folder_path, required_white_page_name)

            if os.path.exists(white_image_path):
                logging.debug(f"Required white page '{required_white_page_name}' already exists. Skipping creation.")
                return True

            ref_image_path = os.path.join(folder_path, first_image_name)
            logging.debug(f"Condition met. Need to create '{required_white_page_name}'. Using '{first_image_name}' for dimensions.")

            try:
                with Image.open(ref_image_path) as img:
                    width, height = img.size

                white_image = Image.new('RGB', (width, height), color='white')
                white_image.save(white_image_path)
                logging.debug(f"Created white page '{required_white_page_name}' (size {width}x{height}).")
                return True

            except FileNotFoundError:
                logging.error(f"Error: Could not find reference image {ref_image_path} to get dimensions.")
                return False
            except UnidentifiedImageError:
                logging.error(f"Error: Could not read reference image {ref_image_path} (corrupt?).")
                return False
            except Exception as e:
                logging.error(f"Error creating white page {required_white_page_name}: {e}")
                return False
        else:
            return True

    except Exception as e:
        logging.error(f"An unexpected error occurred during white page check: {e}")
        return False


# --- Main Function ---

def numerotation(folderpath):
    """
    Processes a folder of manga pages:
    1. Extracts page numbers using EasyOCR.
    2. Determines the most common difference (gap) between filename index and OCR page number.
    3. Renames files based on the calculated gap (e.g., file 001.jpg + gap 2 -> 003.jpg).
    4. Corrects files with original indices < 1 (e.g., -1.jpg, 0.jpg) to 000.jpg, 000a.jpg, etc.
    5. Detects and renames double pages (e.g., 003.jpg -> 002-003.jpg, deletes 002.jpg).
    6. Creates specific white placeholder pages if certain conditions are met after renaming.

    Args:
        folderpath (str): The path to the folder containing manga image files.

    Returns:
        bool: True if the process completed successfully (or with minor warnings),
              False if a critical error occurred preventing completion.
    """
    logging.info(f"\n--- Starting Numerotation for Folder: {folderpath} ---")
    if not os.path.isdir(folderpath):
        logging.error(f"Error: Folder not found: {folderpath}")
        return False

    # --- 1. Initialize OCR ---
    reader = _initialize_easyocr_reader()
    if reader is None:
        logging.error("Error: EasyOCR Reader could not be initialized. Aborting.")
        return False

    # --- 2. List image files ---
    image_files = _get_image_files(folderpath)
    logging.info(f"Found {len(image_files)} image files.")
    if not image_files:
        logging.error("No image files found to process.")
        return True # Completed successfully as there's nothing to do

    # --- 3. Calculate gap using EasyOCR ---
    filename_to_gap = {}
    gap_counter = Counter()
    processed_files_count = 0
    logging.debug("Analyzing page numbers (using EasyOCR)...")
    for filename in image_files:
        file_path = os.path.join(folderpath, filename)
        filename_index = _extract_filename_index(filename)

        # Only process files with a simple numeric index for gap calculation
        if filename_index is not None:
            page_number = _extract_page_number_easyocr(file_path, reader)
            if page_number is not None:
                gap = page_number - filename_index
                filename_to_gap[filename] = gap
                gap_counter[gap] += 1
                processed_files_count += 1
                # logging.info(f"  - {filename}: Index={filename_index}, OCR Page={page_number}, Gap={gap}") # Verbose

                # Stop early if a gap is confirmed
                most_common_gap_val, most_common_gap_count = gap_counter.most_common(1)[0]
                if most_common_gap_count >= MIN_GAP_CONFIRMATIONS:
                    logging.info(f"Confirmed gap {most_common_gap_val} after {processed_files_count} files ({MIN_GAP_CONFIRMATIONS} matches). Stopping analysis.")
                    break
            # else: logging.info(f"  - {filename}: Index={filename_index}, OCR Page=Not Found") # Debug
        # else: logging.info(f"  - {filename}: Index=Not Found (Skipping for gap calc)") # Debug

    if processed_files_count < MIN_FILES_FOR_GAP:
        logging.warning(f"Warning: Not enough files with valid index and OCR page number found ({processed_files_count} found, need at least {MIN_FILES_FOR_GAP}). Cannot reliably determine gap.")
        # Decide whether to proceed without gap adjustment or abort. Let's try proceeding without adjustment.
        logging.info("Proceeding without gap adjustment.")
        most_common_gap = 0 # Assume no gap if calculation failed
    elif not gap_counter:
         logging.warning(f"Warning: No gaps could be calculated from the {processed_files_count} processed files.")
         logging.warning("Proceeding without gap adjustment.")
         most_common_gap = 0 # Assume no gap
    else:
        # Determine the most common gap
        most_common_gap_info = gap_counter.most_common(1)
        most_common_gap, common_count = most_common_gap_info[0]
        logging.info(f"Most common gap found: {most_common_gap} (occurred {common_count} times out of {processed_files_count} processed)")

        # Check if the determined gap is reasonable
        if abs(most_common_gap) > MAX_ALLOWED_GAP:
            logging.error(f"Error: Calculated gap ({most_common_gap}) exceeds the maximum allowed ({MAX_ALLOWED_GAP}). This might indicate a significant OCR or naming issue.")
            logging.error("Please check the files manually. Aborting rename.")
            return False

    # --- 4. Prepare and perform main renaming based on gap ---
    rename_map_main = {}
    logging.info(f"Preparing main rename map using gap: {most_common_gap}")
    # Use the potentially natsorted list from _get_image_files
    current_image_files = _get_image_files(folderpath) # Get fresh list before renaming
    for filename in current_image_files:
        filename_index = _extract_filename_index(filename)
        if filename_index is not None:
            # Calculate new page number based on original index and the common gap
            new_page_number = filename_index + most_common_gap
            _, ext = os.path.splitext(filename)
            ext = ext.lower() # Use lowercase extension

            # Format new name (handle negatives temporarily)
            if new_page_number < 0:
                 # Use at least 2 digits for negative numbers for consistency? e.g., -01, -02...
                 new_filename = f"-{abs(new_page_number):02d}{ext}"
            else:
                 # Keep 3-digit padding for positive numbers
                 new_filename = f"{new_page_number:03d}{ext}"

            # Only add to map if the name actually changes
            if filename != new_filename:
                # !!! REMOVED THE CHECK HERE !!!
                # The _perform_rename function is designed to handle conflicts
                # using the temporary directory, so we don't need to check os.path.exists here.
                rename_map_main[filename] = new_filename
                # logging.debug(f"Planning rename: {filename} -> {new_filename}") # Optional debug
        # else: Keep files without index as they are for now

    # This check should be sufficient before calling perform_rename
    if not rename_map_main:
        logging.info("No file renames required based on the calculated gap.")
        # Skip calling _perform_rename if the map is empty
        rename_step_ok = True # Nothing to do, so it's "ok"
    else:
        rename_step_ok = _perform_rename(folderpath, rename_map_main, operation_name="main_gap_adjust")

    if not rename_step_ok:
        logging.error("Main renaming based on gap failed. Aborting further steps.")
        return False

    # --- 5. Correct indices < 1 (run this *after* the main rename) ---
    correction_step_ok = _correct_negative_and_zero_indices(folderpath)
    if not correction_step_ok:
        logging.error("Correction of indices < 1 failed. Aborting further steps.")
        # This step is often crucial for subsequent steps, so abort if it fails.
        return False

    # --- 6. Handle Double Pages (run *after* main rename and correction) ---
    dp_step_ok = _handle_double_pages(folderpath)
    if not dp_step_ok:
        logging.error("Handling of double pages failed or had errors. Aborting further steps.")
        # DP handling involves deletion, safer to abort if it fails.
        return False

    # --- 7. Create white page if needed (run *last*) ---
    white_page_step_ok = _create_white_page_if_needed(folderpath)
    if not white_page_step_ok:
        logging.error("Warning: Creation of white page failed.")
        # Continue, but maybe return False overall? Let's return True but log warning.

    logging.info(f"--- Numerotation process finished for: {folderpath} ---")
    # Return True only if all critical steps succeeded
    final_success = rename_step_ok and correction_step_ok and dp_step_ok # white_page is optional
    if not final_success:
         logging.info("Numerotation finished with errors.")
    return final_success


#* --- Example Usage ---
#if __name__ == "__main__":
#    logging.info("Manga Numerotation Script v2")
#
#    # --- Option 1: Process a specific folder ---
#    # target_folder = 'path/to/your/manga/chapter' # <--- CHANGE THIS
#    # Example with potentially problematic name (ensure folder exists):
#    target_folder = 'Jalouses T04 copy'
#    # target_folder = 'Call of the Night T06 (Kotoyama) (2023) [Digital-1920] [Manga FR] (PapriKa+)'
#
#    if target_folder and os.path.exists(target_folder):
#         logging.info(f"Processing single folder: {target_folder}")
#         numerotation(target_folder)
#    elif target_folder:
#         logging.info(f"Error: Specified target folder '{target_folder}' not found.")
#         logging.info("Please modify the 'target_folder' variable in the script's __main__ block.")
#    else:
#         logging.info("No specific target_folder set.")
