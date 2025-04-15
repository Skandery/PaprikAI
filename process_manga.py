import os
import sys
import argparse
import logging
import pathlib
import shutil
import zipfile
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import threading

from typing import List, Optional, Dict, Any, Tuple

# Attempt to import rarfile, fail gracefully if not installed/unrar not found
try:
    import rarfile
    # Configure rarfile if necessary (point to unrar executable if not in PATH)
    # rarfile.UNRAR_TOOL = "/path/to/unrar"
    RARFILE_AVAILABLE = True
except ImportError:
    RARFILE_AVAILABLE = False
except rarfile.RarCannotExec as e:
    logging.warning(f"rarfile loaded but 'unrar' executable not found or failed: {e}. .cbr processing disabled.")
    RARFILE_AVAILABLE = False


from tqdm import tqdm

# --- Assume these functions are importable ---
# Make sure these .py files are in the same directory or Python path
try:
    from get_title import find_name_folder # Assuming DB setup is handled within
    # Need Pillow, fuzzywuzzy, python-Levenshtein, etc. for find_name_folder
    from renomage_numerot import numerotation # Assuming EasyOCR, etc. installed
    from renomage_fichier_sans_numerot import bypass_numerot # Assuming PyTorch, timm, etc. installed
    from upscale import process_upscale # Assuming PyTorch, TRT, OpenCV, etc. installed
    from dp import process_dp # Assuming PyTorch, timm, natsort etc. installed
except ImportError as e:
    print(f"Error importing required processing modules: {e}", file=sys.stderr)
    print("Please ensure get_title.py, renomage_numerot.py, etc., are in the Python path and their dependencies are installed.", file=sys.stderr)
    sys.exit(1)
# --- End Imports ---

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
# Silence excessively verbose loggers from dependencies if needed
# logging.getLogger('easyocr').setLevel(logging.WARNING)
# logging.getLogger('PIL').setLevel(logging.WARNING)

# --- Global Lock for GPU-intensive task ---
# Ensures only one upscale process runs at a time across all threads
upscale_lock = Lock()

# == Helper Function: Archive Extraction ==
def extract_archive(archive_path: pathlib.Path, extract_to: pathlib.Path) -> bool:
    """Extracts ZIP, CBZ (ZIP), or CBR (RAR) archives."""
    ext = archive_path.suffix.lower()
    extract_to.mkdir(parents=True, exist_ok=True)
    logging.info(f"Extracting {archive_path.name} to {extract_to}...")
    try:
        if ext in ['.zip', '.cbz']:
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                # Check for directory traversal vulnerability
                for member in zip_ref.namelist():
                     member_path = pathlib.Path(member)
                     # Resolve the absolute path considering potential '..'
                     absolute_member_path = (extract_to / member_path).resolve()
                     # Ensure the resolved path is still within the intended extraction directory
                     if not absolute_member_path.is_relative_to(extract_to.resolve()):
                         logging.error(f"Skipping dangerous path in archive: {member}")
                         continue # Or raise an error? For now skip
                     zip_ref.extract(member, extract_to)

            logging.info(f"Successfully extracted {archive_path.name}")
            return True
        elif ext == '.cbr':
            if not RARFILE_AVAILABLE:
                logging.error(f"Cannot extract {archive_path.name}: rarfile library or unrar tool is not available.")
                return False
            with rarfile.RarFile(archive_path, 'r') as rar_ref:
                 # Check for directory traversal (less standardized in rarfile)
                 # Basic check: Ensure no absolute paths or excessive '..'
                 for member in rar_ref.namelist():
                    if os.path.isabs(member) or '..' in member:
                        logging.error(f"Skipping potentially unsafe path in RAR archive: {member}")
                        continue
                    # We rely more on the system's behavior for RAR extraction path safety here
                    # But check final extracted path safety if possible / library supports it

                 rar_ref.extractall(path=extract_to) # Note: path traversal might still be possible depending on unrar version and rarfile behavior
                 logging.info(f"Successfully extracted {archive_path.name}")
                 return True

        else:
            logging.warning(f"Unsupported archive format for {archive_path.name}: {ext}")
            return False
    except zipfile.BadZipFile:
        logging.error(f"Error: Bad ZIP file for {archive_path.name}")
        return False
    except rarfile.Error as e:
        logging.error(f"Error extracting RAR file {archive_path.name}: {e}")
        return False
    except FileNotFoundError:
        logging.error(f"Error: Archive file not found at {archive_path}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error extracting {archive_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return False

# == Core Processing Function ==
def process_single_item(
    input_item_path: pathlib.Path,
    output_base_dir: pathlib.Path,
    args: argparse.Namespace,
    is_archive: bool
) -> Tuple[pathlib.Path, bool]:
    """
    Processes a single item (folder or archive) through the pipeline.
    Returns the final output path and a boolean indicating success.
    """
    thread_name = threading.current_thread().name
    logging.info(f"[{thread_name}] Starting processing for: {input_item_path.name}")
    start_time = time.time()

    temp_dir: Optional[pathlib.Path] = None
    current_processing_path: pathlib.Path = input_item_path
    final_output_path: Optional[pathlib.Path] = None
    original_name = input_item_path.stem # Name without extension

    try:
        # --- 1. Preparation: Extract Archive if necessary ---
        if is_archive:
            # Create a temporary directory for extraction
            temp_dir = pathlib.Path(tempfile.mkdtemp(prefix=f"manga_proc_{original_name}_"))
            logging.info(f"[{thread_name}] Created temp dir: {temp_dir}")
            if not extract_archive(input_item_path, temp_dir):
                logging.error(f"[{thread_name}] Failed to extract archive: {input_item_path.name}")
                raise RuntimeError("Archive extraction failed")
            current_processing_path = temp_dir # Process the contents of the temp dir

            # Use the *archive* name (without ext) as the initial basis for the output name
            base_name_for_title = original_name
        else:
            # If input is a folder, we process it directly but will move it later
             base_name_for_title = input_item_path.name


        # --- 2. Title Extraction ---
        new_title: Optional[str] = base_name_for_title # Default to original name if skipped or failed
        effective_processing_path = current_processing_path # The folder containing images

        if not args.no_title:
            logging.info(f"[{thread_name}] Attempting title extraction for: {effective_processing_path}")
            try:
                if not args.no_upscale:
                    found_title = find_name_folder(str(effective_processing_path), online_search=args.online_search,upscale=True)
                else:
                    found_title = find_name_folder(str(effective_processing_path), online_search=args.online_search,upscale=False)
                if found_title:
                    # Basic sanitization for folder names
                    sanitized_title = "".join(c for c in found_title if c.isalnum() or c in (' ', '-', '_', '(', ')', '[', ']')).strip()
                    if sanitized_title:
                        new_title = sanitized_title
                        logging.info(f"[{thread_name}] Found and sanitized title: {new_title}")
                    else:
                        logging.warning(f"[{thread_name}] Title found ('{found_title}') but resulted in empty string after sanitization. Using original.")
                        new_title = base_name_for_title
                else:
                    logging.warning(f"[{thread_name}] Title extraction failed for {effective_processing_path.name}. Using original folder name.")
                    new_title = base_name_for_title
            except Exception as e:
                 logging.error(f"[{thread_name}] Error during title extraction: {e}. Using original name.")
                 new_title = base_name_for_title # Fallback to original on error
        else:
            logging.info(f"[{thread_name}] Skipping title extraction.")
            # new_title remains base_name_for_title

        # Define initial destination path (will be used if no upscale/dp)
        # It will be under the *output_base_dir*
        final_output_path = output_base_dir / new_title

        # --- 3. Renaming ---
        # Operates *in place* on current_processing_path
        if not args.no_rename:
            logging.info(f"[{thread_name}] Attempting renaming (numerotation) for: {effective_processing_path}")
            rename_success = False
            try:
                # Ensure the target exists and is a directory before renaming
                if effective_processing_path.is_dir():
                    rename_success = numerotation(str(effective_processing_path))
                    if rename_success:
                        logging.info(f"[{thread_name}] 'numerotation' completed successfully.")
                    else:
                        logging.warning(f"[{thread_name}] 'numerotation' failed or returned False. Attempting fallback 'bypass_numerot'.")
                        # Fallback
                        try:
                            status = bypass_numerot(input_folder=str(effective_processing_path), dry_run=False)
                            if "success" in status:
                                logging.info(f"[{thread_name}] Fallback 'bypass_numerot' completed successfully: {status}")
                                rename_success = True
                            else:
                                logging.error(f"[{thread_name}] Fallback 'bypass_numerot' also failed: {status}")
                        except Exception as e_bypass:
                             logging.error(f"[{thread_name}] Error executing fallback 'bypass_numerot': {e_bypass}")

                else:
                     logging.error(f"[{thread_name}] Cannot perform rename: Path '{effective_processing_path}' is not a directory.")
                     rename_success = False # Explicitly set failure

            except Exception as e_num:
                logging.error(f"[{thread_name}] Error during 'numerotation': {e_num}. Attempting fallback 'bypass_numerot'.")
                 # Fallback on primary function error
                try:
                    if effective_processing_path.is_dir(): # Check again in case error was unrelated to path
                        status = bypass_numerot(input_folder=str(effective_processing_path), dry_run=False)
                        if "success" in status:
                            logging.info(f"[{thread_name}] Fallback 'bypass_numerot' completed successfully after error: {status}")
                            rename_success = True
                        else:
                            logging.error(f"[{thread_name}] Fallback 'bypass_numerot' also failed after error: {status}")
                    else:
                         logging.error(f"[{thread_name}] Cannot perform fallback rename: Path '{effective_processing_path}' is not a directory.")

                except Exception as e_bypass_err:
                    logging.error(f"[{thread_name}] Error executing fallback 'bypass_numerot' after primary error: {e_bypass_err}")


            if not rename_success:
                 logging.warning(f"[{thread_name}] Renaming step failed for {effective_processing_path.name}. Proceeding with original file names.")
                 # Continue processing even if renaming fails
        else:
            logging.info(f"[{thread_name}] Skipping renaming.")


        # Define path for upscale output
        upscale_output_dir: pathlib.Path = output_base_dir / f"{new_title}_upscaled"

        # --- 4. Upscaling ---
        # Uses upscale_lock to ensure serial execution
        # Takes files from effective_processing_path, writes to upscale_output_dir
        did_upscale = False
        if not args.no_upscale:
            logging.info(f"[{thread_name}] Preparing for upscale. Waiting for GPU lock if necessary...")
            with upscale_lock:
                logging.info(f"[{thread_name}] Acquired GPU lock. Starting upscale for: {effective_processing_path.name}")
                # Upscale outputs to its own directory based on title
                # The base folder logic is now simpler: output_base_dir / specific_folder
                # Ensure the target directory exists for upscale output
                upscale_output_dir.mkdir(parents=True, exist_ok=True)
                logging.info(f"[{thread_name}] Upscaling '{effective_processing_path.name}' -> '{upscale_output_dir}'")

                try:
                    if effective_processing_path.is_dir():
                        # Check if source directory actually has images
                        image_extensions = ('.jpg', '.jpeg', '.png', '.webp')
                        has_images = any(f.suffix.lower() in image_extensions for f in effective_processing_path.iterdir() if f.is_file())

                        if has_images:
                            process_upscale(
                                input_folder_path=str(effective_processing_path),
                                output_folder_path=str(upscale_output_dir),
                                force_image_height=args.upscale_height
                            )
                            logging.info(f"[{thread_name}] Upscale process finished for {effective_processing_path.name}.")
                            current_processing_path = upscale_output_dir # Subsequent steps work on the upscaled output
                            final_output_path = upscale_output_dir # Update final path if upscale is the last step
                            did_upscale = True
                        else:
                            logging.warning(f"[{thread_name}] Skipping upscale for {effective_processing_path.name}: No image files found inside.")

                    else:
                        logging.error(f"[{thread_name}] Cannot perform upscale: Path '{effective_processing_path}' is not a directory.")


                except Exception as e:
                     logging.error(f"[{thread_name}] Error during 'process_upscale' for {effective_processing_path.name}: {e}")
                     logging.warning(f"[{thread_name}] Proceeding without upscaling for this item.")
                finally:
                     logging.info(f"[{thread_name}] Releasing GPU lock.")
                     if 'torch' in sys.modules and sys.modules['torch'].cuda.is_available():
                          sys.modules['torch'].cuda.empty_cache() # Try to free VRAM after use

        else:
            logging.info(f"[{thread_name}] Skipping upscale.")

        # --- 5. Process DP ---
        # Operates *in place* on the current_processing_path (which might be the upscaled dir)
        # Takes files from current_processing_path, potentially moves originals within it
        if not args.no_dp:
            logging.info(f"[{thread_name}] Attempting DP processing for: {current_processing_path}")
            try:
                if current_processing_path.is_dir():
                     # process_dp modifies the folder in-place and moves originals
                     # By default moves to subfolder 'DP_Originals'
                     dp_results = process_dp(
                         folderpath=str(current_processing_path),
                         # dp_output_folder=None, # Use default subfolder behavior
                         # skip_first=True, # Use function default
                         # jpeg_quality=92 # Use function default
                     )
                     logging.info(f"[{thread_name}] 'process_dp' completed for {current_processing_path.name}. Results: {dp_results}")
                     final_output_path = current_processing_path # DP was the last step, this is the final folder path
                else:
                     logging.error(f"[{thread_name}] Cannot perform DP processing: Path '{current_processing_path}' is not a directory.")

            except Exception as e:
                logging.error(f"[{thread_name}] Error during 'process_dp' for {current_processing_path.name}: {e}")
                logging.warning(f"[{thread_name}] Proceeding without DP processing for this item.")
        else:
            logging.info(f"[{thread_name}] Skipping DP processing.")

# --- 6. Create CBZ (Optional) ---
        cbz_path: Optional[pathlib.Path] = None
        if args.zip_output:
            logging.info(f"[{thread_name}] Preparing to create .cbz for: {current_processing_path}")
            try:
                if current_processing_path.is_dir():
                    # Define CBZ path: same as final_output_path but with .cbz extension
                    cbz_path = output_base_dir / f"{new_title}.cbz"
                    logging.info(f"[{thread_name}] Creating CBZ at: {cbz_path}")

                    # Ensure only image files are included
                    image_extensions = ('.jpg', '.jpeg', '.png', '.webp')
                    with zipfile.ZipFile(cbz_path, 'w', zipfile.ZIP_STORED) as new_zip:
                        for root, _, files in os.walk(current_processing_path):
                            for file in files:
                                if file.lower().endswith(image_extensions):
                                    file_path = os.path.join(root, file)
                                    # Compute archive name relative to current_processing_path
                                    arcname = os.path.relpath(file_path, current_processing_path)
                                    # Set timestamp to 01/01/2000 00:00:00
                                    zip_info = zipfile.ZipInfo(arcname, date_time=(2000, 1, 1, 0, 0, 0))
                                    with open(file_path, 'rb') as f:
                                        new_zip.writestr(zip_info, f.read())
                    logging.info(f"[{thread_name}] Successfully created CBZ: {cbz_path}")

                    # Delete folder if requested
                    if args.delete_folder:
                        logging.info(f"[{thread_name}] Deleting output folder as requested: {current_processing_path}")
                        try:
                            shutil.rmtree(current_processing_path)
                            logging.info(f"[{thread_name}] Output folder deleted: {current_processing_path}")
                            # Update final_output_path to point to CBZ
                            final_output_path = cbz_path
                        except Exception as e:
                            logging.error(f"[{thread_name}] Failed to delete output folder {current_processing_path}: {e}")
                else:
                    logging.error(f"[{thread_name}] Cannot create CBZ: Path '{current_processing_path}' is not a directory.")
            except Exception as e:
                logging.error(f"[{thread_name}] Error creating CBZ for {current_processing_path.name}: {e}")
                cbz_path = None
                logging.warning(f"[{thread_name}] Proceeding without CBZ creation.")
        else:
            logging.info(f"[{thread_name}] Skipping CBZ creation.")
        # --- 7. Final Output Management ---
        # Ensure final_output_path is set correctly
        if final_output_path is None:
             # This should only happen if input was a folder and no processing modifying the path occurred
             # We need to decide where it goes - default to output_base_dir / original_name or new_title
             final_output_path = output_base_dir / new_title # Use the (potentially new) title
             logging.debug(f"[{thread_name}] Final output path determined as: {final_output_path}")


        # If the source was a folder (not archive) and we processed it in place (or upscale/dp changed current_processing_path)
        # AND the final intended path is different from where it currently is, move it.
        if not is_archive and current_processing_path != final_output_path:
            logging.info(f"[{thread_name}] Moving final processed folder from '{current_processing_path}' to '{final_output_path}'")
            # Ensure parent of final destination exists
            final_output_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.move(str(current_processing_path), str(final_output_path))
                logging.info(f"[{thread_name}] Successfully moved to final destination.")
            except Exception as e:
                logging.error(f"[{thread_name}] Failed to move '{current_processing_path}' to '{final_output_path}': {e}")
                # If move fails, the result is left where it is (e.g., _upscaled folder)
                final_output_path = current_processing_path # Update the reported path to the actual location
                raise RuntimeError("Final move operation failed")

        # If the source was an archive (processed in temp_dir) OR it was a folder processed in place and didn't need moving
        # AND current_processing_path is different from final_output_path (e.g., only title changed)
        elif (is_archive or (not is_archive and current_processing_path == input_item_path)) and current_processing_path != final_output_path :
            # Ensure parent of final destination exists
             final_output_path.parent.mkdir(parents=True, exist_ok=True)
             # Check if destination exists - might need merging or replacement logic? For now, error out if exists.
             if final_output_path.exists():
                 logging.warning(f"[{thread_name}] Final destination '{final_output_path}' already exists. Overwriting is not implemented. Skipping final move.")
                 # Report the path where it was processed, or fail? Let's report where it is.
                 final_output_path = current_processing_path # It stays in the temp dir or original location
             else:
                 logging.info(f"[{thread_name}] Moving final processed content from '{current_processing_path}' to '{final_output_path}'")
                 try:
                     shutil.move(str(current_processing_path), str(final_output_path))
                     logging.info(f"[{thread_name}] Successfully moved to final destination.")
                 except Exception as e:
                     logging.error(f"[{thread_name}] Failed to move '{current_processing_path}' to '{final_output_path}': {e}")
                     # Result might be left in temp dir or original location
                     final_output_path = current_processing_path
                     raise RuntimeError("Final move operation failed")

        elif current_processing_path == final_output_path:
             logging.info(f"[{thread_name}] Final content is already at the destination: '{final_output_path}'")
        else:
             # Should not happen with current logic?
             logging.warning(f"[{thread_name}] Unexpected state for final move/rename. current={current_processing_path}, final={final_output_path}")


        processing_time = time.time() - start_time
        logging.info(f"[{thread_name}] Successfully finished processing {input_item_path.name} in {processing_time:.2f} seconds. Final output at: {final_output_path}")
        return final_output_path, True

    except Exception as e:
        processing_time = time.time() - start_time
        logging.error(f"[{thread_name}] FAILED processing {input_item_path.name} after {processing_time:.2f} seconds. Error: {e}")
        # Optionally log traceback
        # import traceback
        # logging.error(traceback.format_exc())
        return input_item_path, False # Return original path on failure

    finally:
        # --- 7. Cleanup ---
        if temp_dir and temp_dir.exists():
            logging.info(f"[{thread_name}] Cleaning up temporary directory: {temp_dir}")
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logging.error(f"[{thread_name}] Failed to remove temporary directory {temp_dir}: {e}")


# == Input Path Identification and Preparation ==
def identify_and_prepare_inputs(input_path_str: str) -> List[Tuple[pathlib.Path, bool]]:
    """
    Identifies input type and returns a list of items to process.
    Each item is a tuple: (pathlib.Path, is_archive_flag).
    """
    input_path = pathlib.Path(input_path_str)
    items_to_process: List[Tuple[pathlib.Path, bool]] = []
    archive_extensions = {'.zip', '.cbz', '.cbr'}

    if not input_path.exists():
        logging.error(f"Input path does not exist: {input_path}")
        return items_to_process # Empty list

    if input_path.is_file():
        if input_path.suffix.lower() in archive_extensions:
            items_to_process.append((input_path, True))
            logging.info(f"Identified single archive file: {input_path.name}")
        else:
            logging.warning(f"Input is a file but not a supported archive ({archive_extensions}): {input_path.name}. Skipping.")

    elif input_path.is_dir():
        # Check if it's a folder *containing* images (process as single item)
        # Or a folder *containing* archives/folders (process items inside)
        contains_images = False
        contains_archives_or_folders = False
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
        nested_items: List[pathlib.Path] = []

        for item in input_path.iterdir():
            if item.is_file() and item.suffix.lower() in image_extensions:
                contains_images = True
            if item.is_file() and item.suffix.lower() in archive_extensions:
                contains_archives_or_folders = True
                nested_items.append(item)
            elif item.is_dir():
                contains_archives_or_folders = True
                nested_items.append(item)
            # Stop checking early if both found or just images found in a non-nested scenario needed
            # if contains_images and contains_archives_or_folders: break

        if contains_images and not contains_archives_or_folders:
             # Treat as a single folder of images
             items_to_process.append((input_path, False))
             logging.info(f"Identified single folder of images: {input_path.name}")
        elif contains_archives_or_folders:
            # Treat as a container folder - process items inside
            logging.info(f"Identified container folder: {input_path.name}. Processing contents.")
            for item in nested_items:
                 item_is_archive = item.is_file() and item.suffix.lower() in archive_extensions
                 item_is_processable_folder = item.is_dir() # Assuming any sub-dir might be manga

                 if item_is_archive:
                      items_to_process.append((item, True))
                 elif item_is_processable_folder:
                     # Check if this sub-folder itself contains images
                     has_images_inside = any(f.suffix.lower() in image_extensions for f in item.iterdir() if f.is_file())
                     if has_images_inside:
                        items_to_process.append((item, False))
                     else:
                        logging.warning(f"Skipping subfolder '{item.name}' as it contains no supported image files.")
                 # else: ignore other file types in the container folder
        elif not contains_images and not contains_archives_or_folders:
            logging.warning(f"Input folder seems empty or contains unsupported files: {input_path.name}. Nothing to process.")
        # Handle ambiguity: if a folder contains *both* loose images *and* archives/subfolders
        # Current logic prioritizes processing the archives/subfolders within. Add specific handling if needed.
        elif contains_images and contains_archives_or_folders:
            logging.warning(f"Input folder '{input_path.name}' contains both loose images and archives/subfolders.")
            logging.warning("Processing only the archives/subfolders found within.")
            # Logic already handled above by populating nested_items
            for item in nested_items:
                 item_is_archive = item.is_file() and item.suffix.lower() in archive_extensions
                 item_is_processable_folder = item.is_dir() # Assuming any sub-dir might be manga
                 if item_is_archive: items_to_process.append((item, True))
                 elif item_is_processable_folder:
                       has_images_inside = any(f.suffix.lower() in image_extensions for f in item.iterdir() if f.is_file())
                       if has_images_inside: items_to_process.append((item, False))
                       else: logging.warning(f"Skipping subfolder '{item.name}' as it contains no supported image files.")


    else:
        logging.error(f"Input path is neither a file nor a directory: {input_path}")

    return items_to_process


# == Main Execution ==
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process manga folders or archives with title extraction, renaming, upscaling, and DP handling.")
    parser.add_argument("input_path", help="Path to the input folder, archive (.zip, .cbz, .cbr), or folder containing archives/subfolders.")
    parser.add_argument("-o", "--output_base", default=None, help="Base directory for output. If None, defaults to './output'. Final structure will be output_base/new_title.")
    parser.add_argument("--no-title", action="store_true", help="Skip title extraction step.")
    parser.add_argument("--no-rename", action="store_true", help="Skip renaming steps (numerotation/bypass).")
    parser.add_argument("--no-upscale", action="store_true", help="Skip upscale step.")
    parser.add_argument("--upscale-height", type=int, default=None, help="Force manga input height for upscaling model selection and processing.")
    parser.add_argument("--no-dp", action="store_true", help="Skip DP (double page) processing step.")
    parser.add_argument("--online-search", action="store_true", default=True, help="Enable online search during title extraction if DB lookup fails.")
    parser.add_argument("--zip-output", action="store_true", help="Zip the final output folder into a .cbz file.")
    parser.add_argument("--delete-folder", action="store_true", help="Delete the output folder after creating the .cbz file (requires --zip-output).")
    parser.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 1, help="Number of parallel worker threads for processing multiple items.")

    args = parser.parse_args()

    # --- Determine Output Base Directory ---
    if args.output_base:
        output_base_dir = pathlib.Path(args.output_base)
    else:
        # Default to ./output relative to script or cwd
        # Consider making it absolute?
        script_dir = pathlib.Path(__file__).parent if "__file__" in locals() else pathlib.Path.cwd()
        output_base_dir = (script_dir / "output").resolve() # Default to an 'output' subdir, resolved to absolute

    output_base_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output base directory set to: {output_base_dir}")


    # --- Identify Items to Process ---
    items_to_process = identify_and_prepare_inputs(args.input_path)
    num_items = len(items_to_process)

    if num_items == 0:
        logging.info("No processable items found. Exiting.")
        sys.exit(0)

    logging.info(f"Found {num_items} item(s) to process.")
    # Limit jobs if fewer items than requested jobs
    num_workers = min(args.jobs, num_items)
    logging.info(f"Using {num_workers} worker thread(s).")


    # --- Process Items ---
    results = {"success": [], "failed": []}
    overall_start_time = time.time()

    if num_items == 1:
        # Process single item directly without ThreadPoolExecutor
        logging.info("Processing single item...")
        item_path, is_archive = items_to_process[0]
        final_path, success = process_single_item(item_path, output_base_dir, args, is_archive)
        if success:
            results["success"].append(final_path)
        else:
            results["failed"].append(item_path) # Report original path on failure
    else:
        # Process multiple items in parallel
        logging.info("Processing multiple items in parallel...")
        # Use ThreadPoolExecutor for parallel I/O and CPU tasks
        # The upscale_lock handles the GPU bottleneck internally
        with ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix='Worker') as executor:
            # Submit tasks
            futures = {executor.submit(process_single_item, item_path, output_base_dir, args, is_archive): item_path
                       for item_path, is_archive in items_to_process}

            # Process results as they complete with progress bar
            try:
                 for future in tqdm(as_completed(futures), total=num_items, desc="Overall Progress"):
                     original_path = futures[future]
                     try:
                         final_path, success = future.result()
                         if success:
                             results["success"].append(final_path)
                         else:
                             results["failed"].append(original_path) # Report original path
                     except Exception as exc:
                         logging.error(f"Item {original_path.name} generated an exception during processing: {exc}")
                         results["failed"].append(original_path)
                         # Optionally log traceback
                         # import traceback
                         # logging.error(traceback.format_exc())
            except KeyboardInterrupt:
                 logging.warning("\nCtrl+C detected. Shutting down workers...")
                 # Attempt graceful shutdown. Note: running tasks might still finish.
                 executor.shutdown(wait=False, cancel_futures=True) # Python 3.9+ for cancel_futures
                 # or executor.shutdown(wait=True) and let them finish / handle cancellation internally if possible
                 print("Processing interrupted.", file=sys.stderr)
                 # Exit or just report partial results? Reporting results so far.


    # --- Final Summary ---
    overall_end_time = time.time()
    total_duration = overall_end_time - overall_start_time
    logging.info("\n--- Processing Summary ---")
    logging.info(f"Total time: {total_duration:.2f} seconds")
    logging.info(f"Successfully processed: {len(results['success'])} items")
    for path in results["success"]:
        logging.info(f"  - {path}{' (CBZ)' if path.suffix.lower() == '.cbz' else ''}")
    logging.info(f"Failed to process: {len(results['failed'])} items")
    for path in results["failed"]:
        logging.info(f"  - {path.name}")

    if results["failed"]:
         logging.warning("Some items failed processing. Check logs above for details.")
         sys.exit(1) # Exit with error code if failures occurred
    else:
         logging.info("All items processed successfully.")
         sys.exit(0)