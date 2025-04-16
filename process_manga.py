# --- START OF FILE process_manga.py ---

import os
import sys
import argparse
import logging
import pathlib
import shutil
import zipfile
import tempfile
import time
import re
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
    logging.info("rarfile library not found. .cbr processing disabled.")
except rarfile.RarCannotExec as e:
    logging.info(f"rarfile loaded but 'unrar' executable not found or failed: {e}. .cbr processing disabled.")
    RARFILE_AVAILABLE = False
except Exception as e: # Catch other potential rarfile init errors
    logging.warning(f"An unexpected error occurred during rarfile initialization: {e}. .cbr processing disabled.")
    RARFILE_AVAILABLE = False


from tqdm import tqdm
# --- Assume these functions are importable ---
# Make sure these .py files are in the same directory or Python path
try:
    from helpers.get_title import find_name_folder # Assuming DB setup is handled within
    from helpers.renomage_numerot import numerotation # Assuming EasyOCR, etc. installed
    from helpers.renomage_fichier_sans_numerot import bypass_numerot # Assuming PyTorch, timm, etc. installed
    from helpers.upscale import process_upscale # Assuming PyTorch, TRT, OpenCV, etc. installed
    from helpers.dp import process_dp # Assuming PyTorch, timm, natsort etc. installed
except ImportError as e:
    print(f"Error importing required processing modules: {e}", file=sys.stderr)
    print("Please ensure get_title.py, renomage_numerot.py, etc., are in the Python path and their dependencies are installed.", file=sys.stderr)
    sys.exit(1)
# --- End Imports ---

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO, # Changed to INFO for better debugging during development
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
                     # Skip macOS resource fork files and other hidden files/folders if desired
                     if member.startswith('__MACOSX/') or os.path.basename(member).startswith('.'):
                         logging.debug(f"Skipping metadata entry: {member}")
                         continue

                     member_path = pathlib.Path(member)
                     # Resolve the absolute path considering potential '..'
                     # Use os.path.normpath to handle mixed separators and redundant parts
                     normalized_member = os.path.normpath(member)
                     # Prevent absolute paths within the zip
                     if os.path.isabs(normalized_member):
                          logging.error(f"Skipping absolute path in archive: {member}")
                          continue
                     # Construct the full target path
                     target_path = (extract_to / normalized_member).resolve()

                     # Ensure the resolved path is still within the intended extraction directory
                     if not target_path.is_relative_to(extract_to.resolve()):
                         logging.error(f"Skipping dangerous path (directory traversal attempt) in archive: {member}")
                         continue

                     # Extract safely
                     # Check if it's a directory entry (ends with /)
                     if member.endswith('/'):
                         # Create directory if it doesn't exist
                         target_path.mkdir(parents=True, exist_ok=True)
                     else:
                         # Create parent directory if it doesn't exist before extracting file
                         target_path.parent.mkdir(parents=True, exist_ok=True)
                         # Extract the file
                         with open(target_path, 'wb') as outfile:
                             outfile.write(zip_ref.read(member))

            logging.info(f"Successfully extracted {archive_path.name}")
            return True
        elif ext == '.cbr':
            if not RARFILE_AVAILABLE:
                logging.error(f"Cannot extract {archive_path.name}: rarfile library or unrar tool is not available.")
                return False
            with rarfile.RarFile(archive_path, 'r') as rar_ref:
                 # Check for directory traversal (less standardized in rarfile)
                 # Basic check: Ensure no absolute paths or excessive '..'
                 for member_info in rar_ref.infolist():
                    member = member_info.filename
                    # Skip macOS resource fork files and other hidden files/folders if desired
                    if member.startswith('__MACOSX/') or os.path.basename(member).startswith('.'):
                        logging.debug(f"Skipping metadata entry: {member}")
                        continue

                    if os.path.isabs(member) or '..' in member.split(os.path.sep): # Check path components
                        logging.error(f"Skipping potentially unsafe path in RAR archive: {member}")
                        continue

                    # Construct the full target path for checking (rarfile extracts directly)
                    normalized_member = os.path.normpath(member)
                    target_path = (extract_to / normalized_member).resolve()

                    # Ensure the resolved path is still within the intended extraction directory
                    if not target_path.is_relative_to(extract_to.resolve()):
                        logging.error(f"Skipping dangerous path (directory traversal attempt) in RAR archive: {member}")
                        continue

                    # If safe, let rarfile extract (it handles directory creation)
                    # Note: We rely on the check above; rar_ref.extractall might still have risks depending on unrar version
                 rar_ref.extractall(path=str(extract_to)) # Use string path for rarfile
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

    temp_dir_obj: Optional[tempfile.TemporaryDirectory] = None # Use context manager if possible
    temp_dir_path: Optional[pathlib.Path] = None
    current_processing_path: pathlib.Path = input_item_path # Initial path
    final_output_path: Optional[pathlib.Path] = None
    original_name = input_item_path.stem # Name without extension
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp'} # Define once

    try:
        # --- 1. Preparation: Extract Archive if necessary ---
        if is_archive:
            # Create a temporary directory using context manager for auto-cleanup
            temp_dir_obj = tempfile.TemporaryDirectory(prefix=f"manga_proc_{original_name}_")
            temp_dir_path = pathlib.Path(temp_dir_obj.name)
            logging.info(f"[{thread_name}] Created temp dir: {temp_dir_path}")

            if not extract_archive(input_item_path, temp_dir_path):
                logging.error(f"[{thread_name}] Failed to extract archive: {input_item_path.name}")
                raise RuntimeError("Archive extraction failed")

            # --- * NEW: Detect nested folder structure * ---
            # Check if images are directly in temp_dir_path or in a single subdirectory
            root_items = list(temp_dir_path.iterdir())
            # Filter out potential hidden files/folders like .DS_Store or __MACOSX
            root_items_filtered = [item for item in root_items if not item.name.startswith('.') and item.name != '__MACOSX']

            root_contains_images = any(f.is_file() and f.suffix.lower() in image_extensions for f in root_items_filtered)

            if root_contains_images:
                logging.info(f"[{thread_name}] Images found directly in the root of the extracted archive.")
                current_processing_path = temp_dir_path
            else:
                # Look for a single subdirectory containing images
                subdirs = [d for d in root_items_filtered if d.is_dir()]
                potential_image_dir = None
                if len(subdirs) == 1:
                    subdir_path = subdirs[0]
                    subdir_contains_images = any(f.is_file() and f.suffix.lower() in image_extensions for f in subdir_path.iterdir())
                    if subdir_contains_images:
                        potential_image_dir = subdir_path
                        logging.info(f"[{thread_name}] Found single nested folder with images: '{subdir_path.name}'. Using it as processing path.")
                    else:
                         logging.warning(f"[{thread_name}] Found single nested folder '{subdir_path.name}', but it contains no supported images.")
                elif len(subdirs) > 1:
                     logging.warning(f"[{thread_name}] Found multiple ({len(subdirs)}) subdirectories in the archive root. Cannot automatically determine the correct image folder. Will attempt processing from the root, which might fail.")
                else: # No subdirs, and no images in root found earlier
                     logging.warning(f"[{thread_name}] No images found in the archive root and no subdirectories found. Processing will likely fail.")

                if potential_image_dir:
                    current_processing_path = potential_image_dir
                else:
                    # Fallback: Use the temp dir root, even if likely to fail.
                    current_processing_path = temp_dir_path
                    logging.warning(f"[{thread_name}] Proceeding with temp directory root '{temp_dir_path}' as processing path despite potential issues.")

            # Use the *archive* name (without ext) as the initial basis for the output name
            base_name_for_title = original_name
            # --- * END: Detect nested folder structure * ---

        else: # Input is a folder
            current_processing_path = input_item_path # Process the input folder directly
            base_name_for_title = input_item_path.name


        # --- 2. Title Extraction ---
        new_title: Optional[str] = base_name_for_title # Default to original name if skipped or failed
        effective_processing_path = current_processing_path # The folder containing images (potentially nested)

        if not args.no_title:
            logging.info(f"[{thread_name}] Attempting title extraction for: {effective_processing_path}")
            try:
                # Check if the path actually exists before passing to find_name_folder
                if not effective_processing_path.exists():
                     raise FileNotFoundError(f"Effective processing path does not exist: {effective_processing_path}")
                if not effective_processing_path.is_dir():
                     raise NotADirectoryError(f"Effective processing path is not a directory: {effective_processing_path}")

                # Check if directory contains images before calling title extraction
                has_images_for_title = any(f.is_file() and f.suffix.lower() in image_extensions for f in effective_processing_path.iterdir())
                if not has_images_for_title:
                    logging.warning(f"[{thread_name}] No images found in '{effective_processing_path.name}' for title extraction. Skipping title step.")
                    new_title = base_name_for_title # Keep original name
                else:
                    # Proceed with title extraction
                    found_title = find_name_folder(str(effective_processing_path), online_search=args.online_search, upscale=not args.no_upscale)
                    if found_title:
                        # Basic sanitization for folder names
                        sanitized_title = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '', found_title).strip()
                        if sanitized_title:
                            new_title = sanitized_title
                            logging.info(f"[{thread_name}] Found and sanitized title: {new_title}")
                        else:
                            logging.warning(f"[{thread_name}] Title found ('{found_title}') but resulted in empty string after sanitization. Using original.")
                            new_title = base_name_for_title
                    else:
                        logging.warning(f"[{thread_name}] Title extraction failed for {effective_processing_path.name}. Using original folder name.")
                        new_title = base_name_for_title

            except (FileNotFoundError, NotADirectoryError) as path_err:
                 logging.error(f"[{thread_name}] Error accessing path for title extraction: {path_err}. Using original name.")
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
        # Operates *in place* on effective_processing_path
        if not args.no_rename:
            logging.info(f"[{thread_name}] Attempting renaming (numerotation) for: {effective_processing_path}")
            rename_success = False
            try:
                # Ensure the target exists and is a directory before renaming
                if effective_processing_path.is_dir():
                    # Check for images before attempting rename
                    has_images_for_rename = any(f.is_file() and f.suffix.lower() in image_extensions for f in effective_processing_path.iterdir())
                    if not has_images_for_rename:
                         logging.warning(f"[{thread_name}] No images found in '{effective_processing_path.name}' to rename. Skipping rename step.")
                         rename_success = True # Consider skipping as success? Or False? Let's say True as there's nothing to fail on.
                    else:
                        rename_success = numerotation(str(effective_processing_path))
                        if rename_success:
                            logging.info(f"[{thread_name}] 'numerotation' completed successfully.")
                        else:
                            logging.warning(f"[{thread_name}] 'numerotation' failed or returned False. Attempting fallback 'bypass_numerot'.")
                            # Fallback
                            try:
                                status = bypass_numerot(input_folder=str(effective_processing_path), dry_run=False)
                                if isinstance(status, dict) and "success" in status.get("status", ""): # Check dict and status key
                                    logging.info(f"[{thread_name}] Fallback 'bypass_numerot' completed successfully: {status}")
                                    rename_success = True
                                else:
                                    logging.error(f"[{thread_name}] Fallback 'bypass_numerot' also failed or returned unexpected status: {status}")
                            except Exception as e_bypass:
                                 logging.error(f"[{thread_name}] Error executing fallback 'bypass_numerot': {e_bypass}")

                else:
                     logging.error(f"[{thread_name}] Cannot perform rename: Path '{effective_processing_path}' is not a directory or does not exist.")
                     rename_success = False # Explicitly set failure

            except Exception as e_num:
                logging.error(f"[{thread_name}] Error during 'numerotation': {e_num}. Attempting fallback 'bypass_numerot'.")
                 # Fallback on primary function error
                try:
                    if effective_processing_path.is_dir(): # Check again in case error was unrelated to path
                        has_images_for_fallback = any(f.is_file() and f.suffix.lower() in image_extensions for f in effective_processing_path.iterdir())
                        if has_images_for_fallback:
                            status = bypass_numerot(input_folder=str(effective_processing_path), dry_run=False)
                            if isinstance(status, dict) and "success" in status.get("status", ""): # Check dict and status key
                                logging.info(f"[{thread_name}] Fallback 'bypass_numerot' completed successfully after error: {status}")
                                rename_success = True
                            else:
                                logging.error(f"[{thread_name}] Fallback 'bypass_numerot' also failed after error or returned unexpected status: {status}")
                        else:
                            logging.warning(f"[{thread_name}] No images found in '{effective_processing_path.name}' for fallback rename. Skipping.")
                            rename_success = True # Skipping is not a failure here
                    else:
                         logging.error(f"[{thread_name}] Cannot perform fallback rename: Path '{effective_processing_path}' is not a directory or does not exist.")

                except Exception as e_bypass_err:
                    logging.error(f"[{thread_name}] Error executing fallback 'bypass_numerot' after primary error: {e_bypass_err}")


            if not rename_success:
                 # Only log warning if we actually expected renaming to happen (i.e., images were present)
                 if effective_processing_path.is_dir() and any(f.is_file() and f.suffix.lower() in image_extensions for f in effective_processing_path.iterdir()):
                     logging.warning(f"[{thread_name}] Renaming step failed for {effective_processing_path.name}. Proceeding with original file names.")
                 # Continue processing even if renaming fails or was skipped
        else:
            logging.info(f"[{thread_name}] Skipping renaming.")


        # Define path for upscale output
        # IMPORTANT: Use the potentially modified new_title here
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
                # Ensure the target directory exists for upscale output
                upscale_output_dir.mkdir(parents=True, exist_ok=True)
                logging.info(f"[{thread_name}] Upscaling '{effective_processing_path.name}' -> '{upscale_output_dir}'")

                try:
                    if effective_processing_path.is_dir():
                        # Check if source directory actually has images
                        has_images = any(f.is_file() and f.suffix.lower() in image_extensions for f in effective_processing_path.iterdir())

                        if has_images:
                            process_upscale(
                                input_folder_path=str(effective_processing_path),
                                output_folder_path=str(upscale_output_dir),
                                force_image_height=args.upscale_height
                            )
                            logging.info(f"[{thread_name}] Upscale process finished for {effective_processing_path.name}.")
                            # IMPORTANT: Update current_processing_path to the *new* location of the processed files
                            current_processing_path = upscale_output_dir
                            final_output_path = upscale_output_dir # Update final path if upscale is the last step
                            did_upscale = True
                        else:
                            logging.warning(f"[{thread_name}] Skipping upscale for {effective_processing_path.name}: No image files found inside.")
                            # If we skip upscale, current_processing_path remains unchanged (pointing to original or temp/nested)
                            # And final_output_path remains output_base_dir / new_title

                    else:
                        logging.error(f"[{thread_name}] Cannot perform upscale: Path '{effective_processing_path}' is not a directory or does not exist.")


                except Exception as e:
                     logging.error(f"[{thread_name}] Error during 'process_upscale' for {effective_processing_path.name}: {e}")
                     logging.warning(f"[{thread_name}] Proceeding without upscaling for this item.")
                     # Ensure paths are consistent if upscale fails mid-way
                     # current_processing_path should remain where the source images were
                     # final_output_path should revert to the non-upscaled target
                     current_processing_path = effective_processing_path # Revert to source path
                     final_output_path = output_base_dir / new_title # Revert to non-upscaled final path
                finally:
                     logging.info(f"[{thread_name}] Releasing GPU lock.")
                     # Attempt to clear CUDA cache if PyTorch was used
                     if 'torch' in sys.modules and hasattr(sys.modules['torch'], 'cuda') and sys.modules['torch'].cuda.is_available():
                          try:
                              sys.modules['torch'].cuda.empty_cache()
                              logging.debug(f"[{thread_name}] Cleared CUDA cache.")
                          except Exception as cache_e:
                              logging.warning(f"[{thread_name}] Failed to clear CUDA cache: {cache_e}")


        else:
            logging.info(f"[{thread_name}] Skipping upscale.")
            # If upscale is skipped, current_processing_path remains where the images are (original or temp/nested)
            # And final_output_path remains output_base_dir / new_title

        # --- 5. Process DP ---
        # Operates *in place* on the current_processing_path (which might be the upscaled dir, or the original/temp/nested dir)
        if not args.no_dp:
            logging.info(f"[{thread_name}] Attempting DP processing for: {current_processing_path}")
            try:
                if current_processing_path.is_dir():
                     # Check for images before processing DP
                     has_images_for_dp = any(f.is_file() and f.suffix.lower() in image_extensions for f in current_processing_path.iterdir())
                     if not has_images_for_dp:
                          logging.warning(f"[{thread_name}] No images found in '{current_processing_path.name}' for DP processing. Skipping DP step.")
                     else:
                         # process_dp modifies the folder in-place and moves originals
                         dp_results = process_dp(
                             folderpath=str(current_processing_path),
                             # dp_output_folder=None, # Use default subfolder behavior
                             # skip_first=True, # Use function default
                             # jpeg_quality=92 # Use function default
                         )
                         logging.info(f"[{thread_name}] 'process_dp' completed for {current_processing_path.name}. Results: {dp_results}")
                         # The final path is still the directory where DP operated
                         final_output_path = current_processing_path
                else:
                     logging.error(f"[{thread_name}] Cannot perform DP processing: Path '{current_processing_path}' is not a directory or does not exist.")

            except Exception as e:
                logging.error(f"[{thread_name}] Error during 'process_dp' for {current_processing_path.name}: {e}")
                logging.warning(f"[{thread_name}] Proceeding without DP processing for this item.")
                # Ensure final_output_path is consistent if DP fails
                final_output_path = current_processing_path # It remains where it was before DP attempt
        else:
            logging.info(f"[{thread_name}] Skipping DP processing.")
            # If DP skipped, final_output_path remains where it was after potential upscale (or original target)


        # --- 6. Create CBZ (Optional) ---
        cbz_path: Optional[pathlib.Path] = None
        # Use the potentially updated new_title for the CBZ name
        cbz_target_path = output_base_dir / f"{new_title}.cbz"

        if args.zip_output:
            logging.info(f"[{thread_name}] Preparing to create .cbz for: {current_processing_path}")
            try:
                if current_processing_path.is_dir():
                    logging.info(f"[{thread_name}] Creating CBZ at: {cbz_target_path}")

                    # Ensure only image files are included and handle potential subdirs created by DP
                    files_to_zip = []
                    for item in current_processing_path.rglob('*'): # Use rglob to find files in subdirs
                        if item.is_file() and item.suffix.lower() in image_extensions:
                            # Exclude files in known non-image subdirs if necessary (e.g., DP_Originals)
                            if "DP_Originals" not in item.parts: # Basic check, adjust if needed
                                files_to_zip.append(item)

                    if not files_to_zip:
                         logging.warning(f"[{thread_name}] No image files found in '{current_processing_path}' to create CBZ. Skipping CBZ creation.")
                    else:
                        # Sort files naturally for correct reading order (important!)
                        try:
                            import natsort
                            files_to_zip = natsort.natsorted(files_to_zip, key=lambda x: x.name)
                        except ImportError:
                            logging.warning(f"[{thread_name}] 'natsort' library not found. Files will be zipped in OS default order.")
                            files_to_zip.sort(key=lambda x: x.name)


                        with zipfile.ZipFile(cbz_target_path, 'w', zipfile.ZIP_STORED) as new_zip:
                            for file_path in files_to_zip:
                                # Compute archive name relative to current_processing_path
                                arcname = os.path.relpath(file_path, current_processing_path)
                                # Set timestamp to 01/01/2000 00:00:00 for consistency
                                zip_info = zipfile.ZipInfo(arcname, date_time=(2000, 1, 1, 0, 0, 0))
                                zip_info.compress_type = zipfile.ZIP_STORED # Ensure no compression
                                with open(file_path, 'rb') as f:
                                    new_zip.writestr(zip_info, f.read())

                        logging.info(f"[{thread_name}] Successfully created CBZ: {cbz_target_path}")
                        cbz_path = cbz_target_path # Store the path of the created CBZ

                        # Delete folder if requested *and* CBZ creation was successful
                        if args.delete_folder:
                            logging.info(f"[{thread_name}] Deleting output folder as requested: {current_processing_path}")
                            try:
                                shutil.rmtree(current_processing_path)
                                logging.info(f"[{thread_name}] Output folder deleted: {current_processing_path}")
                                # Update final_output_path to point to CBZ since folder is gone
                                final_output_path = cbz_path
                            except Exception as e:
                                logging.error(f"[{thread_name}] Failed to delete output folder {current_processing_path}: {e}")
                                # If deletion fails, the final path is still the folder (or the CBZ if that's preferred?)
                                # Let's keep final_output_path pointing to the CBZ as it was created.
                                final_output_path = cbz_path
                else:
                    logging.error(f"[{thread_name}] Cannot create CBZ: Source path '{current_processing_path}' is not a directory or does not exist.")
            except Exception as e:
                logging.error(f"[{thread_name}] Error creating CBZ for {current_processing_path.name}: {e}")
                cbz_path = None # Ensure cbz_path is None on failure
                logging.warning(f"[{thread_name}] Proceeding without CBZ creation.")
                # If CBZ fails, final_output_path remains the folder path
                final_output_path = current_processing_path
        else:
            logging.info(f"[{thread_name}] Skipping CBZ creation.")
            # If skipping CBZ, final_output_path remains the folder path


        # --- 7. Final Output Management ---
        # At this point:
        # - `current_processing_path` points to the folder containing the final image files (could be temp, temp/nested, original input, or _upscaled folder).
        # - `final_output_path` points to the *intended* final location/name (e.g., output_base/new_title, output_base/new_title_upscaled, or the CBZ path).
        # - If CBZ was created and folder deleted, `final_output_path` should be the CBZ path.

        # We need to move/rename the `current_processing_path` folder to its final destination *unless*
        # a) CBZ was created and the folder was deleted.
        # b) The current path is already the final path.

        is_final_path_cbz = args.zip_output and cbz_path is not None and final_output_path == cbz_path
        should_move_folder = not is_final_path_cbz and current_processing_path != final_output_path

        if should_move_folder:
            logging.info(f"[{thread_name}] Moving final processed folder from '{current_processing_path}' to '{final_output_path}'")
            # Ensure parent of final destination exists
            final_output_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                # Check if destination exists - simple overwrite for now, add handling if needed
                if final_output_path.exists():
                     logging.warning(f"[{thread_name}] Destination '{final_output_path}' already exists. Overwriting.")
                     if final_output_path.is_dir():
                         shutil.rmtree(final_output_path)
                     else:
                         final_output_path.unlink()

                shutil.move(str(current_processing_path), str(final_output_path))
                logging.info(f"[{thread_name}] Successfully moved to final destination: {final_output_path}")
                # Update current_processing_path to reflect the move, though it's less critical now
                current_processing_path = final_output_path
            except Exception as e:
                logging.error(f"[{thread_name}] Failed to move '{current_processing_path}' to '{final_output_path}': {e}")
                # If move fails, the result is left where it is. Update final_output_path to report actual location.
                final_output_path = current_processing_path
                # Don't raise error here, just report failure and continue cleanup
                # raise RuntimeError("Final move operation failed") # Optional: make it a fatal error

        elif is_final_path_cbz:
             logging.info(f"[{thread_name}] Final output is CBZ file: {final_output_path}. Source folder potentially deleted.")
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
        import traceback
        logging.error(traceback.format_exc())
        # Return original path on failure, ensure final_output_path is sensible if possible
        return final_output_path or input_item_path, False

    finally:
        # --- 8. Cleanup ---
        # Use the context manager's cleanup by exiting the 'with' block if temp_dir_obj was used
        if temp_dir_obj:
            try:
                temp_dir_obj.cleanup()
                logging.info(f"[{thread_name}] Cleaned up temporary directory: {temp_dir_path}")
            except Exception as e:
                 # Log error but don't crash if cleanup fails
                 logging.error(f"[{thread_name}] Error cleaning up temporary directory {temp_dir_path}: {e}")
        elif temp_dir_path and temp_dir_path.exists(): # Manual cleanup if context manager wasn't used (shouldn't happen now)
             logging.warning(f"[{thread_name}] Attempting manual cleanup of temp dir (context manager recommended): {temp_dir_path}")
             shutil.rmtree(temp_dir_path, ignore_errors=True)


# == Input Path Identification and Preparation ==
def identify_and_prepare_inputs(input_path_str: str) -> List[Tuple[pathlib.Path, bool]]:
    """
    Identifies input type and returns a list of items to process.
    Each item is a tuple: (pathlib.Path, is_archive_flag).
    """
    input_path = pathlib.Path(input_path_str).resolve() # Resolve to absolute path
    items_to_process: List[Tuple[pathlib.Path, bool]] = []
    archive_extensions = {'.zip', '.cbz', '.cbr'}
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp'} # Define once

    if not input_path.exists():
        logging.error(f"Input path does not exist: {input_path}")
        return items_to_process # Empty list

    if input_path.is_file():
        if input_path.suffix.lower() in archive_extensions:
            # Check if CBR processing is enabled
            if input_path.suffix.lower() == '.cbr' and not RARFILE_AVAILABLE:
                 logging.warning(f"Skipping CBR file '{input_path.name}' because rarfile/unrar is not available.")
            else:
                items_to_process.append((input_path, True))
                logging.info(f"Identified single archive file: {input_path.name}")
        else:
            logging.warning(f"Input is a file but not a supported archive ({archive_extensions}): {input_path.name}. Skipping.")

    elif input_path.is_dir():
        logging.info(f"Scanning input directory: {input_path}")
        # Check if it's a folder *containing* images (process as single item)
        # Or a folder *containing* archives/folders (process items inside)
        contains_images = False
        contains_archives = False
        contains_subfolders = False
        nested_items: List[pathlib.Path] = []

        try:
            for item in input_path.iterdir():
                # Skip hidden files/folders
                if item.name.startswith('.'):
                    continue

                if item.is_file():
                    ext = item.suffix.lower()
                    if ext in image_extensions:
                        contains_images = True
                    elif ext in archive_extensions:
                        # Check if CBR processing is enabled
                        if ext == '.cbr' and not RARFILE_AVAILABLE:
                             logging.warning(f"Skipping nested CBR file '{item.name}' because rarfile/unrar is not available.")
                        else:
                            contains_archives = True
                            nested_items.append(item)
                    # else: ignore other file types
                elif item.is_dir():
                    # Check if the subdirectory itself contains images before adding it
                    # This prevents adding empty folders or folders with non-image content
                    if any(f.is_file() and f.suffix.lower() in image_extensions for f in item.iterdir()):
                        contains_subfolders = True
                        nested_items.append(item)
                    else:
                        logging.info(f"Skipping subfolder '{item.name}' as it doesn't appear to contain supported image files.")

        except OSError as e:
             logging.error(f"Error reading directory {input_path}: {e}")
             return items_to_process # Return empty list on directory read error


        # --- Decide how to process the directory ---
        if contains_images and not contains_archives and not contains_subfolders:
             # Scenario 1: Folder contains only images (directly)
             items_to_process.append((input_path, False))
             logging.info(f"Identified single folder of images: {input_path.name}")

        elif (contains_archives or contains_subfolders):
             # Scenario 2: Folder contains archives and/or processable subfolders
             # Process the items found inside, ignore any loose images at the top level
             logging.info(f"Identified container folder: {input_path.name}. Processing contents ({len(nested_items)} items).")
             if contains_images:
                 logging.warning(f"Folder '{input_path.name}' contains loose images alongside archives/subfolders. Loose images will be ignored.")

             for item in nested_items:
                 item_is_archive = item.is_file() # Already filtered by archive_extensions
                 item_is_folder = item.is_dir()   # Already filtered to contain images

                 if item_is_archive:
                      items_to_process.append((item, True))
                 elif item_is_folder:
                      items_to_process.append((item, False))

        elif not contains_images and not contains_archives and not contains_subfolders:
            # Scenario 3: Folder is empty or contains only unsupported files/empty subfolders
            logging.warning(f"Input folder seems empty or contains no processable items: {input_path.name}. Nothing to process.")

        # Note: The case contains_images AND (contains_archives OR contains_subfolders) is handled by Scenario 2.

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
    parser.add_argument("--online-search", action="store_true", default=True, help="Enable online search during title extraction if DB lookup fails.") # Keep default True? Or make explicit?
    parser.add_argument("--no-online-search", action="store_false", dest="online_search", help="Disable online search during title extraction.") # Add explicit disable flag
    parser.add_argument("--zip-output", action="store_true", help="Zip the final output folder into a .cbz file.")
    parser.add_argument("--delete-folder", action="store_true", help="Delete the output folder after creating the .cbz file (requires --zip-output).")
    parser.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 1, help="Number of parallel worker threads for processing multiple items.")
    parser.add_argument("-v", "--verbose", action="store_const", dest="loglevel", const=logging.DEBUG, default=logging.INFO, help="Increase output verbosity to DEBUG level.")
    parser.add_argument("-q", "--quiet", action="store_const", dest="loglevel", const=logging.WARNING, help="Decrease output verbosity to WARNING level.")


    args = parser.parse_args()

    # --- Configure Logging Level ---
    logging.getLogger().setLevel(args.loglevel) # Set root logger level

    # --- Validate Arguments ---
    if args.delete_folder and not args.zip_output:
        parser.error("--delete-folder requires --zip-output to be enabled.")

    # --- Determine Output Base Directory ---
    if args.output_base:
        output_base_dir = pathlib.Path(args.output_base)
    else:
        # Default to ./output relative to script or cwd
        script_dir = pathlib.Path(__file__).parent if "__file__" in locals() else pathlib.Path.cwd()
        output_base_dir = (script_dir / "output").resolve() # Default to an 'output' subdir, resolved to absolute

    try:
        output_base_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Output base directory set to: {output_base_dir}")
    except OSError as e:
        logging.error(f"Failed to create or access output directory '{output_base_dir}': {e}")
        sys.exit(1)


    # --- Identify Items to Process ---
    items_to_process = identify_and_prepare_inputs(args.input_path)
    num_items = len(items_to_process)

    if num_items == 0:
        logging.info("No processable items found. Exiting.")
        sys.exit(0)

    logging.info(f"Found {num_items} item(s) to process.")
    # Limit jobs if fewer items than requested jobs
    num_workers = min(args.jobs, num_items)
    if num_workers < 1: num_workers = 1 # Ensure at least one worker
    logging.info(f"Using {num_workers} worker thread(s).")


    # --- Process Items ---
    results = {"success": [], "failed": []}
    overall_start_time = time.time()

    if num_items == 1:
        # Process single item directly without ThreadPoolExecutor for simpler debugging
        logging.info("Processing single item in main thread...")
        item_path, is_archive = items_to_process[0]
        final_path, success = process_single_item(item_path, output_base_dir, args, is_archive)
        if success:
            results["success"].append(final_path)
        else:
            results["failed"].append(item_path) # Report original path on failure
    else:
        # Process multiple items in parallel
        logging.info("Processing multiple items in parallel...")
        with ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix='Worker') as executor:
            # Submit tasks
            futures = {executor.submit(process_single_item, item_path, output_base_dir, args, is_archive): (item_path, is_archive)
                       for item_path, is_archive in items_to_process}

            # Process results as they complete with progress bar
            try:
                 # Use tqdm only if log level is INFO or lower, otherwise it might interfere with WARNING/ERROR logs
                 progress_bar = tqdm(as_completed(futures), total=num_items, desc="Overall Progress", disable=args.loglevel > logging.INFO)
                 for future in progress_bar:
                     original_path, _ = futures[future] # Get original path from the mapping
                     progress_bar.set_postfix_str(f"Processing {original_path.name}", refresh=True)
                     try:
                         final_path, success = future.result()
                         if success:
                             results["success"].append(final_path)
                         else:
                             results["failed"].append(original_path) # Report original path
                         # Update progress bar description after completion (optional)
                         # progress_bar.set_postfix_str(f"Finished {original_path.name}", refresh=True)
                     except Exception as exc:
                         logging.error(f"Item {original_path.name} generated an unhandled exception during processing: {exc}")
                         results["failed"].append(original_path)
                         # Optionally log traceback for unexpected errors
                         # import traceback
                         # logging.error(traceback.format_exc())

            except KeyboardInterrupt:
                 logging.warning("\nCtrl+C detected. Attempting graceful shutdown...")
                 # Shutdown executor - cancel pending futures (Python 3.9+)
                 # Set cancel_futures=True if available and desired
                 try:
                     executor.shutdown(wait=False, cancel_futures=True) # Requires Python 3.9+
                     logging.info("Pending tasks cancelled.")
                 except TypeError: # cancel_futures not available before 3.9
                     executor.shutdown(wait=False)
                     logging.info("Executor shut down. Running tasks may continue.")
                 print("Processing interrupted by user.", file=sys.stderr)
                 # Report partial results below


    # --- Final Summary ---
    overall_end_time = time.time()
    total_duration = overall_end_time - overall_start_time
    summary_level = logging.INFO if not results["failed"] else logging.WARNING
    logging.log(summary_level, "\n--- Processing Summary ---")
    logging.log(summary_level, f"Total time: {total_duration:.2f} seconds")
    logging.log(summary_level, f"Successfully processed: {len(results['success'])} items")
    for path in results["success"]:
        # Check if path exists before reporting suffix (might have been deleted if CBZ failed after folder deletion attempt)
        status_suffix = ""
        if path.exists():
             status_suffix = ' (CBZ)' if path.suffix.lower() == '.cbz' else ' (Folder)'
        elif path.suffix.lower() == '.cbz': # Infer if it was meant to be a CBZ
             status_suffix = ' (CBZ - File Missing!)'
        logging.log(summary_level, f"  - {path}{status_suffix}")

    logging.log(summary_level, f"Failed to process: {len(results['failed'])} items")
    for path in results["failed"]:
        logging.log(summary_level, f"  - {path.name}")

    if results["failed"]:
         logging.warning("Some items failed processing. Check logs above for details.")
         sys.exit(1) # Exit with error code if failures occurred
    else:
         logging.info("All items processed successfully.")
         sys.exit(0)

# --- END OF FILE process_manga.py ---