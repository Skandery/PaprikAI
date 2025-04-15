# --- START OF FILE manga_namer.py ---

import os
import re
import sqlite3
from typing import List, Tuple, Optional, Any

# Image Processing & OCR
from PIL import Image
import imagehash
import pytesseract # Requires Tesseract installation

# Web Requests & Parsing (Optional)
import requests
from bs4 import BeautifulSoup

# String Matching & Date Parsing
from fuzzywuzzy import fuzz
import dateparser
from lingua import Language, LanguageDetectorBuilder
from titlecase import titlecase
import unicodedata

import logging # Use logging for cleaner output control
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Configuration ---

# Database and Image Paths (MODIFY THESE PATHS)
IMAGE_BASE_PATH = r"C:\Paprika2\Projet 2" # Base path where 'images' folder resides
DB_PATH = os.path.join(IMAGE_BASE_PATH, "./manga_db/manga.db")
CURRENT_FILE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
# Matching Thresholds
FUZZY_SERIES_MATCH_THRESHOLD = 85  # Minimum score (0-100) for fuzzy title matching
MAX_SERIES_CANDIDATES = 5         # Max number of series to check covers for after fuzzy match
IMAGE_HASH_DISTANCE_THRESHOLD = 15 # Maximum perceptual hash distance for cover match
NUM_IMAGES_FOR_ISBN_CHECK = 4     # How many images from the end to check for ISBN

# --- Database Helper Functions ---

def get_db_connection(db_path: str) -> Optional[sqlite3.Connection]:
    """Establishes a connection to the SQLite database."""
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row # Return rows as dictionary-like objects
        return conn
    except sqlite3.Error as e:
        logging.error(f"Error connecting to database '{db_path}': {e}")
        return None

def query_volume_by_isbn(conn: sqlite3.Connection, isbn: str) -> Optional[sqlite3.Row]:
    """Finds a volume in the database by its ISBN (ean_code field)."""
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM volumes WHERE ean_code = ?", (isbn,))
        return cursor.fetchone()
    except sqlite3.Error as e:
        logging.error(f"Error querying volume by ISBN {isbn}: {e}")
        return None

def query_volume_by_ean(conn: sqlite3.Connection, ean: str) -> Optional[sqlite3.Row]:
    """Finds a volume in the database by its EAN (ean_code field)."""
    # Assuming EAN Papier and ISBN might both be stored in ean_code
    return query_volume_by_isbn(conn, ean)

def query_series_by_fuzzy_title(conn: sqlite3.Connection, cleaned_title: str, threshold: int, limit: int) -> List[Tuple[int, str, int]]:
    """Finds candidate series by fuzzy matching the title."""
    scores = []
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, title FROM series")
        all_series = cursor.fetchall()

        for series_row in all_series:
            series_id = series_row['id']
            db_title = series_row['title']

            # Use a robust fuzzy matching score
            ratio = fuzz.ratio(cleaned_title.lower(), db_title.lower())
            partial_ratio = fuzz.partial_ratio(cleaned_title.lower(), db_title.lower())
            token_sort_ratio = fuzz.token_sort_ratio(cleaned_title.lower(), db_title.lower())
            token_set_ratio = fuzz.token_set_ratio(cleaned_title.lower(), db_title.lower())
            weighted_score = int((ratio * 0.2) + (partial_ratio * 0.3) + (token_sort_ratio * 0.2) + (token_set_ratio * 0.3))

            if weighted_score >= threshold:
                scores.append((series_id, db_title, weighted_score))

        # Return top N matches sorted by score
        return sorted(scores, key=lambda x: x[2], reverse=True)[:limit]

    except sqlite3.Error as e:
        logging.error(f"Error querying series by fuzzy title: {e}")
        return []

def query_volumes_for_series(conn: sqlite3.Connection, series_id: int) -> List[sqlite3.Row]:
    """Gets all volumes for a given series ID."""
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, title, volume_number, cover_image_path, url FROM volumes WHERE series_id = ?", (series_id,))
        return cursor.fetchall()
    except sqlite3.Error as e:
        logging.error(f"Error querying volumes for series ID {series_id}: {e}")
        return []

def get_series_details(conn: sqlite3.Connection, series_id: int) -> Optional[sqlite3.Row]:
    """Gets full details for a specific series ID."""
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM series WHERE id = ?", (series_id,))
        return cursor.fetchone()
    except sqlite3.Error as e:
        logging.error(f"Error getting series details for ID {series_id}: {e}")
        return None

def get_volume_details(conn: sqlite3.Connection, volume_id: int) -> Optional[sqlite3.Row]:
    """Gets full details for a specific volume ID."""
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM volumes WHERE id = ?", (volume_id,))
        return cursor.fetchone()
    except sqlite3.Error as e:
        logging.error(f"Error getting volume details for ID {volume_id}: {e}")
        return None

# --- Core Logic Functions (Adapted from original script) ---

# Keep necessary functions from the original script, potentially with minor adaptations
# (Assuming these are mostly correct and focus on their specific tasks)

def clean_folder_name(folder_name: str) -> str:
    """Extract the core manga title from a folder name."""
    # First, handle special cases with brackets
    if '[' in folder_name:
        # For formats like "[Compressé] Death Note [Team Chromatique]"
        match = re.search(r']\s*([^[]+?)\s*\[', folder_name)
        if match and match.group(1):
            cleaned = match.group(1).strip()
            # Remove tome/volume information more aggressively
            cleaned = re.sub(r'(?i)\s*-\s*[Tt](?:ome)?\s*#?\d+.*$', '', cleaned)
            cleaned = re.sub(r'(?i)\s+[Tt]\d+.*$', '', cleaned) # Handle "Almark T02" within brackets
            if cleaned:
                return cleaned.strip() # Return early if successful

    # Remove volume/tome indicators (more robustly)
    name = re.sub(r'(?i)\s*-\s*[Tt](?:ome)?\s*#?\d+.*?($|\s|\[|\()', ' ', folder_name)
    name = re.sub(r'(?i)\s+[Tt]\d+\b.*?($|\s|\[|\()', ' ', name) # Match T followed by digits as a word

    # Remove everything in brackets and parentheses AFTER volume removal
    name = re.sub(r'\[.*?\]|\(.*?\)', '', name)

    # Handle dot-separated names like "Ars.Magna"
    if '.' in name and not name.lower().endswith(('.cbz', '.cbr', '.zip', '.rar')):
        # Try to rejoin parts, assuming dots might be spaces
        potential_title = ' '.join(name.split('.'))
        # Re-clean potential volume info introduced by joining
        potential_title = re.sub(r'(?i)\s*-\s*[Tt](?:ome)?\s*#?\d+.*?($|\s|\[|\()', ' ', potential_title)
        potential_title = re.sub(r'(?i)\s+[Tt]\d+\b.*?($|\s|\[|\()', ' ', potential_title)
        name = potential_title.strip()


    # Clean up any remaining special characters and extra spaces
    name = re.sub(r'[._]', ' ', name)  # Replace dots and underscores with spaces
    name = re.sub(r'\s+', ' ', name)  # Normalize spaces

    return name.strip()

def parse_volume_from_foldername(folder_name: str) -> Optional[str]:
    """
    Extracts the volume number from various folder name patterns.
    Returns the volume number as a string, or None if no volume number is found.
    """
    # Check if it's a "One Shot" manga - these don't have volume numbers
    if re.search(r'(?i)\bone[\s-]*shot\b', folder_name):
        return None
    
    # Define patterns to exclude (scan IDs, resolutions, and years)
    excluded_patterns = [
        r'(?i)(?:Scan\s+)?SP[-]?\d+',        # Scan SP-1935
        r'(?i)\b\d{3,4}x\d{3,4}\b',          # 1920x2733 (resolution)
        r'(?i)\b(?:19[8-9]\d|20[0-2]\d)\b'   # Years 1980-2025
    ]
    
    # Primary volume patterns (sorted by confidence)
    volume_patterns = [
        # T01, Tome01 patterns (highest confidence)
        r'(?i)(?:^|\s|\[|\(|-)T(?:ome)?[\s._]?#?(\d{1,3})(?:\b|\s|$|\]|\)|\.)',
        
        # Vol.1 patterns (high confidence)
        r'(?i)(?:^|\s|\[|\()Vol\.?[\s._]?(\d{1,3})(?:\b|\s|$|\]|\)|\.)',
        
        # Inuyashiki style: "-.07" pattern (medium confidence)
        r'(?i)[-\.](\d{1,2})(?:\s|-|$|\]|\)|\.)',
        
        # Bracket volume indicators like [V1] (medium confidence)
        r'(?i)\[V[\._ ]?(\d{1,3})\]',
        
        # Dash or space surrounded numbers like "- 07 -" (lower confidence)
        r'(?i)(?:^|\s|\[|\()[-\s]+(\d{1,2})[-\s]+(?:\s|$|\]|\)|\.)',
    ]
    
    # Process each volume pattern in order of confidence
    for pattern in volume_patterns:
        matches = re.finditer(pattern, folder_name)
        for match in matches:
            # Skip if the match overlaps with any excluded pattern
            is_excluded = False
            for ex_pattern in excluded_patterns:
                if re.search(ex_pattern, match.group(0)):
                    is_excluded = True
                    break
            
            if not is_excluded:
                num_str = match.group(1).lstrip('0')
                num_str = "0" if not num_str else num_str
                vol_num = int(num_str)
                
                # Validate: volume numbers are between 0 and 200
                # Also exclude resolutions (800-4000)
                if 0 <= vol_num <= 200:
                    return num_str
    
    # Fallback for chapter ranges (low confidence, might indicate volume)
    chapter_pattern = r'(?i)[-\s.]\s*C(?:hap)?(?:itre)?\.?\s*(\d{1,3})\s*[-àa]\s*C(?:hap)?(?:itre)?\.?\s*\d+'
    match = re.search(chapter_pattern, folder_name)
    if match:
        num_str = match.group(1).lstrip('0')
        vol_num = int(num_str) if num_str else 0
        if 0 < vol_num <= 200:
            return num_str
    
    return None  # No volume number found

# --- ISBN Extraction (Keep as is) ---
def is_valid_isbn10(isbn):
    if len(isbn) != 10 or not isbn[:-1].isdigit(): return False
    d10 = int(isbn[-1]) if isbn[-1].isdigit() else 10 if isbn[-1].upper() == 'X' else -1
    if d10 == -1: return False
    return (sum((10 - i) * int(isbn[i]) for i in range(9)) + d10) % 11 == 0

def is_valid_isbn13(isbn):
    if len(isbn) != 13 or not isbn.isdigit(): return False
    return sum(int(isbn[i]) * (1 if i % 2 == 0 else 3) for i in range(13)) % 10 == 0

def clean_isbn(candidate):
    return candidate.replace(' ', '').replace('-', '')

def find_isbn(text):
    # Prioritize ISBN prefix
    for match in re.finditer(r'ISBN(?:-1[03])?[-\s]*((?:[\dX]{10})|(?:[\d]{13}))', text, re.IGNORECASE):
        cleaned = clean_isbn(match.group(1))
        if len(cleaned) == 13 and is_valid_isbn13(cleaned): return cleaned
        if len(cleaned) == 10 and is_valid_isbn10(cleaned): return cleaned

    # Regex for potential ISBNs (10 or 13 digits, possibly with hyphens/spaces)
    # 978/979 prefix for ISBN-13 is common
    # ISBN-10 can end in X
    isbn_pattern = r'(?<!\d)(?:(?:97[89][-\s]?)?\d{1,5}[-\s]?\d{1,7}[-\s]?\d{1,6}[-\s]?[\dX])(?!\d)'
    potential_isbns = set() # Use set to avoid re-checking duplicates
    for match in re.finditer(isbn_pattern, text):
         candidate = match.group(0)
         cleaned = clean_isbn(candidate)
         potential_isbns.add(cleaned)

    for cleaned in potential_isbns:
        if len(cleaned) == 13 and is_valid_isbn13(cleaned): return cleaned
        if len(cleaned) == 10 and is_valid_isbn10(cleaned): return cleaned

    return None

def extract_isbn_from_image(image_path: str) -> Optional[str]:
    """Extract ISBN from an image using OCR."""
    try:
        image = Image.open(image_path)
        # Preprocessing can be added here (grayscale, thresholding, etc.) if needed
        # image = image.convert('L')
        text = pytesseract.image_to_string(image, config='--psm 6') # PSM 6: Assume a single uniform block of text.
        isbn = find_isbn(text)
        if isbn:
            logging.info(f"Found potential ISBN: {isbn} in {os.path.basename(image_path)}")
        return isbn
    except FileNotFoundError:
        logging.error(f'Error: Image file "{image_path}" not found.')
        return None
    except Exception as e:
        logging.error(f'Error processing image {image_path} for ISBN: {e}')
        return None

# --- EAN Papier Retrieval (Keep as is, used only if online_search=True) ---
def get_ean_papier(isbn: str) -> Optional[str]:
    """Retrieve the EAN papier for a given ISBN from Numilog."""
    url = f"https://www.numilog.com/Pages/Livres/Fiche.aspx?ISBN={isbn}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        ean_label = soup.find('span', id='ctl02_CentrePage_FicheDetaille1_LabelReference', string='EAN papier')
        if ean_label:
            row = ean_label.find_parent('div').find_parent('div')
            ean_value = row.find('span', id='ctl02_CentrePage_FicheDetaille1_lblReference')
            if ean_value and ean_value.text.strip().isdigit(): # Basic validation
                logging.info(f"Found EAN Papier {ean_value.text.strip()} for ISBN {isbn} on Numilog")
                return ean_value.text.strip()
            else:
                 logging.warning(f"EAN papier value not found or invalid on Numilog for ISBN {isbn}")
                 return None
        else:
            logging.warning(f"EAN papier label not found on Numilog for ISBN {isbn}")
            return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching EAN from Numilog for ISBN {isbn}: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error fetching EAN from Numilog: {e}")
        return None

# --- Manga News URL Retrieval (Keep as is, used only if online_search=True as fallback) ---
def ibsn_to_url(isbn_or_ean: str) -> Optional[str]:
    """Search Manga News for an ISBN or EAN and return the volume URL."""
    search_url = f"https://www.manga-news.com/index.php/recherche/?q={isbn_or_ean}"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36'}
    try:
        search_response = requests.get(search_url, headers=headers, timeout=10)
        search_response.raise_for_status()
        search_soup = BeautifulSoup(search_response.text, 'html.parser')
        # Look specifically for volume results (li.resManga a)
        manga_item = search_soup.select_one('li.resManga a')
        if manga_item and manga_item.get('href'):
            manga_url = manga_item['href']
            logging.info(f"Found Manga News URL {manga_url} for {isbn_or_ean}")
            return manga_url
        else:
            logging.warning(f"No manga volume found on Manga News search for {isbn_or_ean}")
            return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Error searching Manga News for {isbn_or_ean}: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error searching Manga News: {e}")
        return None

# --- Text Formatting Helpers (Keep as is) ---
def extract_year_from_date(date_string: Optional[str]) -> str:
    if not date_string or date_string == "Not found": return "Unknown"
    try:
        # Added settings to prefer French day/month order
        parsed_date = dateparser.parse(date_string, languages=['fr'], settings={'PREFER_DAY_OF_MONTH': 'first'})
        return str(parsed_date.year) if parsed_date else "Unknown"
    except Exception:
        return "Unknown" # Catch potential dateparser errors

# Initialize language detector globally (or within a class if preferred)
# Consider downloading models if not present: lingua-py download --all
try:
    DETECTOR = LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()
    # Load French nouns once
    FRENCH_NOUNS_PATH = os.path.join(CURRENT_FILE_DIRECTORY,"./resources/liste.de.mots.francais.frgut.txt") # Ensure this file exists
    FRENCH_NOUNS = set()
    if os.path.exists(FRENCH_NOUNS_PATH):
        with open(FRENCH_NOUNS_PATH, 'r', encoding='utf-8') as f:
            FRENCH_NOUNS = set(unicodedata.normalize('NFC', word.strip().lower()) for word in f)
    else:
        logging.error(f"Warning: French dictionary '{FRENCH_NOUNS_PATH}' not found. French title correction might be less accurate.")
except Exception as e:
    logging.error(f"Error initializing language detector or loading French nouns: {e}")
    DETECTOR = None
    FRENCH_NOUNS = set()

def correct_title_typo(title: str) -> str:
    if not title or not DETECTOR: return title
    try:
        detected_language = DETECTOR.detect_language_of(title)
        # print(f"Detected language for '{title}': {detected_language}") # Debug

        if detected_language == Language.FRENCH and FRENCH_NOUNS:
            normalized_title = unicodedata.normalize('NFC', title)
            words = normalized_title.split()
            if not words: return title
            corrected_title = [words[0]]
            for word in words[1:]:
                normalized_word = unicodedata.normalize('NFC', word.lower())
                # Lowercase if it's a common noun AND not an acronym (e.g., "ADN")
                if normalized_word in FRENCH_NOUNS and not word.isupper():
                    corrected_title.append(word.lower())
                else:
                    corrected_title.append(word)
            return " ".join(corrected_title)
        elif detected_language == Language.ENGLISH:
            # Apply titlecase, but be careful with existing acronyms
            # Simple titlecase might mess up things like "X-Men" -> "X-Men" (good)
            # but "FBI Files" -> "Fbi Files" (bad). Titlecase lib handles some cases.
            return titlecase(title)
        else:
            return title # Return as is for other languages or if detection fails
    except Exception as e:
        logging.error(f"Error during title correction for '{title}': {e}")
        return title # Return original on error

def correct_series_name_formatting(series_name: str) -> str:
    if not series_name: return "Unknown Series"

    # Capitalize letter after " - "
    if " - " in series_name:
        parts = series_name.split(" - ", 1) # Split only once
        if len(parts) > 1 and parts[1]: # Ensure there is a second part
             # Capitalize first letter of the second part, handle empty string
            parts[1] = parts[1][0].upper() + parts[1][1:] if parts[1] else ""
            series_name = " - ".join(parts)

    # Handle "The " prefix -> " (The)" suffix
    if re.match(r"^The\s+", series_name, re.IGNORECASE):
        series_name = re.sub(r"^The\s+", "", series_name, flags=re.IGNORECASE).strip()
        series_name = f"{series_name} (The)"

    # Handle "(un)" / "(une)" suffix -> prefix
    if re.search(r"\s+\(une\)$", series_name, re.IGNORECASE):
        series_name = re.sub(r"\s+\(une\)$", "", series_name, flags=re.IGNORECASE).strip()
        series_name = "Une " + series_name
    elif re.search(r"\s+\(un\)$", series_name, re.IGNORECASE):
        series_name = re.sub(r"\s+\(un\)$", "", series_name, flags=re.IGNORECASE).strip()
        series_name = "Un " + series_name

    # Capitalize articles in parentheses (specific cases)
    series_name = series_name.replace("(l')", "(L')")
    series_name = series_name.replace("(la)", "(La)")
    series_name = series_name.replace("(le)", "(Le)")
    series_name = series_name.replace("(les)", "(Les)")

     # Apply general typo correction first
    series_name = correct_title_typo(series_name)

    return series_name.strip()

def parse_name(name: str) -> str:
    """Cleans up author/artist names, returning just the last name properly capitalized."""
    if not name: return "Unknown"
    
    # Step 1: Remove anything after common separators like " - ", " / "
    name = re.split(r'\s+-\s+|\s+/\s+', name, 1)[0].strip()

    # Step 2: Split the name into words
    words = name.split()
    
    # If no words found, return "Unknown"
    if not words:
        return "Unknown"
    
    # Step 3: Identify which word is the last name (usually the last word in Japanese names,
    # or words in ALL CAPS are often last names)
    last_name = None
    
    # Check for all caps word which is likely a last name
    for word in words:
        if word.isupper() and len(word) > 1:
            last_name = word
            break
    
    # If no all-caps word found, assume the last word is the last name
    if not last_name:
        last_name = words[-1]
    
    # Step 4: Properly capitalize the last name
    # If it was in ALL CAPS, convert to title case
    if last_name.isupper():
        last_name = last_name.title()
    # Otherwise ensure it's properly capitalized (first letter uppercase, rest lowercase)
    else:
        last_name = last_name[0].upper() + last_name[1:].lower() if last_name else ""
    
    return last_name


def format_authors(author_string: Optional[str]) -> str:
    """Formats author/artist string from DB (delimited by ;;) using only last names."""
    if not author_string:
        return "Unknown"
    
    raw_authors = [a.strip() for a in author_string.split(';;') if a.strip()]
    parsed_authors = [parse_name(author) for author in raw_authors]
    
    # Filter out "Unknown"
    valid_authors = [author for author in parsed_authors if author != "Unknown"]
    
    if valid_authors:
        return "-".join(valid_authors)
    else:
        return "Unknown"

def format_volume_number(volume_num_str: Optional[str]) -> str:
    """Formats volume number as T01, T10, T100 etc."""
    if volume_num_str is None or not volume_num_str.isdigit():
        return "" # Return empty if no valid number
    num = int(volume_num_str)
    if num < 0: return "" # Ignore negative
    if num < 100:
        return f"T{num:02d}"
    else:
        return f"T{num:03d}"

def format_manga_title_from_db(
    series_details: sqlite3.Row,
    volume_details: sqlite3.Row,
    image_width: int,
    upscale: bool = False
) -> str:
    """Formats the final manga filename using data retrieved from the database."""

    # 1. Series Title (Corrected)
    series_title = correct_series_name_formatting(series_details['title'])

    # 2. Volume Number (Formatted)
    volume_part = format_volume_number(volume_details['volume_number'])

    # Combine Series and Volume
    title = f"{series_title} {volume_part}".strip() # Add space only if volume exists

    # 3. Authors (Writer-Artist)
    # Prefer volume-specific authors, fall back to series authors
    writer_str = format_authors(volume_details['writer'] or series_details['writer'])
    artist_str = format_authors(volume_details['artist'] or series_details['artist'])

    # Combine authors, handle same person, handle unknowns
    author_part = ""
    if writer_str != "Unknown" and artist_str != "Unknown":
        if writer_str == artist_str:
            author_part = f"({writer_str})"
        else:
            author_part = f"({writer_str}-{artist_str})"
    elif writer_str != "Unknown":
        author_part = f"({writer_str})"
    elif artist_str != "Unknown":
        author_part = f"({artist_str})"
    # else: author_part remains "" if both are Unknown

    # 4. Publication Year
    pub_date = extract_year_from_date(volume_details['publication_date']) # Use volume date
    year_part = f"({pub_date})" if pub_date != "Unknown" else ""

    # 5. Fixed Parts
    if upscale:
        size_part = f"[Digital-1920u]"
    else:
        size_part = f"[Digital-{image_width}]" if image_width > 0 else "[Digital-X]"

    manga_fr_part = "[Manga FR]"
    source_part = "(Paprika+)" # Assuming this is constant

    # 6. Assemble the final title string
    final_title = f"{title} {author_part} {year_part} {size_part} {manga_fr_part} {source_part}"

    # Clean up extra spaces that might result from missing parts
    final_title = re.sub(r'\s{2,}', ' ', final_title).strip()

    # Remove characters forbidden in filenames
    final_title = re.sub(r'[\\/*?:"<>|]', '', final_title)

    return final_title


# --- Image Matching Function (Using Database) ---

def find_matching_volume_db(
    first_image_path: str,
    candidate_series_ids: List[int],
    conn: sqlite3.Connection,
    image_base_path: str,
    hash_threshold: int
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Finds the best matching volume from DB candidates based on cover image hash.

    Args:
        first_image_path: Path to the cover image from the input folder.
        candidate_series_ids: List of series IDs to check volumes for.
        conn: Active database connection.
        image_base_path: Base directory where DB image paths are relative to.
        hash_threshold: Maximum allowed perceptual hash distance.

    Returns:
        Tuple (best_volume_id, best_series_id, best_distance) or (None, None, None).
    """
    try:
        with Image.open(first_image_path) as test_img:
            test_hash = imagehash.phash(test_img)
    except Exception as e:
        logging.error(f"Error loading or hashing input image {first_image_path}: {e}")
        return None, None, None

    best_match_volume_id = None
    best_match_series_id = None
    best_distance = float('inf')

    logging.info(f"Comparing cover hash {test_hash} against candidates...")

    for series_id in candidate_series_ids:
        volumes = query_volumes_for_series(conn, series_id)
        if not volumes:
            continue

        # print(f"Checking {len(volumes)} volumes for series ID {series_id}...")
        for volume_row in volumes:
            if not volume_row['cover_image_path']:
                continue

            # Construct full path to the local DB cover image
            db_image_rel_path = volume_row['cover_image_path']
            # Ensure the path uses the correct OS separator and remove potential leading separators
            db_image_rel_path = db_image_rel_path.replace('/', os.sep).replace('\\', os.sep).lstrip(os.sep)
            db_image_full_path = os.path.join(image_base_path, db_image_rel_path)


            if not os.path.exists(db_image_full_path):
                # print(f"Skipping missing DB image: {db_image_full_path}")
                continue

            try:
                with Image.open(db_image_full_path) as db_img:
                    volume_hash = imagehash.phash(db_img)
                    distance = test_hash - volume_hash

                    # print(f"  Comparing with {os.path.basename(db_image_full_path)} (Hash: {volume_hash}, Dist: {distance})") # Debug

                    if distance < best_distance:
                        best_distance = distance
                        best_match_volume_id = volume_row['id']
                        best_match_series_id = series_id
                        # print(f"    New best match: Vol ID {best_match_volume_id}, Series ID {best_match_series_id}, Dist {best_distance}") # Debug

            except Exception as e:
                logging.error(f"Error processing DB image {db_image_full_path}: {e}")
                continue # Skip this image

    if best_match_volume_id is not None and best_distance <= hash_threshold:
        logging.info(f"Found best match: Volume ID {best_match_volume_id}, Series ID {best_match_series_id} with distance {best_distance} (Threshold: {hash_threshold})")
        return best_match_volume_id, best_match_series_id, best_distance
    else:
        if best_match_volume_id is not None:
             logging.error(f"No match found within threshold. Best distance was {best_distance} (Threshold: {hash_threshold})")
        else:
             logging.error("No comparable cover images found for candidates.")
        return None, None, None


# --- Main Naming Function ---

def find_name_folder(folder_path: str, online_search: bool = True, upscale: bool = True) -> Optional[str]:
    """
    Determines the standardized manga name for a folder of images using database
    and optionally online lookups.

    Args:
        folder_path (str): Path to the folder containing manga images.
        online_search (bool): Whether to allow online lookups (Numilog, Manga-News)
                               if database lookups fail. Defaults to True.

    Returns:
        str: The formatted manga title string, or None if identification fails.
    """
    if not os.path.isdir(folder_path):
        logging.error(f"Error: Input path is not a valid directory: {folder_path}")
        return None

    folder_name = os.path.basename(folder_path)
    logging.info(f"Processing folder: {folder_name}")

    # List and sort image files (case-insensitive extension check)
    try:
        all_files = [os.path.join(root, file)
                     for root, _, files in os.walk(folder_path)
                     for file in files]
        image_files = sorted([f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]) # Added webp
    except Exception as e:
        logging.error(f"Error listing files in {folder_path}: {e}")
        return None

    if not image_files:
        logging.error(f"No image files found in {folder_path}")
        return None

    logging.info(f"Found {len(image_files)} image files. Using first image for cover/size: {os.path.basename(image_files[0])}")
    first_image_path = image_files[0]

    # Get image width from the first image
    image_width = 0
    try:
        with Image.open(first_image_path) as img:
            image_width = img.width
    except Exception as e:
        logging.error(f"Warning: Could not read first image width from {first_image_path}: {e}")


    conn = get_db_connection(DB_PATH)
    if not conn:
        return None # Cannot proceed without DB

    found_volume_id = None
    found_series_id = None
    found_method = None
    volume_db_url = None # Store the URL from the DB volume entry

    try:
        # --- Strategy 1: ISBN/EAN Lookup ---
        logging.info("\nAttempting ISBN/EAN lookup...")
        isbn_found_on_image = None
        for i in range(min(NUM_IMAGES_FOR_ISBN_CHECK, len(image_files))):
            img_idx = len(image_files) - 1 - i
            img_to_check = image_files[img_idx]
            logging.info(f"Checking image {os.path.basename(img_to_check)} for ISBN...")
            isbn = extract_isbn_from_image(img_to_check)
            if isbn:
                isbn_found_on_image = isbn
                logging.info(f"Found potential ISBN {isbn} on {os.path.basename(img_to_check)}.")
                # 1a: Check ISBN directly in DB
                volume_details = query_volume_by_isbn(conn, isbn)
                if volume_details:
                    found_volume_id = volume_details['id']
                    found_series_id = volume_details['series_id']
                    volume_db_url = volume_details['url']
                    found_method = f"DB ISBN ({isbn})"
                    logging.info(f"Match found in DB via ISBN: Volume ID {found_volume_id}")
                    break # Found via ISBN in DB

                # 1b: Try getting EAN Papier (if online search allowed)
                if not found_volume_id and online_search:
                    ean_papier = get_ean_papier(isbn)
                    if ean_papier:
                        # 1c: Check EAN Papier in DB
                        volume_details = query_volume_by_ean(conn, ean_papier)
                        if volume_details:
                            found_volume_id = volume_details['id']
                            found_series_id = volume_details['series_id']
                            volume_db_url = volume_details['url']
                            found_method = f"DB EAN ({ean_papier} from ISBN {isbn})"
                            logging.info(f"Match found in DB via EAN Papier: Volume ID {found_volume_id}")
                            break # Found via EAN in DB
            if found_volume_id: break # Exit loop if match found

        if not found_volume_id:
             logging.warning("ISBN/EAN lookup did not yield a DB match.")


        # --- Strategy 2: Title Fuzzy Match + Cover Hash Match ---
        if not found_volume_id:
            logging.info("\nAttempting Title + Cover Match...")
            cleaned_title = clean_folder_name(folder_name)
            logging.info(f"Cleaned folder name for matching: '{cleaned_title}'")
            if cleaned_title:
                candidate_series = query_series_by_fuzzy_title(conn, cleaned_title, FUZZY_SERIES_MATCH_THRESHOLD, MAX_SERIES_CANDIDATES)

                if candidate_series:
                    logging.info(f"Found {len(candidate_series)} potential series candidates:")
                    for s_id, s_title, s_score in candidate_series:
                        logging.info(f"  - ID: {s_id}, Title: '{s_title}', Score: {s_score}")

                    candidate_ids = [s[0] for s in candidate_series]
                    best_vol_id, best_series_id, best_dist = find_matching_volume_db(
                        first_image_path, candidate_ids, conn, IMAGE_BASE_PATH, IMAGE_HASH_DISTANCE_THRESHOLD
                    )

                    if best_vol_id is not None:
                        found_volume_id = best_vol_id
                        found_series_id = best_series_id
                        volume_details = get_volume_details(conn, found_volume_id) # Fetch details again
                        volume_db_url = volume_details['url'] if volume_details else None
                        found_method = f"Title Fuzzy Match + Cover Hash (Dist: {best_dist})"
                        logging.info(f"Match found via Title/Cover: Volume ID {found_volume_id}")
                    else:
                        logging.error("Cover hash matching did not find a suitable volume.")
                else:
                    logging.error("No series candidates found via fuzzy title matching.")
            else:
                logging.error("Could not extract a clean title from folder name for matching.")


        # --- Final Step: Format Title if Match Found ---
        if found_volume_id and found_series_id:
            logging.info(f"Match found using method: {found_method}")
            series_details = get_series_details(conn, found_series_id)
            volume_details = get_volume_details(conn, found_volume_id)
            if series_details and volume_details:
                # Validate Volume Number
                parsed_vol_num = parse_volume_from_foldername(folder_name)
                db_vol_num = volume_details['volume_number']

                logging.info(f"Volume number from folder name: '{parsed_vol_num}'")
                logging.info(f"Volume number from database:    '{db_vol_num}'")

                if parsed_vol_num is not None and db_vol_num is not None:
                    # Normalize comparison (e.g., '06' vs '6')
                    if str(parsed_vol_num).lstrip('0') != str(db_vol_num).lstrip('0'):
                        logging.warning(f"WARNING: Volume number mismatch! Folder='{parsed_vol_num}', DB='{db_vol_num}'. Using DB value for title.")
                elif parsed_vol_num is not None and db_vol_num is None:
                     logging.warning(f"WARNING: Volume number found in folder ('{parsed_vol_num}') but not in DB. Using DB value (None) for title.")
                elif parsed_vol_num is None and db_vol_num is not None:
                     logging.warning(f"INFO: Volume number found in DB ('{db_vol_num}') but not parsed from folder name.")
                # Else: both are None or match, which is fine.

                # Format the final title
                formatted_title = format_manga_title_from_db(series_details, volume_details, image_width, upscale)
                logging.info(f"Generated Title: {formatted_title}")
                # Optionally return URL as well if needed later: return formatted_title, volume_db_url
                return formatted_title
            else:
                logging.error("Error: Could not retrieve full series or volume details from DB for the matched IDs.")
                return None
        else:
            logging.error("\nFailed to identify manga volume using all available methods.")
            return None

    except Exception as e:
        logging.error(f"An unexpected error occurred during processing: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        if conn:
            conn.close()
            # print("Database connection closed.")


# --- Example Usage ---


#if __name__ == "__main__":
#    # Ensure the path uses correct separators for your OS and exists
#    example_folder = r"C:\Paprika2\Git3\ComfyUI-Upscaler-Tensorrt - Copie\Un Dragon Dans Ma Cuisine - T04"
#
#    if os.path.exists(example_folder):
#        logging.info(f"--- Running Example for: {example_folder} ---")
#        # Set online_search=False if you want to test DB-only mode
#        final_name = find_name_folder(example_folder, online_search=True)
#
#        if final_name:
#            logging.info(f"--- Example Result ---")
#            logging.info(f"Final Manga Name: {final_name}")
#        else:
#            logging.info(f"--- Example Result ---")
#            logging.error("Could not determine manga name for the example folder.")
#    else:
#        logging.error(f"Error: Example folder path does not exist: {example_folder}")
#        logging.error("Please update the 'example_folder' variable in the script.")
#