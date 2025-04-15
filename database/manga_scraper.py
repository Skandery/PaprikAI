#!/usr/bin/env python3
"""
Manga Scraper and Database Builder

This script scrapes manga data from manga-news.com and organizes it into a SQL database.
It traverses alphabetical listings, extracts series information (including synopsis and status),
and collects volume data including cover images and detailed metadata.

Features:
- Scrapes series metadata (title, author, genre, synopsis, status, etc.)
- Collects volume information (including detailed metadata and synopsis) and cover images from editions pages
- Organizes data in a structured SQLite database based on specified schema
- Implements rate limiting to avoid server overload
- Handles errors and edge cases gracefully
- Provides detailed logging
- Tracks scraping status (pending, complete, failed) in the database
- Avoids re-scraping completed items
- Allows retrying failed items
- Configurable scraping speed (delay)
"""

import os
import re
import time
import json
import logging
import sqlite3
import argparse
import requests
from bs4 import BeautifulSoup, Tag
from urllib.parse import urljoin, quote_plus, urlparse, parse_qs
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime

# --- Constants for Scrape Status ---
STATUS_PENDING = "pending"
STATUS_COMPLETE = "complete"
STATUS_FAILED = "failed"
# ---

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s', # Added funcName
    handlers=[
        logging.FileHandler("manga_scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MangaScraper:
    """
    A class to scrape manga data from manga-news.com and build a database.
    Includes status tracking and retry mechanisms.
    """

    BASE_URL = "https://www.manga-news.com"
    SERIES_INDEX_URL = f"{BASE_URL}/index.php/series/"

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    DEFAULT_REQUEST_DELAY = 1.5

    def __init__(self, db_path: str = "manga_db/manga.db", images_dir: str = "manga_db/images", request_delay: float = DEFAULT_REQUEST_DELAY):
        """
        Initialize the scraper.

        Args:
            db_path: Path to the SQLite database file.
            images_dir: Directory to store manga cover images.
            request_delay: Delay between HTTP requests in seconds.
        """
        self.db_path = db_path
        self.images_dir = images_dir
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
        self.last_request_time = 0
        self.request_delay = request_delay

        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)
        logger.info(f"Database directory: {os.path.dirname(db_path)}")
        logger.info(f"Images directory: {images_dir}")
        logger.info(f"Request delay set to: {self.request_delay} seconds")

        self._init_database()

    def _init_database(self):
        """Initialize the SQLite database, adding status and multi-author columns if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # --- Enhanced Series Table Schema ---
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS series (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            url TEXT UNIQUE NOT NULL,
            title_vo TEXT,
            title_translated TEXT,
            artist TEXT,                 -- Will store delimited names
            artist_urls TEXT,            -- Added: Delimited URLs
            writer TEXT,                 -- Will store delimited names
            writer_urls TEXT,            -- Added: Delimited URLs
            translator TEXT,
            publisher_fr TEXT,
            collection TEXT,
            type TEXT,
            genre TEXT,
            publisher_original TEXT,
            pre_publication TEXT,
            illustration TEXT,
            origin TEXT,
            synopsis TEXT,
            volumes_vf_count TEXT,
            volumes_vo_count TEXT,
            status_vf TEXT,
            status_vo TEXT,
            scrape_status TEXT DEFAULT 'pending',
            last_scrape_attempt DATETIME,
            last_scrape_error TEXT
        )
        ''')

        # --- Enhanced Volumes Table Schema ---
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS volumes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            series_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            volume_number TEXT,
            cover_image_path TEXT,
            cover_image_url TEXT,
            url TEXT UNIQUE NOT NULL,
            title_vo TEXT,
            title_translated TEXT,
            artist TEXT,                 -- Will store delimited names
            artist_urls TEXT,            -- Added: Delimited URLs
            writer TEXT,                 -- Will store delimited names
            writer_urls TEXT,            -- Added: Delimited URLs
            translator TEXT,
            publisher_fr TEXT,
            collection TEXT,
            type TEXT,
            genre TEXT,
            publisher_original TEXT,
            pre_publication TEXT,
            publication_date TEXT,
            page_count TEXT,
            illustration TEXT,
            origin TEXT,
            ean_code TEXT,
            price_code TEXT,
            synopsis TEXT,
            scrape_status TEXT DEFAULT 'pending',
            last_scrape_attempt DATETIME,
            last_scrape_error TEXT,
            FOREIGN KEY (series_id) REFERENCES series (id)
        )
        ''')

        # Add columns if they don't exist (for backward compatibility)
        def add_column_if_not_exists(table, column, definition):
            try:
                # Check if column exists by trying to select it
                # Using PRAGMA is generally more reliable
                cursor.execute(f"PRAGMA table_info({table})")
                columns = [info[1] for info in cursor.fetchall()]
                if column in columns:
                     logger.debug(f"Column '{column}' already exists in table '{table}'.")
                     return # Column exists, do nothing
                else:
                     logger.info(f"Adding column '{column}' to table '{table}'.")
                     cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

            except sqlite3.Error as e:
                 # This might happen if the initial check fails unexpectedly
                 logger.error(f"Error checking/adding column '{column}' in table '{table}': {e}. Attempting add anyway.")
                 try:
                     cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
                     logger.info(f"Adding column '{column}' to table '{table}' (after error).")
                 except sqlite3.OperationalError as add_e:
                     # If adding also fails, it might already exist or be another issue
                     logger.warning(f"Could not add column '{column}' to table '{table}', it might already exist or there's another issue: {add_e}")


        # Status columns
        add_column_if_not_exists("series", "scrape_status", f"TEXT DEFAULT '{STATUS_PENDING}'")
        add_column_if_not_exists("series", "last_scrape_attempt", "DATETIME")
        add_column_if_not_exists("series", "last_scrape_error", "TEXT")
        add_column_if_not_exists("volumes", "scrape_status", f"TEXT DEFAULT '{STATUS_PENDING}'")
        add_column_if_not_exists("volumes", "last_scrape_attempt", "DATETIME")
        add_column_if_not_exists("volumes", "last_scrape_error", "TEXT")

        # New author/artist URL columns
        add_column_if_not_exists("series", "artist_urls", "TEXT")
        add_column_if_not_exists("series", "writer_urls", "TEXT")
        add_column_if_not_exists("volumes", "artist_urls", "TEXT")
        add_column_if_not_exists("volumes", "writer_urls", "TEXT")


        conn.commit()
        conn.close()
        logger.info("Database initialized/updated with status tracking and multi-author URL columns.")



    def _rate_limit(self):
        """Enforces request delay."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.request_delay:
            sleep_time = self.request_delay - time_since_last_request
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def _request_page(self, url: str) -> Optional[BeautifulSoup]:
        """Make a GET request with rate limiting and error handling."""
        self._rate_limit()
        logger.info(f"Requesting URL: {url}")
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            if 'html' not in response.headers.get('Content-Type', '').lower():
                logger.warning(f"Non-HTML content type received from {url}: {response.headers.get('Content-Type')}")
                # Treat as failure for scraping purposes, but maybe not a network error
                return None # Or raise a specific exception? Returning None seems safer.
            return BeautifulSoup(response.text, 'html.parser')
        except requests.exceptions.Timeout:
            logger.error(f"Timeout error requesting {url}")
            return None
        except requests.exceptions.HTTPError as e:
             logger.error(f"HTTP error requesting {url}: {e.response.status_code} {e.response.reason}")
             return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error requesting {url}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error parsing {url}: {str(e)}", exc_info=True)
            return None

    def _execute_db_query(self, query: str, params: tuple = (), fetch_one: bool = False, fetch_all: bool = False, commit: bool = False) -> Any:
        """Helper function for database operations."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=10) # Added timeout
            cursor = conn.cursor()
            cursor.execute(query, params)
            if commit:
                conn.commit()
                return cursor.lastrowid if "INSERT" in query.upper() else cursor.rowcount
            if fetch_one:
                return cursor.fetchone()
            if fetch_all:
                return cursor.fetchall()
            return None # Default case if no fetch/commit specified
        except sqlite3.Error as e:
            logger.error(f"Database error executing query '{query}' with params {params}: {e}", exc_info=True)
            if conn:
                 conn.rollback() # Rollback on error
            return None # Indicate failure
        finally:
            if conn:
                conn.close()

    def _update_scrape_status(self, table: str, item_url: str, status: str, error_message: Optional[str] = None):
        """Updates the scrape status, timestamp, and error message for an item."""
        now_iso = datetime.now().isoformat()
        query = f"""
            UPDATE {table}
            SET scrape_status = ?, last_scrape_attempt = ?, last_scrape_error = ?
            WHERE url = ?
        """
        params = (status, now_iso, error_message, item_url)
        self._execute_db_query(query, params, commit=True)
        logger.debug(f"Updated status for {table} item {item_url} to '{status}'")

    def _get_item_status(self, table: str, item_url: str) -> Optional[str]:
        """Gets the current scrape status of an item by URL."""
        query = f"SELECT scrape_status FROM {table} WHERE url = ?"
        result = self._execute_db_query(query, (item_url,), fetch_one=True)
        return result[0] if result else None

    def _get_or_create_pending_series(self, series_title: str, series_url: str) -> Optional[int]:
        """Gets series ID if exists, otherwise creates a pending entry."""
        query_select = "SELECT id, scrape_status FROM series WHERE url = ?"
        result = self._execute_db_query(query_select, (series_url,), fetch_one=True)

        if result:
            series_id, status = result
            logger.debug(f"Series found in DB: URL={series_url}, ID={series_id}, Status={status}")
            return series_id # Return existing ID regardless of status (status check happens later)
        else:
            logger.info(f"Creating new pending entry for series: '{series_title}' ({series_url})")
            query_insert = """
                INSERT INTO series (title, url, scrape_status, last_scrape_attempt)
                VALUES (?, ?, ?, ?)
            """
            now_iso = datetime.now().isoformat()
            params = (series_title, series_url, STATUS_PENDING, now_iso)
            new_id = self._execute_db_query(query_insert, params, commit=True)
            if new_id:
                 logger.info(f"Pending series entry created with ID: {new_id}")
                 return new_id
            else:
                 logger.error(f"Failed to create pending entry for series: {series_url}")
                 return None


    def _get_or_create_pending_volume(self, series_id: int, volume_title: str, volume_url: str, cover_image_url: str) -> Optional[int]:
        """Gets volume ID if exists, otherwise creates a pending entry."""
        query_select = "SELECT id, scrape_status FROM volumes WHERE url = ? AND series_id = ?"
        result = self._execute_db_query(query_select, (volume_url, series_id), fetch_one=True)

        if result:
            volume_id, status = result
            logger.debug(f"Volume found in DB: URL={volume_url}, ID={volume_id}, Status={status}")
            return volume_id
        else:
            logger.info(f"Creating new pending entry for volume: '{volume_title}' ({volume_url}) for series ID {series_id}")
            query_insert = """
                INSERT INTO volumes (series_id, title, url, cover_image_url, scrape_status, last_scrape_attempt)
                VALUES (?, ?, ?, ?, ?, ?)
            """
            now_iso = datetime.now().isoformat()
            params = (series_id, volume_title, volume_url, cover_image_url, STATUS_PENDING, now_iso)
            new_id = self._execute_db_query(query_insert, params, commit=True)
            if new_id:
                logger.info(f"Pending volume entry created with ID: {new_id}")
                return new_id
            else:
                logger.error(f"Failed to create pending entry for volume: {volume_url}")
                return None

    # --- SCRAPING METHODS (with status updates) ---

    def scrape_alphabetical_index(self, letter: str = "") -> List[Tuple[str, str]]:
        """
        Scrape series titles/URLs from index.
        (No fundamental change needed here, discovery happens first)
        """
        # ... (keep existing implementation) ...
        url = f"{self.SERIES_INDEX_URL}{letter}"
        soup = self._request_page(url)
        if not soup:
            return []

        series_list = []
        series_table = soup.select_one('#seriesList tbody')
        if not series_table:
            logger.warning(f"Could not find series table (#seriesList tbody) on {url}")
            return []

        rows = series_table.find_all('tr', recursive=False)
        logger.debug(f"Found {len(rows)} rows in the series table.")

        for row in rows:
            title_cell = row.select_one('td.title div.list-1-item a')
            if title_cell and title_cell.get('href'):
                series_url = urljoin(self.BASE_URL, title_cell['href'])
                title_span = title_cell.select_one('span.item-list-content-title')
                if title_span:
                    series_title = title_span.text.strip()
                elif title_cell.get('title'):
                     series_title = title_cell['title'].strip()
                else:
                    series_title = title_cell.text.strip()

                if series_title and series_url:
                    series_list.append((series_title, series_url))
                    logger.debug(f"Found series: '{series_title}' - {series_url}")
                else:
                     logger.warning(f"Could not extract title or URL from row: {row.prettify()}")
            else:
                logger.debug(f"Skipping row, could not find title cell link: {row.prettify()}")

        logger.info(f"Found {len(series_list)} series for index '{letter or 'non-alpha'}'")
        return series_list


    def _extract_entry_infos(self, info_list_element: Optional[Tag]) -> Dict[str, str]:
        """Helper to extract data from a <ul class="entryInfos"> list, handling multiple artists/writers."""
        data = {}
        if not info_list_element:
            logger.debug("Info list element not provided or not found.")
            return data

        # Map French labels to database columns
        key_map = {
            "titre vo": "title_vo",
            "titre traduit": "title_translated",
            "dessin": "artist",
            "scénario": "writer",
            "auteur": "artist", # Handle cases where only 'Auteur' is listed (will populate both artist/writer if needed)
            "traducteur": "translator",
            "editeur vf": "publisher_fr",
            "collection": "collection",
            "type": "type",
            "genre": "genre",
            "editeur vo": "publisher_original",
            "prépublication": "pre_publication",
            "illustration": "illustration",
            "origine": "origin",
            "date de publication": "publication_date",
            "nombre de pages": "page_count", # Sometimes present
            "code ean": "ean_code",
            "code prix": "price_code"
        }
        # Define the delimiter for multi-valued fields
        MULTI_VALUE_DELIMITER = ";; "

        list_items = info_list_element.find_all('li', recursive=False)
        logger.debug(f"Found {len(list_items)} items in entryInfos list.")

        # --- First pass: Handle specific 'Dessin' and 'Scénario' ---
        processed_keys = set() # Keep track of keys processed for multi-authors
        for item in list_items:
            strong_tag = item.find('strong')
            if not strong_tag:
                continue

            key_raw = strong_tag.text.strip().rstrip(':').lower()
            db_key = key_map.get(key_raw)

            if not db_key:
                logger.debug(f"Unmapped entryInfos key: '{key_raw}'")
                continue

            # --- Special handling for Artist ('Dessin') and Writer ('Scénario') ---
            if db_key in ["artist", "writer"] and key_raw != "auteur": # Process specific keys first
                links = item.find_all('a', href=True) # Find all links with href
                if links:
                    names = []
                    urls = []
                    for link in links:
                        name = link.text.strip()
                        raw_url = link.get('href')
                        if name and raw_url:
                            names.append(name)
                            urls.append(urljoin(self.BASE_URL, raw_url)) # Make URL absolute

                    if names: # Only update if we found valid names/urls
                        data[db_key] = MULTI_VALUE_DELIMITER.join(names)
                        data[db_key + '_urls'] = MULTI_VALUE_DELIMITER.join(urls)
                        logger.debug(f"Extracted multiple '{db_key}': Names='{data[db_key]}', URLs='{data[db_key + '_urls']}'")
                        processed_keys.add(db_key) # Mark as processed
                    else:
                        # Fallback if links found but no valid name/url extracted
                        value = item.get_text(separator=' ', strip=True).replace(strong_tag.text, '').strip().lstrip(':').strip()
                        if value:
                            data[db_key] = value
                            logger.debug(f"Extracted single '{db_key}' (fallback): '{value}'")
                            processed_keys.add(db_key)

                else:
                    # No links found, extract text as single value
                    value = item.get_text(separator=' ', strip=True).replace(strong_tag.text, '').strip().lstrip(':').strip()
                    if value:
                        data[db_key] = value
                        logger.debug(f"Extracted single '{db_key}' (no links): '{value}'")
                        processed_keys.add(db_key)

        # --- Second pass: Handle other keys and the 'Auteur' fallback ---
        for item in list_items:
            strong_tag = item.find('strong')
            if not strong_tag: continue
            key_raw = strong_tag.text.strip().rstrip(':').lower()
            db_key = key_map.get(key_raw)
            if not db_key: continue

            # Skip if already processed as multi-author or if it's a multi-author key we handle separately
            if db_key in processed_keys or (db_key in ["artist", "writer"] and key_raw != "auteur"):
                continue

            # Extract value - handle different structures (text, links)
            value = ""
            links = item.find_all('a')
            spans = item.find_all('span', class_='entry-data-wrapper')

            # --- Handle 'Auteur' (only if specific artist/writer not already found) ---
            if key_raw == "auteur":
                links = item.find_all('a', href=True)
                if links:
                    names = []
                    urls = []
                    for link in links:
                        name = link.text.strip()
                        raw_url = link.get('href')
                        if name and raw_url:
                            names.append(name)
                            urls.append(urljoin(self.BASE_URL, raw_url))

                    if names:
                        names_str = MULTI_VALUE_DELIMITER.join(names)
                        urls_str = MULTI_VALUE_DELIMITER.join(urls)
                        # Populate artist only if not already set
                        if "artist" not in data and "artist" not in processed_keys:
                            data["artist"] = names_str
                            data["artist_urls"] = urls_str
                            logger.debug(f"Extracted 'auteur' as artist: Names='{names_str}', URLs='{urls_str}'")
                        # Populate writer only if not already set
                        if "writer" not in data and "writer" not in processed_keys:
                            data["writer"] = names_str
                            data["writer_urls"] = urls_str
                            logger.debug(f"Extracted 'auteur' as writer: Names='{names_str}', URLs='{urls_str}'")
                else:
                    # Fallback for 'Auteur' if no links
                    value = item.get_text(separator=' ', strip=True).replace(strong_tag.text, '').strip().lstrip(':').strip()
                    if value:
                        if "artist" not in data and "artist" not in processed_keys:
                            data["artist"] = value
                            logger.debug(f"Extracted single 'auteur' as artist (no links): '{value}'")
                        if "writer" not in data and "writer" not in processed_keys:
                            data["writer"] = value
                            logger.debug(f"Extracted single 'auteur' as writer (no links): '{value}'")
                continue # Handled 'auteur', move to next item

            # --- Handle other keys (Genre, Date, EAN, Illustration, etc.) ---
            # (Keep the existing logic for these keys, slightly adjusted for clarity)
            elif db_key == "genre" and links:
                value = ", ".join(a.text.strip() for a in links if a.text.strip())
            elif db_key == "publication_date" and links:
                 value = links[0].text.strip()
            elif db_key == "ean_code":
                 ean_span = item.find('span', itemprop='isbn')
                 if ean_span: value = ean_span.text.strip()
                 elif spans: value = spans[0].get_text(strip=True)
                 else: value = item.get_text(separator=' ', strip=True).replace(strong_tag.text, '').strip().lstrip(':').strip()
            elif db_key == "illustration":
                 page_span = item.find('span', itemprop='numberOfPages')
                 page_count_str = ""
                 if page_span: page_count_str = page_span.text.strip()
                 elif re.search(r'(\d+)\s*pages', item.get_text(), re.IGNORECASE):
                     page_count_str = re.search(r'(\d+)\s*pages', item.get_text(), re.IGNORECASE).group(1)

                 if page_count_str.isdigit():
                     data["page_count"] = page_count_str
                     # Get remaining text
                     value = item.get_text(separator=' ', strip=True).replace(strong_tag.text, '').replace(page_count_str, '').replace('pages', '').strip().lstrip(':').strip()
                 else:
                     value = item.get_text(separator=' ', strip=True).replace(strong_tag.text, '').strip().lstrip(':').strip()
                     # Try extracting page count from the value itself if not found earlier
                     if not data.get("page_count"):
                         page_match_val = re.search(r'(\d+)\s*pages', value, re.IGNORECASE)
                         if page_match_val:
                             data["page_count"] = page_match_val.group(1)
                             value = value.replace(page_match_val.group(0),'').strip()

            elif links: # General case: use first link text if available
                value = links[0].text.strip()
            elif spans: # Use span text if available
                value = spans[0].get_text(strip=True)
            else: # Fallback: get all text in the <li> after the <strong>
                value = item.get_text(separator=' ', strip=True).replace(strong_tag.text, '').strip().lstrip(':').strip()

            # Assign the extracted value if it's not empty
            if value:
                data[db_key] = value
                logger.debug(f"Extracted '{db_key}': '{value}'")

        return data

    def scrape_series_data(self, series_url: str) -> Optional[Dict[str, Any]]:
        """Scrape data for a specific series page."""
        # This function *only* scrapes, it doesn't save. Saving happens in _save_series_to_db
        soup = self._request_page(series_url)
        if not soup:
            # Request failure handled by _request_page logging
            return None

        logger.info(f"Extracting series data from {series_url}")
        series_data = {"url": series_url} # Start with the URL

        # Extract title from H1
        title_element = soup.find('h1')
        series_data["title"] = title_element.text.strip() if title_element else "Unknown Series Title"
        logger.info(f"Series Title (H1): {series_data['title']}")
        if not series_data["title"] or series_data["title"] == "Unknown Series Title":
            logger.warning(f"Could not determine primary title for {series_url}. Using placeholder.")

        # Extract metadata from the info list
        info_list = soup.select_one('#topinfo ul.entryInfos')
        if info_list:
            extracted_infos = self._extract_entry_infos(info_list)
            logger.debug(f"Extracted info list data: {extracted_infos}")
            series_data.update(extracted_infos)
        else:
            logger.warning(f"Could not find main info list (#topinfo ul.entryInfos) on {series_url}")

        # Extract Synopsis
        synopsis_heading = soup.find('h2', string=re.compile(r'R[ée]sum[ée]|Synopsis', re.IGNORECASE)) # Added Synopsis
        if synopsis_heading:
            synopsis_content = []
            # Look for divs or p tags immediately following the heading
            for sibling in synopsis_heading.find_next_siblings():
                if sibling.name == 'h2': # Stop if we hit the next heading
                     break
                if sibling.name == 'div' and ('bigsize' in sibling.get('class', []) or 'synopsis' in sibling.get('class', [])):
                    p_tags = sibling.find_all('p')
                    if p_tags:
                        synopsis_content.extend(p.get_text(strip=True) for p in p_tags if p.get_text(strip=True))
                    else:
                        text = sibling.get_text(strip=True)
                        if text: synopsis_content.append(text)
                elif sibling.name == 'p': # Sometimes synopsis is just in p tags
                     text = sibling.get_text(strip=True)
                     if text: synopsis_content.append(text)

            if synopsis_content:
                 series_data["synopsis"] = "\n".join(synopsis_content)
                 logger.info(f"Extracted synopsis (length: {len(series_data['synopsis'])}).")
            else:
                 logger.warning(f"Found synopsis heading but couldn't extract content from subsequent siblings on {series_url}")
                 series_data["synopsis"] = ""
        else:
            logger.warning(f"Could not find synopsis heading on {series_url}")
            series_data["synopsis"] = ""


        # Extract Volume Counts and Status from #numberblock
        number_block = soup.select_one('#numberblock')
        series_data["volumes_vf_count"] = ""
        series_data["status_vf"] = ""
        series_data["volumes_vo_count"] = ""
        series_data["status_vo"] = ""

        if number_block:
            logger.debug(f"Found numberblock") # Optional: add .prettify() if needed: {number_block.prettify()}")
        
            # Iterate through all spans with class "version" to find VF and VO data
            for version_span in number_block.find_all('span', class_='version'):
                version_text = version_span.text.strip().lower()
                # parent_div = version_span.find_parent('div') # No longer needed for the primary logic
                # if not parent_div: continue # No longer needed
        
                if 'vf' in version_text:
                    # --- Logic adapted from the working script ---
                    # VF data: number and status are expected in sibling spans
                    number_span = version_span.find_next_sibling('span') # Find the immediate next span sibling
                    if number_span:
                        series_data["volumes_vf_count"] = number_span.text.strip()
                        # Status is the next sibling of the number_span with class 'small'
                        status_span = number_span.find_next_sibling('span', class_='small')
                        if status_span:
                            series_data["status_vf"] = status_span.text.strip('() ')
                    logger.debug(f"VF extracted: count={series_data['volumes_vf_count']}, status={series_data['status_vf']}")
                    # --- End of adapted logic ---
        
                elif 'vo' in version_text:
                    # --- Logic adapted from the working script ---
                    # VO data: check if inside an <a> tag first
                    parent_a = version_span.find_parent('a')
                    if parent_a:
                        # Extract text from the <a> tag and parse with regex
                        a_text = parent_a.get_text(separator=' ', strip=True)
                        # Regex to find "VO : number" (number can be digits or '?')
                        vo_match = re.search(r'VO\s*:\s*([\d\?]+)', a_text, re.IGNORECASE)
                        # Regex to find text within parentheses for status
                        status_match = re.search(r'\((.*?)\)', a_text)
                        if vo_match:
                            series_data["volumes_vo_count"] = vo_match.group(1).strip()
                        if status_match:
                            series_data["status_vo"] = status_match.group(1).strip()
                        logger.debug(f"VO extracted from <a>: text='{a_text}', count={series_data['volumes_vo_count']}, status={series_data['status_vo']}")
                    else:
                         # Fallback: If not in <a>, extract like VF using siblings
                         number_span = version_span.find_next_sibling('span') # Find the immediate next span sibling
                         if number_span:
                            series_data["volumes_vo_count"] = number_span.text.strip()
                            # Status is the next sibling of the number_span with class 'small'
                            status_span = number_span.find_next_sibling('span', class_='small')
                            if status_span:
                                series_data["status_vo"] = status_span.text.strip('() ')
                            logger.debug(f"VO extracted (non-<a>): count={series_data['volumes_vo_count']}, status={series_data['status_vo']}")
                    # --- End of adapted logic ---
        else:
            logger.warning(f"Could not find volume count/status block (#numberblock) on {series_url}")

        # Log extracted data for debugging
        logger.info(f"Parsed series data for '{series_data['title']}':")
        # for key, value in series_data.items():
        #     if value and key not in ["url", "synopsis"]:
        #         logger.debug(f"  {key}: {value}")
        #     elif key == "synopsis" and value:
        #         logger.debug(f"  {key}: Present (length {len(value)})")
        logger.debug(f"Parsed series data dict: {json.dumps(series_data, indent=2, ensure_ascii=False)}")


        return series_data


    def get_volume_info_from_editions_page(self, series_url: str) -> List[Tuple[str, str, str]]:
        """Gets volume URLs, cover image URLs, and titles from the series' editions page."""
        # ... (keep existing implementation, maybe improve selectors if needed) ...
        parsed_url = urlparse(series_url)
        path_parts = parsed_url.path.strip('/').split('/')
        editions_url = None
        if len(path_parts) >= 2 and path_parts[-2] == 'serie':
            series_slug = path_parts[-1]
            # Try common patterns for editions URL
            possible_paths = [f"/index.php/serie/editions/{series_slug}", f"/serie/editions/{series_slug}.html"]
            for path in possible_paths:
                test_url = urljoin(self.BASE_URL, path)
                logger.debug(f"Attempting editions URL: {test_url}")
                self._rate_limit() # Rate limit before HEAD request
                try:
                    response = self.session.head(test_url, timeout=10, allow_redirects=True)
                    if response.status_code == 200:
                         editions_url = response.url # Use the final URL after redirects
                         logger.info(f"Found valid editions URL: {editions_url}")
                         break
                    else:
                         logger.debug(f"Editions URL attempt {test_url} failed with status {response.status_code}")
                except requests.exceptions.RequestException as e:
                     logger.warning(f"Error checking editions URL {test_url}: {e}")
                # time.sleep(0.5) # Small delay between HEAD requests if needed

            if not editions_url:
                 logger.error(f"Could not determine or verify editions URL for series URL: {series_url}")
                 return []
        else:
            logger.error(f"Could not determine series slug from URL: {series_url}")
            return []


        soup = self._request_page(editions_url)
        if not soup:
            logger.warning(f"Failed to get editions page content: {editions_url}")
            return []

        volume_info_list = []
        logger.info(f"Looking for volumes on editions page: {editions_url}")

        # More robust selectors for volume blocks
        volume_blocks = soup.select('.boxedContent-items .serieVolumesImgBlock, #volumesList .serieVolumesImgBlock, .vols .serieVolumesImgBlock, .chapter-list .serieVolumesImgBlock')
        logger.debug(f"Found {len(volume_blocks)} potential volume blocks.")
        if not volume_blocks:
             logger.warning(f"No volume blocks found matching selectors on {editions_url}. Check page structure.")

        for block in volume_blocks:
            link_tag = block.select_one('a')
            img_tag = block.select_one('a img')

            if link_tag and link_tag.get('href') and img_tag and img_tag.get('src'):
                raw_volume_page_url = link_tag['href']
                raw_cover_image_url = img_tag['src']

                # Ensure URLs are absolute
                volume_page_url = urljoin(editions_url, raw_volume_page_url) # Base on editions_url for relative links
                cover_image_url = urljoin(editions_url, raw_cover_image_url)

                # Prioritize title attribute on link, fallback to img alt
                volume_title = link_tag.get('title', '').strip()
                if not volume_title:
                    volume_title = img_tag.get('alt', 'Unknown Volume Title').strip()
                    # Clean up common prefixes from alt text
                    volume_title = re.sub(r'^(Manga\s*-\s*Manhwa\s*-\s*)', '', volume_title, flags=re.IGNORECASE).strip()
                    volume_title = re.sub(r'^Couverture\s*', '', volume_title, flags=re.IGNORECASE).strip()


                # Basic validation: Check if it looks like a volume page URL
                if '/manga/' in volume_page_url or '/volume-' in volume_page_url:
                    # Check for duplicates based on volume_page_url
                    if not any(v[0] == volume_page_url for v in volume_info_list):
                         volume_info_list.append((volume_page_url, cover_image_url, volume_title))
                         logger.debug(f"Found volume: '{volume_title}' -> {volume_page_url} | Img: {cover_image_url}")
                    else:
                         logger.debug(f"Skipping duplicate volume URL: {volume_page_url}")
                else:
                    logger.debug(f"Skipping non-manga/volume link found in block: {volume_page_url}")
            else:
                 logger.debug(f"Skipping block, missing link href or image src: {block.prettify()[:200]}...")

        logger.info(f"Found {len(volume_info_list)} unique volume links on editions page.")
        return volume_info_list


    def scrape_volume_data(self, volume_url: str, series_id: int, cover_image_url: str, initial_volume_title: str) -> Optional[Dict[str, Any]]:
        """Scrape data for a specific volume page and plan image download."""
        # This function *only* scrapes, it doesn't save. Saving happens in _save_volume_to_db

        soup = self._request_page(volume_url)
        if not soup:
            return None # Request failed

        logger.info(f"Extracting data for volume: {initial_volume_title} ({volume_url})")

        # Determine best title (H1 often more complete)
        volume_title = initial_volume_title
        h1_title_element = soup.find('h1')
        if h1_title_element:
            h1_title = h1_title_element.text.strip()
            # Use H1 if it contains the initial title or if initial title was generic
            if (initial_volume_title in h1_title and len(h1_title) > len(initial_volume_title)) or \
               initial_volume_title == "Unknown Volume Title":
                 logger.debug(f"Refining volume title using H1: '{h1_title}'")
                 volume_title = h1_title
            elif not volume_title: # Fallback if initial was empty
                 volume_title = h1_title

        if not volume_title or volume_title == "Unknown Volume Title":
             logger.warning(f"Could not determine volume title for {volume_url}. Using placeholder.")
             volume_title = f"Volume at {volume_url}" # Ensure some title exists

        logger.info(f"Final Volume Title: {volume_title}")

        # Extract volume number (more robustly)
        volume_number = ""
        # Try specific patterns first
        patterns = [
            r'(?:Vol\.?|Tome|Volume|#)\s*([\d\.]+[A-Z]?)\b', # Matches 1, 01, 15.5, 1A etc.
            r'\b(\d+)\b' # Match any number as fallback
        ]
        for pat in patterns:
            vol_match = re.search(pat, volume_title, re.IGNORECASE)
            if vol_match:
                volume_number = vol_match.group(1).strip()
                break # Found one

        # Check for special cases if no number found
        if not volume_number:
             if re.search(r'Intégrale', volume_title, re.IGNORECASE): volume_number = "Intégrale"
             elif re.search(r'One-?shot|Récit complet', volume_title, re.IGNORECASE): volume_number = "One-shot"
             elif re.search(r'Art-?book', volume_title, re.IGNORECASE): volume_number = "Artbook"
             elif re.search(r'Light Novel|Roman', volume_title, re.IGNORECASE): volume_number = "Light Novel" # Or Roman?

        if volume_number:
            logger.info(f"Extracted volume number: {volume_number}")
        else:
            logger.info(f"Could not extract specific volume number/type from title: '{volume_title}'")


        # Initialize volume data dict
        volume_data = {
            "series_id": series_id,
            "title": volume_title,
            "volume_number": volume_number,
            "url": volume_url,
            "cover_image_url": cover_image_url,
            "cover_image_path": "", # Will be set after successful download
            "synopsis": "",
            # Initialize other fields to avoid KeyErrors later if extraction fails
            "title_vo": "", "title_translated": "", "artist": "", "writer": "", "translator": "",
            "publisher_fr": "", "collection": "", "type": "", "genre": "", "publisher_original": "",
            "pre_publication": "", "publication_date": "", "page_count": "", "illustration": "",
            "origin": "", "ean_code": "", "price_code": ""
        }


        # Extract metadata from the info list
        content_area = soup.select_one('#content, #main') # Try common content wrappers
        info_list = content_area.select_one('ul.entryInfos') if content_area else soup.select_one('ul.entryInfos') # Fallback search anywhere

        if info_list:
            extracted_infos = self._extract_entry_infos(info_list)
            logger.debug(f"Extracted volume info list data: {extracted_infos}")
            volume_data.update(extracted_infos) # Update volume_data with extracted fields
        else:
            logger.warning(f"Could not find main info list (ul.entryInfos) on volume page {volume_url}")

        # Extract Synopsis for the volume
        # Try finding heading first
        synopsis_heading = soup.find(['h2', 'h3'], string=re.compile(r'R[ée]sum[ée]|Synopsis|Pr[ée]sentation|Contenu', re.IGNORECASE))
        synopsis_div = None
        if synopsis_heading:
            # Look for specific description divs or just following divs/paragraphs
            possible_synopsis_containers = synopsis_heading.find_next_siblings(['div', 'p'])
            for container in possible_synopsis_containers:
                if container.name == 'div' and (container.get('itemprop') == 'description' or 'bigsize' in container.get('class', []) or 'synopsis' in container.get('class', [])):
                    synopsis_div = container
                    break
                elif container.name == 'p': # Synopsis might just be in a <p> tag
                    synopsis_div = container
                    break
                elif container.name in ['h2', 'h3']: # Stop if we hit another heading
                    break
        else:
            # Fallback: Look for description itemprop anywhere if no heading found
            synopsis_div = soup.find(itemprop='description')
            if synopsis_div and synopsis_div.name not in ['div', 'p']: # Ensure it's a content tag
                synopsis_div = None


        if synopsis_div:
            p_tags = synopsis_div.find_all('p') if synopsis_div.name == 'div' else []
            if p_tags:
                volume_data["synopsis"] = "\n".join(p.get_text(strip=True) for p in p_tags if p.get_text(strip=True))
            else:
                # Get text directly from the div or p tag
                volume_data["synopsis"] = synopsis_div.get_text(strip=True)

            if volume_data["synopsis"]:
                logger.info(f"Extracted volume synopsis (length: {len(volume_data['synopsis'])}).")
                # logger.debug(f"Synopsis content: {volume_data['synopsis'][:100]}...")
            else:
                 logger.warning(f"Found synopsis container but it was empty on {volume_url}")
        else:
            logger.warning(f"Could not find synopsis heading or container on volume page {volume_url}")

        # Plan image download (actual download happens *after* DB save attempt)
        if cover_image_url:
            # Generate a relatively safe and unique filename
            safe_series_id = str(series_id)
            safe_vol_num = re.sub(r'\W+', '_', volume_number) if volume_number else "vol"
            safe_title_part = re.sub(r'\W+', '_', volume_title.split('-')[0].split(' ')[0])[:15] # Short, safe part of title
            img_extension = os.path.splitext(urlparse(cover_image_url).path)[1] or '.jpg' # Ensure extension
            if not img_extension.startswith('.'): img_extension = '.' + img_extension
            # Limit overall filename length
            base_filename = f"s{safe_series_id}_v{safe_vol_num}_{safe_title_part}"
            max_base_len = 200 - len(img_extension) # Leave space for extension and potential counter
            img_filename = base_filename[:max_base_len] + img_extension

            volume_data["_image_planned_path"] = os.path.join(self.images_dir, img_filename) # Store planned path
            logger.debug(f"Planned image path: {volume_data['_image_planned_path']}")
        else:
            logger.info("No cover image URL provided or found for this volume.")
            volume_data["_image_planned_path"] = None

        # Log extracted data for debugging
        logger.info("Parsed volume data:")
        # for key, value in volume_data.items():
        #     if not key.startswith('_') and value and key not in ["url", "series_id", "cover_image_path", "cover_image_url", "synopsis"]:
        #         logger.debug(f"  {key}: {value}")
        #     elif key == "synopsis" and value:
        #          logger.debug(f"  {key}: Present (length {len(value)})")
        #     elif key in ["cover_image_path", "cover_image_url"] and value:
        #          logger.debug(f"  {key}: {value}")
        logger.debug(f"Parsed volume data dict: {json.dumps({k:v for k,v in volume_data.items() if not k.startswith('_')}, indent=2, ensure_ascii=False)}")


        return volume_data


    def _download_image(self, img_url: str, save_path: str) -> bool:
        """Download an image with rate limiting, return success status."""
        if os.path.exists(save_path):
            logger.info(f"Image already exists, skipping download: {save_path}")
            return True # Already exists is considered success for linking

        # Check if directory exists before downloading
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        self._rate_limit()
        logger.info(f"Downloading image: {img_url} -> {save_path}")
        try:
            response = self.session.get(img_url, stream=True, timeout=45)
            response.raise_for_status()
            content_type = response.headers.get('Content-Type', '').lower()
            if not content_type.startswith('image/'):
                 logger.error(f"Failed to download image {img_url}: Content-Type is not image ({content_type})")
                 return False

            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            # Verify file size after download (optional but good)
            if os.path.getsize(save_path) > 0:
                logger.info(f"Image downloaded successfully: {save_path}")
                return True
            else:
                logger.error(f"Image downloaded but file is empty: {save_path}. Deleting.")
                os.remove(save_path)
                return False

        except requests.exceptions.Timeout:
            logger.error(f"Timeout downloading image {img_url}")
            return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading image {img_url}: {str(e)}")
            return False
        except IOError as e:
             logger.error(f"IO error saving image {img_url} to {save_path}: {str(e)}")
             return False
        except Exception as e:
            logger.error(f"Unexpected error saving image {img_url} to {save_path}: {str(e)}", exc_info=True)
            # Clean up potentially corrupted file
            if os.path.exists(save_path):
                try: os.remove(save_path)
                except OSError: pass
            return False

    # --- DATABASE SAVING METHODS (with status updates) ---

    def _save_series_to_db(self, series_data: Dict[str, Any]) -> Tuple[Optional[int], bool]:
        """
        Save full series data to the database, updating status. Includes multi-author URLs.

        Returns:
            Tuple: (series_id, was_successful)
        """
        # ... (keep existing initial checks) ...
        if not series_data or "url" not in series_data:
            logger.error("Cannot save series: Invalid data provided (missing url).")
            return None, False

        series_url = series_data["url"]
        series_title = series_data.get("title", "Title N/A") # Use for logging
        now_iso = datetime.now().isoformat()

        # Fields corresponding to the series table schema (including new URL fields)
        fields = [
            "title", "url", "title_vo", "title_translated",
            "artist", "artist_urls", # Added artist_urls
            "writer", "writer_urls", # Added writer_urls
            "translator", "publisher_fr", "collection", "type", "genre",
            "publisher_original", "pre_publication", "illustration", "origin",
            "synopsis", "volumes_vf_count", "volumes_vo_count", "status_vf", "status_vo",
            "scrape_status", "last_scrape_attempt", "last_scrape_error"
        ]

        # Prepare data tuple for update/insert
        data_values = {f: series_data.get(f, None) for f in fields} # Use None for missing values
        data_values["scrape_status"] = STATUS_COMPLETE # Mark as complete on successful save
        data_values["last_scrape_attempt"] = now_iso
        data_values["last_scrape_error"] = None # Clear previous error on success

        # Ensure None values are handled correctly by DB (or use empty string if preferred)
        for key, value in data_values.items():
            if value is None:
                data_values[key] = "" # Or keep as None if DB handles it

        update_fields = ", ".join(f"{f} = :{f}" for f in fields if f != 'url') # Use named placeholders
        query = f"""
            UPDATE series
            SET {update_fields}
            WHERE url = :url
        """

        rows_affected = self._execute_db_query(query, data_values, commit=True)

        # ... (keep existing success/failure logic) ...
        if rows_affected is not None and rows_affected > 0:
            logger.info(f"Successfully updated series data in DB: '{series_title}' ({series_url})")
            # Fetch the ID after update
            series_id_result = self._execute_db_query("SELECT id FROM series WHERE url = ?", (series_url,), fetch_one=True)
            series_id = series_id_result[0] if series_id_result else None
            return series_id, True
        elif rows_affected == 0:
             logger.error(f"Failed to update series - record not found despite prior check/creation: {series_url}")
             self._update_scrape_status("series", series_url, STATUS_FAILED, "DB update failed - record missing")
             return None, False
        else: # rows_affected is None (DB error)
            logger.error(f"Database error occurred while updating series: '{series_title}' ({series_url})")
            self._update_scrape_status("series", series_url, STATUS_FAILED, "DB update failed")
            return None, False


    def _save_volume_to_db(self, volume_data: Dict[str, Any]) -> Tuple[Optional[int], bool]:
        """
        Save full volume data, attempt image download, update status. Includes multi-author URLs.

        Returns:
            Tuple: (volume_id, was_successful)
        """
        # ... (keep existing initial checks and image download logic) ...
        if not volume_data or "url" not in volume_data or "series_id" not in volume_data:
             logger.error("Cannot save volume: Invalid data provided (missing url or series_id).")
             return None, False

        volume_url = volume_data["url"]
        volume_title = volume_data.get("title", "Volume Title N/A")
        series_id = volume_data["series_id"]
        now_iso = datetime.now().isoformat()
        planned_img_path = volume_data.pop("_image_planned_path", None) # Get and remove planned path
        image_download_successful = False
        final_img_path = ""

        # 1. Attempt Image Download (if planned)
        if planned_img_path and volume_data.get("cover_image_url"):
            image_download_successful = self._download_image(volume_data["cover_image_url"], planned_img_path)
            if image_download_successful:
                final_img_path = planned_img_path
                volume_data["cover_image_path"] = final_img_path # Update data dict with actual path
                logger.info(f"Image download successful, path set to: {final_img_path}")
            else:
                logger.warning(f"Image download failed for volume {volume_url}. Path will not be saved in DB.")
                volume_data["cover_image_path"] = "" # Ensure path is empty if download failed
        else:
             volume_data["cover_image_path"] = "" # Ensure path is empty if no image url/plan


        # 2. Save Volume Metadata to DB
        # Fields corresponding to the volumes table (including new URL fields)
        fields = [
            "series_id", "title", "volume_number", "cover_image_path", "cover_image_url",
            "url", "title_vo", "title_translated",
            "artist", "artist_urls", # Added artist_urls
            "writer", "writer_urls", # Added writer_urls
            "translator", "publisher_fr", "collection", "type", "genre", "publisher_original",
            "pre_publication", "publication_date", "page_count", "illustration",
            "origin", "ean_code", "price_code", "synopsis",
            "scrape_status", "last_scrape_attempt", "last_scrape_error"
        ]

        data_values = {f: volume_data.get(f, None) for f in fields} # Use None for missing
        data_values["scrape_status"] = STATUS_COMPLETE # Mark complete after successful scrape/download attempt
        data_values["last_scrape_attempt"] = now_iso
        data_values["last_scrape_error"] = None # Clear error on success

        # Ensure None values are handled correctly by DB (or use empty string if preferred)
        for key, value in data_values.items():
            if value is None:
                data_values[key] = "" # Or keep as None if DB handles it

        update_fields = ", ".join(f"{f} = :{f}" for f in fields if f != 'url' and f != 'series_id') # Use named placeholders
        query = f"""
            UPDATE volumes
            SET {update_fields}
            WHERE url = :url AND series_id = :series_id
        """

        rows_affected = self._execute_db_query(query, data_values, commit=True)

        # ... (keep existing success/failure logic) ...
        if rows_affected is not None and rows_affected > 0:
            logger.info(f"Successfully updated volume data in DB: '{volume_title}' ({volume_url})")
            # Fetch the ID
            volume_id_result = self._execute_db_query("SELECT id FROM volumes WHERE url = ? AND series_id = ?", (volume_url, series_id), fetch_one=True)
            volume_id = volume_id_result[0] if volume_id_result else None
            return volume_id, True
        elif rows_affected == 0:
             logger.error(f"Failed to update volume - record not found despite prior check/creation: {volume_url}")
             self._update_scrape_status("volumes", volume_url, STATUS_FAILED, "DB update failed - record missing")
             return None, False
        else: # rows_affected is None (DB error)
            logger.error(f"Database error occurred while updating volume: '{volume_title}' ({volume_url})")
            self._update_scrape_status("volumes", volume_url, STATUS_FAILED, "DB update failed")
            return None, False

    # --- MAIN WORKFLOW METHODS ---

    def scrape_specific_manga(self, series_title: str, series_url: str, force_rescrape: bool = False) -> bool:
        """
        Scrape a specific manga series and its volumes, respecting status and force flag.
        Handles status updates for series and volumes based on success/failure.
        Returns True if the series was processed (even if some volumes failed), False if the series itself failed critically.
        """
        logger.info(f"--- Processing series: '{series_title}' ({series_url}) ---")

        # 1. Get or create pending series entry & check status
        series_id = self._get_or_create_pending_series(series_title, series_url)
        if series_id is None:
             logger.error(f"Failed to get or create DB entry for series {series_url}. Aborting.")
             # Status should have been logged by _get_or_create... if insert failed
             return False # Critical failure

        current_status = self._get_item_status("series", series_url)

        if current_status == STATUS_COMPLETE and not force_rescrape:
            logger.info(f"Series '{series_title}' is already marked as complete. Skipping. Use --force to rescrape.")
            return True # Considered processed successfully (skipped)
        elif current_status == STATUS_FAILED and not force_rescrape:
             logger.warning(f"Series '{series_title}' is marked as failed. Skipping. Use --retry-failed or --force to rescrape.")
             return True # Processed (skipped due to failure state)
        # Proceed if pending, failed+force, complete+force, or other unknown status

        # 2. Scrape Series Data
        logger.info(f"Attempting to scrape series details for: {series_url}")
        series_scrape_data = None
        series_scrape_error = None
        try:
            series_scrape_data = self.scrape_series_data(series_url)
            if series_scrape_data is None:
                # _request_page handles logging specific HTTP/network errors
                series_scrape_error = "Failed to fetch or parse series page"
                logger.error(f"{series_scrape_error}: {series_url}")
        except Exception as e:
            series_scrape_error = f"Unexpected error during series scrape: {str(e)}"
            logger.error(f"{series_scrape_error}", exc_info=True)

        # 3. Save Series Data (or update status to failed)
        series_save_successful = False
        if series_scrape_data:
            # Ensure the obtained series_id is passed along if needed (though _save updates by URL)
            series_id_from_save, series_save_successful = self._save_series_to_db(series_scrape_data)
            if series_save_successful and series_id_from_save:
                 series_id = series_id_from_save # Update series_id just in case
                 logger.info(f"Series data saved successfully (ID: {series_id}).")
            else:
                 logger.error(f"Failed to save series data to DB for {series_url}.")
                 # Status should have been updated to failed by _save_series_to_db
        else:
            # Scrape failed, update status to failed
            logger.error(f"Series scraping failed for {series_url}. Marking as failed.")
            self._update_scrape_status("series", series_url, STATUS_FAILED, series_scrape_error or "Series scraping returned no data")
            return False # Critical series failure

        # If series save failed, we can't proceed to volumes
        if not series_save_successful:
            return False

        # --- If Series Scrape & Save OK, Proceed to Volumes ---
        logger.info(f"Scraping volumes for series ID {series_id}...")

        # 4. Get Volume Info from Editions Page
        volume_info_list = []
        try:
             volume_info_list = self.get_volume_info_from_editions_page(series_url)
        except Exception as e:
             logger.error(f"Error getting volume list from editions page for {series_url}: {e}", exc_info=True)
             # Mark series as potentially incomplete? Or just log? Let's log and continue.
             # The series *data* was saved, but volume discovery failed.

        if not volume_info_list:
            logger.warning(f"No volumes found or extracted from editions page for {series_title}. Series marked complete.")
            return True # Series is complete, just has no volumes listed/found

        # 5. Process Each Volume
        processed_volumes_count = 0
        successful_volumes_count = 0
        total_volumes_found = len(volume_info_list)

        for vol_idx, (volume_url, cover_image_url, initial_volume_title) in enumerate(volume_info_list):
            logger.info(f"--- Processing Volume {vol_idx + 1}/{total_volumes_found}: '{initial_volume_title}' ---")

            # 5a. Get or create pending volume entry & check status
            volume_id = self._get_or_create_pending_volume(series_id, initial_volume_title, volume_url, cover_image_url)
            if volume_id is None:
                 logger.error(f"Failed to get or create DB entry for volume {volume_url}. Skipping.")
                 continue # Skip this volume

            volume_status = self._get_item_status("volumes", volume_url)

            if volume_status == STATUS_COMPLETE and not force_rescrape:
                logger.info(f"Volume '{initial_volume_title}' is already complete. Skipping.")
                processed_volumes_count += 1
                successful_volumes_count += 1 # Count skipped complete as success
                continue
            elif volume_status == STATUS_FAILED and not force_rescrape:
                logger.warning(f"Volume '{initial_volume_title}' is marked as failed. Skipping. Use --retry-failed or --force.")
                processed_volumes_count += 1 # Count skipped failed as processed
                continue

            # 5b. Scrape Volume Data
            logger.info(f"Attempting to scrape volume details for: {volume_url}")
            volume_scrape_data = None
            volume_scrape_error = None
            try:
                volume_scrape_data = self.scrape_volume_data(volume_url, series_id, cover_image_url, initial_volume_title)
                if volume_scrape_data is None:
                    volume_scrape_error = "Failed to fetch or parse volume page"
                    logger.error(f"{volume_scrape_error}: {volume_url}")
            except Exception as e:
                volume_scrape_error = f"Unexpected error during volume scrape: {str(e)}"
                logger.error(f"{volume_scrape_error}", exc_info=True)

            # 5c. Save Volume Data (or update status to failed)
            if volume_scrape_data:
                # Pass the known series_id
                volume_scrape_data["series_id"] = series_id
                _, volume_save_successful = self._save_volume_to_db(volume_scrape_data)
                if volume_save_successful:
                    logger.info(f"Volume data saved successfully for: {volume_url}")
                    successful_volumes_count += 1
                else:
                    logger.error(f"Failed to save volume data to DB for {volume_url}.")
                    # Status updated by _save_volume_to_db
            else:
                # Scrape failed, update status to failed
                logger.error(f"Volume scraping failed for {volume_url}. Marking as failed.")
                self._update_scrape_status("volumes", volume_url, STATUS_FAILED, volume_scrape_error or "Volume scraping returned no data")

            processed_volumes_count += 1
            # Optional extra delay between volumes? Already have base delay.
            # time.sleep(self.request_delay * 0.5)

        logger.info(f"--- Completed volume processing for series '{series_title}'. Processed: {processed_volumes_count}/{total_volumes_found}. Successful: {successful_volumes_count}/{total_volumes_found}. ---")
        return True # Series processed, return True even if some volumes failed


    def scrape_manga_list(self, series_list: List[Tuple[str, str]], force_rescrape: bool = False):
        """Scrapes a list of (title, url) tuples."""
        total_series = len(series_list)
        processed_count = 0
        success_count = 0
        logger.info(f"Starting processing of {total_series} series...")

        for i, (series_title, series_url) in enumerate(series_list):
             logger.info(f"--- >>> Processing Series {i+1}/{total_series} <<< ---")
             success = self.scrape_specific_manga(series_title, series_url, force_rescrape)
             processed_count += 1
             if success:
                 success_count += 1
             # Add extra delay between processing *different* series
             if i < total_series - 1: # Don't sleep after the last one
                 sleep_duration = self.request_delay * 1.5
                 logger.info(f"--- Delaying {sleep_duration:.2f}s before next series ---")
                 time.sleep(sleep_duration)

        logger.info(f"Finished processing list. Total Series: {total_series}, Processed: {processed_count}, Succeeded (or skipped complete/failed): {success_count}")

    def scrape_all_manga(self, letters: List[str] = None, max_per_letter: Optional[int] = None, force_rescrape: bool = False):
        """Scrape manga series from specified alphabetical indexes."""
        if letters is None:
            # Default: A-Z and '#' for non-alpha
            letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [""] # "" is for non-alpha start

        total_series_processed_overall = 0
        total_series_found_overall = 0

        for letter in letters:
            index_name = f"'{letter}'" if letter else "'non-alphabetical'"
            logger.info(f"===== Scraping manga index: {index_name} =====")

            # 1. Discover series from index page
            series_tuples = self.scrape_alphabetical_index(letter)
            total_series_found_overall += len(series_tuples)

            if not series_tuples:
                logger.info(f"No series found for index {index_name}.")
                continue

            series_to_process = series_tuples
            if max_per_letter is not None and max_per_letter > 0:
                logger.info(f"Limiting to {max_per_letter} series for index {index_name}.")
                series_to_process = series_tuples[:max_per_letter]

            logger.info(f"Processing {len(series_to_process)} series for index {index_name}...")

            # 2. Process the discovered list for this letter
            self.scrape_manga_list(series_to_process, force_rescrape)
            # Note: scrape_manga_list now handles the counting and logging per list

            logger.info(f"===== Finished processing index: {index_name} =====")

        logger.info(f"+++++ Scrape All Manga Complete +++++")
        logger.info(f"Total series URLs found across all specified indexes: {total_series_found_overall}")
        # Note: A detailed count of *successfully* processed series requires tracking return values,
        # but the logs within scrape_manga_list provide this per-letter.

    def list_items_by_status(self, table: str, status: str):
        """Lists items (series or volumes) with a specific status."""
        valid_tables = ["series", "volumes"]
        if table not in valid_tables:
            logger.error(f"Invalid table specified for listing: {table}. Use 'series' or 'volumes'.")
            return

        logger.info(f"--- Listing {table} with status '{status}' ---")
        query = f"SELECT id, title, url, last_scrape_attempt, last_scrape_error FROM {table} WHERE scrape_status = ? ORDER BY last_scrape_attempt DESC"
        items = self._execute_db_query(query, (status,), fetch_all=True)

        if not items:
            logger.info(f"No {table} found with status '{status}'.")
            return

        logger.info(f"Found {len(items)} {table} with status '{status}':")
        for item in items:
            item_id, title, url, attempt_time, error = item
            error_msg = f" | Error: {error}" if error else ""
            attempt_str = f" | Last Attempt: {attempt_time}" if attempt_time else ""
            logger.info(f"  - ID: {item_id}, Title: '{title}', URL: {url}{attempt_str}{error_msg}")
        logger.info(f"--- End of {status} {table} list ---")


    def get_failed_series_urls(self) -> List[Tuple[str, str]]:
        """Gets URLs and titles of series marked as failed."""
        query = f"SELECT title, url FROM series WHERE scrape_status = ? ORDER BY last_scrape_attempt"
        results = self._execute_db_query(query, (STATUS_FAILED,), fetch_all=True)
        return results if results else []

    def retry_failed_series(self, force_rescrape: bool = True):
        """Fetches failed series from DB and attempts to scrape them again."""
        failed_series = self.get_failed_series_urls()
        if not failed_series:
            logger.info("No failed series found in the database to retry.")
            return

        logger.info(f"Found {len(failed_series)} failed series to retry...")
        # force_rescrape=True ensures it retries even if marked failed
        self.scrape_manga_list(failed_series, force_rescrape=force_rescrape)
        logger.info("Finished retrying failed series.")

# --- Main Execution ---
def main():
    """Main function to parse arguments and run the scraper."""
    parser = argparse.ArgumentParser(
        description="Manga Scraper for manga-news.com with status tracking and retries.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show defaults
    )

    # --- Database/Filesystem Arguments ---
    parser.add_argument("--db", default="manga_db/manga.db", help="Path to SQLite database file")
    parser.add_argument("--images", default="manga_db/images", help="Directory to store cover images")

    # --- Scraping Targets ---
    target_group = parser.add_argument_group('Scraping Targets (Choose One)')
    target_group.add_argument("--series-url", help="Scrape a specific series by its full URL")
    target_group.add_argument("--series-title", help="Title for logging (optional, used with --series-url)")
    target_group.add_argument("--letter", help="Scrape series starting with a specific letter (e.g., A)")
    target_group.add_argument("--letters", nargs='+', help="Scrape series for a list of letters (e.g., A B C)")
    target_group.add_argument("--non-alpha", action="store_true", help="Scrape series starting with non-alphabetic chars")
    target_group.add_argument("--all", action="store_true", help="Scrape all series (A-Z & non-alpha). Warning: LONG!")

    # --- Control Arguments ---
    control_group = parser.add_argument_group('Control Options')
    control_group.add_argument("--max", type=int, help="Max series per letter/index (for --letter, --letters, --non-alpha, --all)")
    control_group.add_argument("--delay", type=float, default=MangaScraper.DEFAULT_REQUEST_DELAY, help="Base delay between HTTP requests (seconds)")
    control_group.add_argument("--force", action="store_true", help="Force rescraping of items already marked 'complete' or 'failed'")

    # --- Status/Retry Arguments ---
    status_group = parser.add_argument_group('Status & Retry')
    status_group.add_argument("--list-failed-series", action="store_true", help="List all series marked as 'failed'")
    status_group.add_argument("--list-failed-volumes", action="store_true", help="List all volumes marked as 'failed'")
    status_group.add_argument("--list-pending-series", action="store_true", help="List all series marked as 'pending'")
    status_group.add_argument("--list-pending-volumes", action="store_true", help="List all volumes marked as 'pending'")
    status_group.add_argument("--retry-failed", action="store_true", help="Attempt to rescrape all series marked as 'failed'")


    args = parser.parse_args()

    # --- Initialize Scraper ---
    scraper = MangaScraper(db_path=args.db, images_dir=args.images, request_delay=args.delay)

    # --- Execute Action ---
    action_taken = False

    # Listing actions take priority and exit
    if args.list_failed_series:
        scraper.list_items_by_status("series", STATUS_FAILED)
        action_taken = True
    if args.list_failed_volumes:
        scraper.list_items_by_status("volumes", STATUS_FAILED)
        action_taken = True
    if args.list_pending_series:
         scraper.list_items_by_status("series", STATUS_PENDING)
         action_taken = True
    if args.list_pending_volumes:
         scraper.list_items_by_status("volumes", STATUS_PENDING)
         action_taken = True

    if action_taken:
        logger.info("Listing action complete.")
        return # Exit after listing

    # Retry action
    if args.retry_failed:
        logger.info("--- Starting Retry Failed Series ---")
        scraper.retry_failed_series(force_rescrape=True) # Retry implies forcing past the 'failed' status
        action_taken = True
        logger.info("--- Retry Failed Series Complete ---")
        # Allow other scraping actions after retrying if specified? Or make retry exclusive?
        # Let's make retry exclusive for simplicity for now.
        return

    # Scraping actions
    if args.series_url:
        series_title = args.series_title or f"Series at {args.series_url}"
        scraper.scrape_specific_manga(series_title, args.series_url, force_rescrape=args.force)
        action_taken = True
    elif args.all:
        logger.info("Starting scrape for ALL series...")
        scraper.scrape_all_manga(letters=None, max_per_letter=args.max, force_rescrape=args.force)
        action_taken = True
    elif args.letters:
        logger.info(f"Starting scrape for letters: {', '.join(args.letters)}...")
        scraper.scrape_all_manga(letters=[l.upper() for l in args.letters], max_per_letter=args.max, force_rescrape=args.force)
        action_taken = True
    elif args.letter:
        logger.info(f"Starting scrape for letter: {args.letter.upper()}...")
        scraper.scrape_all_manga(letters=[args.letter.upper()], max_per_letter=args.max, force_rescrape=args.force)
        action_taken = True
    elif args.non_alpha:
        logger.info("Starting scrape for non-alphabetical series...")
        scraper.scrape_all_manga(letters=[""], max_per_letter=args.max, force_rescrape=args.force)
        action_taken = True

    # --- Final Message ---
    if not action_taken:
        parser.print_help()
        logger.info("No action specified. Use a scraping target (--series-url, --letter, etc.) or a status/retry command (--list-*, --retry-failed).")
    else:
        logger.info("Script execution finished.")
        logger.info(f"Database stored at: {os.path.abspath(args.db)}")
        logger.info(f"Images stored in: {os.path.abspath(args.images)}")


if __name__ == "__main__":
    main()