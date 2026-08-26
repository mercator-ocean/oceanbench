#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 16:02:10 2026

@author: alequilleuc
"""
# pip install python-dotenv

import os
import re
import logging
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import s3fs
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


from dotenv import load_dotenv


load_dotenv() #load the .env file 

# CREATE .ENV FILE WITH CREDENTIALS TO LOG INTO AVISO AND EDITO


# =============================================================================
# CONFIGURATION
# =============================================================================


AVISO_BASE_URL = "https://tds-odatis.aviso.altimetry.fr"

AVISO_USERNAME = os.getenv("AVISO_USERNAME")
AVISO_PASSWORD = os.getenv("AVISO_PASSWORD")


CLOUDFERRO_ENDPOINT = "https://s3.waw3-1.cloudferro.com"
CLOUDFERRO_REGION = "waw3-1"
TARGET_BUCKET = "oceanbench-bucket"
TARGET_PREFIX = "class4/swot/l3"

cf_key = os.getenv("CLOUDFERRO_KEY")
cf_secret = os.getenv("CLOUDFERRO_SECRET") 

# =============================================================================
# CONFIGURATION S3FS
# =============================================================================

target_storage_options = {
    "key": cf_key,
    "secret": cf_secret,
    "client_kwargs": {
        "endpoint_url": CLOUDFERRO_ENDPOINT,
        "region_name": CLOUDFERRO_REGION,
    },
    "config_kwargs": {
        "s3": {"addressing_style": "path"},
        "max_pool_connections": 64,
        "retries": {"max_attempts": 10},
    },
}
target_fs = s3fs.S3FileSystem(**target_storage_options)


# =============================================================================
# SECONDARY SUPPORT FUNCTIONS
# =============================================================================

def remote_key(prefix):
    """Build the total path for CloudFerro"""
    return f"{TARGET_BUCKET}/{prefix}"

def assert_cloudferro_write_works():
    """Writing test for CloudFerro"""
    test_prefix = f"{TARGET_PREFIX}/_write_test_{os.urandom(4).hex()}.txt"
    path = remote_key(test_prefix)
    try:
        with target_fs.open(path, "wb") as f:
            f.write(b"ok\n")
        target_fs.rm(path)
        logging.info("Writing test on CloudFerro: OK")
        return True
    except Exception as e:
        logging.error(f"Writing test on CloudFerro: {e}")
        return False

def extract_year_from_url(url: str) -> str:
    """Get the year from the name of the SWOT file"""
    filename = Path(url).name
    if match := re.search(r"_(\d{4})\d{4}T\d{6}_", filename):
        return match.group(1)
    raise ValueError(f"Impossible to get the year: {filename}")
    
    

# =============================================================================
# STRUCTURE FOR SWOT FILE ON EDITO / FILTERING FILES TO ADD
# =============================================================================
"""
Explanation:
    We are not taking all the swot l3 files ("Basic" level) from AVISO server
    We are taking only the files from the v3_0
    we will organise file by years in 2 folders: forward and reprocessed
    
    -> v2_0_1 (2023) is outdated
    -> v3_0 / Reprocessed = 2023, 2025(until May)
    -> v3_0 / foward = 2025(from May), 2026, until yesterday
    
Filtering files that should be added on EDITO (avoiding duplicates)
"""


def build_s3_path(url: str) -> str:
    """
    Build a path with the following structure:
    oceanbench-bucket/class4/swot/l3/{forward|reproc}/{year}/{filename}
    """
    filename = Path(url).name
    year = extract_year_from_url(url)

    # Identify the sub-folder (forward or reproc) from the URL:
    if "/Basic/forward/" in url:
        subfolder = "forward"
    elif "/Basic/reproc/" in url:
        subfolder = "reproc"
    else:
        raise ValueError(f"Undertermined (forward/reproc) for: {url}")

    return f"{TARGET_BUCKET}/{TARGET_PREFIX}/{subfolder}/{year}/{filename}"




#list the file already in oceanbench bucket on edito to avoid creating duplicates
def list_existing_remote_keys() -> set:
    """List all the files under TARGET_BUCKET/TARGET_PREFIX/ (including forward/ and reproc/)"""
    base = remote_key(TARGET_PREFIX)
    try:
        keys = set(target_fs.find(base))
    except FileNotFoundError:
        keys = set()
    logging.info(f"Existing files under {base}: {len(keys)}")
    return keys




def filter_new_urls(urls: list, existing_keys: set) -> list:
    """
    Filter URLs (AVISO) to keep only the ones not on EDITO
    following the new file organisation.
    """
    to_transfer = []
    skipped = 0
    
    for url in urls:
        try:
            filename = Path(url).name
            year = extract_year_from_url(url)

            # Identify the subfolder
            if "/Basic/forward/" in url:
                subfolder = "forward"
            elif "/Basic/reproc/" in url:
                subfolder = "reproc"
            else:
                continue  # Ignore if not forward nor reproc

            remote_path = remote_key(f"{TARGET_PREFIX}/{subfolder}/{year}/{filename}")

            if remote_path in existing_keys:
                skipped += 1
            else:
                to_transfer.append(url)

        except ValueError as e:
            logging.warning(f"Ignored (could not read name): {e}")
            continue

    logging.info(f"File already in bucket: {skipped} | To transfer: {len(to_transfer)}")
    return to_transfer



# =============================================================================
# GETTING FILES FROM AVISO
# =============================================================================


def create_aviso_session() -> requests.Session:
    """Create HTTP session with retries for AVISO"""
    session = requests.Session()
    session.auth = (AVISO_USERNAME, AVISO_PASSWORD)
    session.headers.update({"User-Agent": "SWOT-AVISO-to-EDITO/2.0"})
    retry = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=16)
    session.mount("https://", adapter)
    return session



def get_aviso_l3_basic_urls(
    thredds_base_url: str = AVISO_BASE_URL,
    username: str = AVISO_USERNAME,
    password: str = AVISO_PASSWORD,
    version: str = "v3_0",
    sub_folders: tuple[str, ...] = ("forward", "reproc"),
    max_workers: int = 16,
    cycle_names: list[str] | None = None,
    stop_event: threading.Event = None,
) -> list:
    
    
    """Discovering AVISO URLs (forward + reproc)"""
    thredds_base_url = thredds_base_url.rstrip("/")
    session = create_aviso_session()
    all_cycle_infos = []

    try:
        for sub_folder in sub_folders:
            catalog_url = (
                f"{thredds_base_url}/thredds/catalog/"
                f"dataset-l3-swot-karin-nadir-validated/l3_lr_ssh/"
                f"{version}/Basic/{sub_folder}/catalog.xml"
            )
            try:
                response = session.get(catalog_url, timeout=(15, 60))
                response.raise_for_status()
                soup = BeautifulSoup(response.content, "xml")
                for catalog_ref in soup.find_all("catalogRef"):
                    cycle_name = catalog_ref.get("name")
                    href = catalog_ref.get("xlink:href")
                    if not cycle_name or not cycle_name.startswith("cycle_"):
                        continue
                    if cycle_names is not None and cycle_name not in cycle_names:
                        continue
                    if not href:
                        continue
                    cycle_catalog_url = urljoin(catalog_url, href)
                    all_cycle_infos.append((sub_folder, cycle_name, cycle_catalog_url))
            except Exception as e:
                logging.error(f"Error with catalog {sub_folder}: {e}")
    finally:
        session.close()
        
        
    def read_cycle_catalog(cycle_info):
        if stop_event is not None and stop_event.is_set():
            return []
        sub_folder, cycle_name, cycle_catalog_url = cycle_info
        session = create_aviso_session()
        try:
            response = session.get(cycle_catalog_url, timeout=(15, 60))
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "xml")
            urls = []
            for dataset in soup.find_all("dataset"):
                filename = dataset.get("name")
                url_path = dataset.get("urlPath")
                if not filename or not filename.endswith(".nc"):
                    continue
                if not url_path:
                    continue
                file_url = urljoin(
                    f"{thredds_base_url}/thredds/fileServer/",
                    url_path.lstrip("/"),
                )
                urls.append(file_url)
            logging.info(f"📂 {sub_folder}/{cycle_name}: {len(urls)} fichiers")
            return urls
        except Exception as e:
            logging.error(f"Error {sub_folder}/{cycle_name}: {e}")
            return []
        finally:
            session.close()
            
         
    #Multithreading
    
    executor = ThreadPoolExecutor(max_workers=max_workers)
    futures = [executor.submit(read_cycle_catalog, info) for info in all_cycle_infos]
    urls = []
    try:
        for future in as_completed(futures):
            urls.extend(future.result())
            
    except KeyboardInterrupt:
        logging.warning("Interrupted while discovering AVISO files")
        if stop_event is not None:
            stop_event.set()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)

    urls = list(dict.fromkeys(urls))  # Getting rid of duplicates
    logging.info(f"Total AVISO files: {len(urls)}")
    return urls






# =============================================================================
# TRANSFERT FILES TO EDITO (CLOUDFERRO BUCKET)
# =============================================================================


def transfer_to_cloudferro(url: str, stop_event: threading.Event) -> bool:
    """Transfer of 1 single swot l3 file from AVISO to CLOUDFERRO bucket on EDITO"""
    if stop_event.is_set():
        return False

    remote_path = None
    session = None
    try:
        filename = Path(url).name
        year = extract_year_from_url(url)

        # Indentify the subfolder
        if "/Basic/forward/" in url:
            subfolder = "forward"
        elif "/Basic/reproc/" in url:
            subfolder = "reproc"
        else:
            raise ValueError(f"Undertermined type: {url}")

        remote_path = remote_key(f"{TARGET_PREFIX}/{subfolder}/{year}/{filename}")
        session = create_aviso_session()

        with session.get(url, stream=True, timeout=(30, 600)) as response:
            response.raise_for_status()
            size = int(response.headers.get("Content-Length", 0))
            logging.info(f"→ {filename} ({size/1024/1024:.1f} MB) → {remote_path}")

            with target_fs.open(remote_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                    if stop_event.is_set():
                        logging.warning(f"Transfer interrupted: {filename}")
                        raise KeyboardInterrupt
                    if chunk:
                        f.write(chunk)

        logging.info(f"OK: {remote_path}")
        return True

    #interruption during the transfer and the file has been only partially loaded
    # the file is removed from the bucket
    except KeyboardInterrupt:
        if remote_path and target_fs.exists(remote_path):
            try:
                target_fs.rm(remote_path)
                logging.info(f"File partially deleted: {remote_path}")
            except Exception:
                pass
        return False

    except Exception as e:
        logging.error(f"ERROR {url}: {e}")
        return False

    finally:
        if session is not None:
            session.close()
            
            
            
            
            
            

def transfer_all(urls: list, stop_event: threading.Event, max_workers: int = 4) -> dict:
    """Transfer in parallel of all the files to EDITO"""
    success = 0
    failed = 0
    executor = ThreadPoolExecutor(max_workers=max_workers)
    futures = {executor.submit(transfer_to_cloudferro, url, stop_event): url for url in urls}

    try:
        for future in as_completed(futures):
            url = futures[future]
            try:
                if future.result():
                    success += 1
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                logging.error(f"Critical Failure {url}: {e}")
    except KeyboardInterrupt:
        logging.warning("Demand for interruption: suspension of transfers...")
        stop_event.set()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)

    return {"total": len(urls), "transferred": success, "failed": failed}






# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

    stop_event = threading.Event()

    try:
        # 1. Writing test on CloudFerro
        logging.info("Writing test on CloudFerro...")
        if not assert_cloudferro_write_works():
            raise Exception("Failed writing test on CloudFerro")


        # 2. Inventory of EDITO OceanBench swot bucket
        logging.info("Inventory of swot data on EDITO...")
        existing_keys = list_existing_remote_keys()


        # 3. Discovering AVISO files
        logging.info("Discovering AVISO files...")
        urls = get_aviso_l3_basic_urls(
            thredds_base_url=AVISO_BASE_URL,
            username=AVISO_USERNAME,
            password=AVISO_PASSWORD,
            version="v3_0",
            sub_folders=("forward", "reproc"),
            max_workers=16,
            stop_event=stop_event
        )


        # 4. Filter to avoid duplicates
        logging.info("Filtering duplicates...")
        urls = filter_new_urls(urls, existing_keys)

        if not urls:
            logging.info("Nothing to transfer...Everything is already in OceanBench on EDITO")
        
        else:
            # 5. Test with 1 file
            logging.info("Testing transfer of 1 file")
            test_urls = urls[:1]
            result = transfer_all(test_urls, stop_event, max_workers=1)

            if result["failed"] == 0:
                logging.info("Test Completed Successfully! Launching transfer of all files...")
                remaining_urls = urls[1:]  # To avoid transferring again the test file
                full_result = transfer_all(remaining_urls, stop_event, max_workers=4)
                
                result = {
                    "total": result["total"] + full_result["total"],
                    "transferred": result["transferred"] + full_result["transferred"],
                    "failed": result["failed"] + full_result["failed"],
                }
                
            else:
                logging.error("Test failed! Some errors need to be corrected before proceeding")

            logging.info("FINAL RESULT")
            logging.info(f"Total: {result['total']} | Transférés: {result['transferred']} | Échecs: {result['failed']}")

    except KeyboardInterrupt:
        logging.warning("The user interrupted the process")

    except Exception:
        logging.exception("Process failed")