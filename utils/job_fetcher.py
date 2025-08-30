import os
import toml
import pandas as pd
import requests
from pathlib import Path
from datetime import datetime
import time
import random

API_HOST = "jsearch.p.rapidapi.com"
BASE_URL = f"https://{API_HOST}/search"

# Optional freshness filter if the API supports it (e.g., '7days', '30days', 'month'); set to None to disable
DATE_POSTED = None

FIELDS_TO_KEEP = [
    "job_id", "job_title", "job_description", "job_posted_at_datetime_utc",
    "job_highlights.Qualifications", "job_highlights.Responsibilities"
]

OUTPUT_PATH = Path("data/job_postings_cleaned/data_scientist_us.csv")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def get_api_key():
    """
    Priority:
    1) Environment variable (CI/CD, Streamlit Cloud, local export)
    2) .env
    3) .streamlit/secrets.toml
    """
    key = os.getenv("JSEARCH_API_KEY")
    if key:
        return key

    paths = [".env", ".streamlit/secrets.toml"]
    for path in paths:
        if os.path.exists(path):
            if path.endswith(".env"):
                from dotenv import load_dotenv
                load_dotenv(path)
                key = os.getenv("JSEARCH_API_KEY")
                if key:
                    return key
            else:
                secrets = toml.load(path)
                if "JSEARCH_API_KEY" in secrets:
                    return secrets["JSEARCH_API_KEY"]

    raise RuntimeError("JSEARCH_API_KEY not found")

def get_headers():
    return {
        "X-RapidAPI-Key": get_api_key(),
        "X-RapidAPI-Host": API_HOST
    }

job_queries = [
    "data scientist in US",
    "data scientist remote",
    "data scientist in new york",
    "data scientist in chicago",
    "data scientist in california",
    "data scientist in Massachusetts",
    "data analyst in US",
    "data analyst remote",
    "data analyst in new york",
    "data analyst in chicago",
    "data analyst in california",
    "data analyst in Massachusetts",
]

def _rate_limits_exhausted(resp) -> bool:
    """
    Best-effort detection of exhausted rate limits using common RapidAPI headers.
    Returns True if headers indicate no remaining requests.
    """
    headers = {k.lower(): v for k, v in resp.headers.items()}
    candidates = [
        "x-ratelimit-requests-remaining",
        "x-ratelimit-remaining",
        "ratelimit-remaining",
    ]
    for k in candidates:
        if k in headers:
            try:
                remaining = int(headers[k])
                if remaining <= 0:
                    return True
            except Exception:
                pass
    return False

def fetch_us_jobs(queries, max_pages=50, delay_seconds=1.5):
    """
    Fetch jobs across multiple queries with pagination.
    Robust to 429 (rate limit): uses exponential backoff and returns partial results collected so far.
    """
    all_jobs = []
    headers = get_headers()

    for query in queries:
        for page in range(1, max_pages + 1):
            params = {"query": query, "page": page, "country": "us", "language": "en"}
            if DATE_POSTED:
                # Use only if supported by the API; otherwise keep as None
                params["date_posted"] = DATE_POSTED

            attempt = 0
            max_retries = 3
            success = False
            no_more_pages = False

            while attempt <= max_retries:
                try:
                    r = requests.get(BASE_URL, headers=headers, params=params, timeout=30)

                    if r.status_code == 429:
                        print(f"⚠️ 429 on '{query}' page {page} (attempt {attempt+1}/{max_retries+1}).")

                        # If headers suggest exhaustion (e.g., monthly cap), return what we already have
                        if _rate_limits_exhausted(r):
                            print("ℹ️ Rate limit appears fully exhausted. Returning partial results collected so far.")
                            if all_jobs:
                                df_partial = pd.json_normalize(all_jobs)[FIELDS_TO_KEEP + ["extraction_date"]]
                                return df_partial
                            else:
                                return pd.DataFrame(columns=FIELDS_TO_KEEP + ["extraction_date"])

                        # Otherwise do exponential backoff with jitter and retry
                        sleep_s = (2 ** attempt) + random.uniform(0, 0.5)
                        print(f"⏳ Backing off for ~{sleep_s:.1f}s before retry.")
                        time.sleep(sleep_s)
                        attempt += 1
                        continue

                    # Non-429 responses
                    r.raise_for_status()
                    data = r.json().get("data", [])
                    if not data:
                        # No more pages for this query
                        no_more_pages = True
                        success = True  # successful call but no more data
                        break

                    # Add extraction date
                    extraction_date = datetime.today().strftime("%Y-%m-%d")
                    for job in data:
                        job["extraction_date"] = extraction_date

                    all_jobs.extend(data)
                    time.sleep(delay_seconds)
                    success = True
                    break

                except requests.RequestException as e:
                    print(f"❌ Request error on '{query}' page {page} (attempt {attempt+1}): {e}")
                    sleep_s = (2 ** attempt) + random.uniform(0, 0.5)
                    time.sleep(sleep_s)
                    attempt += 1

            # After retry loop
            if no_more_pages:
                break  # move to next query
            if not success:
                # Retries exhausted for this page → give up on this query and move on
                print(f"⚠️ Giving up on '{query}' page {page} after retries. Proceeding to next query.")
                break

    if not all_jobs:
        print("⚠️ No jobs retrieved across all queries (rate limits or empty results).")
        return pd.DataFrame(columns=FIELDS_TO_KEEP + ["extraction_date"])

    return pd.json_normalize(all_jobs)[FIELDS_TO_KEEP + ["extraction_date"]]
