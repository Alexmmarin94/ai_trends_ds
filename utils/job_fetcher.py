import os
import toml
import pandas as pd
import requests
from pathlib import Path
from datetime import datetime
import time

API_HOST = "jsearch.p.rapidapi.com"
BASE_URL = f"https://{API_HOST}/search"
FIELDS_TO_KEEP = [
    "job_id", "job_title", "job_description", "job_posted_at_datetime_utc",
    "job_highlights.Qualifications", "job_highlights.Responsibilities"
]
OUTPUT_PATH = Path("data/job_postings_cleaned/data_scientist_us.csv")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def get_api_key():
    paths = [
        ".env",
        ".streamlit/secrets.toml",
    ]
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
                return secrets["JSEARCH_API_KEY"]
    raise RuntimeError("JSEARCH_API_KEY not found")

headers = {
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

def fetch_us_jobs(queries, max_pages=50, delay_seconds=1.5):
    all_jobs = []
    for query in queries:
        for page in range(1, max_pages + 1):
            params = {"query": query, "page": page, "country": "us", "language": "en"}
            try:
                r = requests.get(BASE_URL, headers=headers, params=params)
                if r.status_code == 429:
                    print(f"⚠️ Rate limit hit for query '{query}' page {page}.")
                    return pd.DataFrame()  # 🔁 Exit early and return empty DataFrame
                r.raise_for_status()
                jobs = r.json().get("data", [])
                if not jobs:
                    break
                for job in jobs:
                    job["extraction_date"] = datetime.today().strftime("%Y-%m-%d")
                all_jobs.extend(jobs)
                time.sleep(delay_seconds)
            except Exception as e:
                print(f"❌ Error on '{query}' page {page}: {e}")
                break
    if not all_jobs:
        print("⚠️ No jobs retrieved from any query.")
        return pd.DataFrame()
    return pd.json_normalize(all_jobs)[FIELDS_TO_KEEP + ["extraction_date"]]
