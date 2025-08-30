import os
import toml
import json
import pandas as pd
import requests
from pathlib import Path
from datetime import datetime
import time
import random

API_HOST = "jsearch.p.rapidapi.com"
BASE_URL = f"https://{API_HOST}/search"

# Optional freshness filter if the API supports it: '7days' | '30days' | 'month' | None
DATE_POSTED = os.getenv("JSEARCH_DATE_POSTED", None)

# Hard cap to protect your wallet (can be overridden by env)
MAX_REQUESTS_PER_RUN = int(os.getenv("JSEARCH_MAX_REQUESTS_PER_RUN", "800"))

FIELDS_TO_KEEP = [
    "job_id", "job_title", "job_description", "job_posted_at_datetime_utc",
    "job_highlights.Qualifications", "job_highlights.Responsibilities"
]

OUTPUT_DIR = Path("data/job_postings_cleaned")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STATUS_PATH = OUTPUT_DIR / "fetch_status.json"

def _write_status(**kwargs):
    """Persist a small JSON status report for CI/artifacts."""
    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        **kwargs,
    }
    try:
        with open(STATUS_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ Could not write status file: {e}")

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

    for path in [".env", ".streamlit/secrets.toml"]:
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
    """Best-effort detection of exhausted rate limits using common RapidAPI headers."""
    headers = {k.lower(): v for k, v in resp.headers.items()}
    for k in ("x-ratelimit-requests-remaining", "x-ratelimit-remaining", "ratelimit-remaining"):
        if k in headers:
            try:
                return int(headers[k]) <= 0
            except Exception:
                pass
    return False

def _to_df(all_jobs):
    if not all_jobs:
        return pd.DataFrame(columns=FIELDS_TO_KEEP + ["extraction_date"])
    df = pd.json_normalize(all_jobs)
    cols = FIELDS_TO_KEEP + ["extraction_date"]
    return df[cols] if all(c in df.columns for c in cols) else df

def fetch_us_jobs(queries, max_pages=50, delay_seconds=1.2):
    """
    Cost-aware fetcher:
      - Returns partial results if 429 happens (monthly cap or QPS).
      - Exponential backoff with jitter on transient failures.
      - Writes a status JSON with where/why it stopped.
      - Enforces a hard request budget per run to avoid overspend.
    """
    headers = get_headers()
    all_jobs = []
    requests_used = 0
    last_query = None
    last_page = None

    try:
        for query in queries:
            empty_pages_streak = 0
            for page in range(1, max_pages + 1):
                # Budget guard
                if requests_used >= MAX_REQUESTS_PER_RUN:
                    print(f"ℹ️ Request budget reached ({requests_used}/{MAX_REQUESTS_PER_RUN}). Returning partial results.")
                    df_partial = _to_df(all_jobs).reset_index(drop=True)
                    _write_status(
                        event="budget_cap",
                        last_query=last_query,
                        last_page=last_page,
                        requests_used=requests_used,
                        fetched_rows=len(df_partial),
                        unique_job_ids=int(df_partial["job_id"].nunique()) if "job_id" in df_partial else 0,
                    )
                    return df_partial

                params = {"query": query, "page": page, "country": "us", "language": "en"}
                if DATE_POSTED:
                    params["date_posted"] = DATE_POSTED

                attempt, max_retries = 0, 3
                while attempt <= max_retries:
                    try:
                        resp = requests.get(BASE_URL, headers=headers, params=params, timeout=30)
                        requests_used += 1
                        last_query, last_page = query, page

                        if resp.status_code == 429:
                            print(f"⚠️ 429 on '{query}' page {page} (attempt {attempt+1}/{max_retries+1}).")
                            if _rate_limits_exhausted(resp):
                                print("ℹ️ Rate limit fully exhausted. Returning partial results.")
                                df_partial = _to_df(all_jobs).reset_index(drop=True)
                                _write_status(
                                    event="rate_limit_exhausted",
                                    last_query=query,
                                    last_page=page,
                                    requests_used=requests_used,
                                    fetched_rows=len(df_partial),
                                    unique_job_ids=int(df_partial["job_id"].nunique()) if "job_id" in df_partial else 0,
                                )
                                return df_partial

                            backoff = (2 ** attempt) + random.uniform(0, 0.5)
                            print(f"⏳ Backing off ~{backoff:.1f}s and retrying.")
                            time.sleep(backoff)
                            attempt += 1
                            continue

                        resp.raise_for_status()
                        data = resp.json().get("data", [])
                        if not data:
                            # No more pages for this query
                            empty_pages_streak = 0
                            break

                        extraction_date = datetime.today().strftime("%Y-%m-%d")
                        for job in data:
                            job["extraction_date"] = extraction_date

                        all_jobs.extend(data)
                        empty_pages_streak = 0
                        time.sleep(delay_seconds)
                        break  # success for this page

                    except requests.RequestException as e:
                        backoff = (2 ** attempt) + random.uniform(0, 0.5)
                        print(f"❌ Request error '{query}' p{page} (attempt {attempt+1}): {e} -> retrying in ~{backoff:.1f}s")
                        time.sleep(backoff)
                        attempt += 1
                else:
                    # Retries exhausted for this page -> move to next query
                    print(f"⚠️ Giving up on '{query}' page {page} after retries. Proceeding to next query.")
                    break

        # Completed loop
        df_final = _to_df(all_jobs).reset_index(drop=True)
        _write_status(
            event="completed",
            last_query=last_query,
            last_page=last_page,
            requests_used=requests_used,
            fetched_rows=len(df_final),
            unique_job_ids=int(df_final["job_id"].nunique()) if "job_id" in df_final else 0,
        )
        if df_final.empty:
            print("⚠️ No jobs retrieved across all queries (rate limits or empty results).")
        return df_final

    except Exception as e:
        # Unexpected crash -> still persist what we collected
        df_partial = _to_df(all_jobs).reset_index(drop=True)
        _write_status(
            event="unexpected_error",
            error=str(e),
            last_query=last_query,
            last_page=last_page,
            requests_used=requests_used,
            fetched_rows=len(df_partial),
            unique_job_ids=int(df_partial["job_id"].nunique()) if "job_id" in df_partial else 0,
        )
        print(f"❌ Unexpected error: {e}")
        return df_partial
