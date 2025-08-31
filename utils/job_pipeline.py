import os
import numpy as np
import pandas as pd
from pathlib import Path

from utils.job_fetcher import fetch_us_jobs, job_queries
from utils.job_classifier import classify_job
from utils.ai_extraction import extract_ai_mentions

# Paths
OUTPUT_PATH = Path("data/job_postings_cleaned/data_scientist_us.csv")
CLASSIFIED_PATH = Path("data/job_postings_cleaned/data_scientist_us_classified.csv")
AI_ENRICHED_PATH = Path("data/job_postings_cleaned/data_scientist_us_ai_enriched.csv")

# Expected schemas
EXPECTED_BASE_COLS = [
    "job_id",
    "job_title",
    "job_description",
    "job_posted_at_datetime_utc",
    "job_highlights.Qualifications",
    "job_highlights.Responsibilities",
    "extraction_date",
    "is_valid_job",
    "ai_mentions",
    "ai_details",
    "data_mentions",
    "data_details",
]

EXPECTED_FETCH_COLS = [
    "job_id",
    "job_title",
    "job_description",
    "job_posted_at_datetime_utc",
    "job_highlights.Qualifications",
    "job_highlights.Responsibilities",
    "extraction_date",
]

def _ensure_columns(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Ensure columns exist; create as None if missing."""
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df

def run_job_pipeline():
    print("🚀 Starting job pipeline...")

    # --- Load or initialize base dataset
    if OUTPUT_PATH.exists():
        print("📂 Loading existing data from CSV...")
        df_old = pd.read_csv(OUTPUT_PATH, low_memory=False)
    else:
        print("📂 No existing file found. Creating empty dataset...")
        df_old = pd.DataFrame(columns=EXPECTED_BASE_COLS)

    df_old = _ensure_columns(df_old, EXPECTED_BASE_COLS)

    # --- Fetch new jobs
    print("🌐 Fetching new jobs from API...")
    try:
        df_new_raw = fetch_us_jobs(job_queries)
        if not df_new_raw.empty:
            df_new_raw = df_new_raw.reset_index(drop=True)
            df_new_raw = _ensure_columns(df_new_raw, EXPECTED_FETCH_COLS)
    except Exception as e:
        print(f"❌ Failed to fetch jobs from API: {e}")
        df_new_raw = pd.DataFrame(columns=EXPECTED_FETCH_COLS)

    # --- Deduplicate vs historical by job_id + near-duplicate by text (robust positional drop)
    if not df_new_raw.empty:
        # Drop duplicates within the fetched batch by job_id first
        df_new_raw = df_new_raw.drop_duplicates(subset=["job_id"], keep="last").reset_index(drop=True)

        already_ids = set(df_old["job_id"].dropna()) if not df_old.empty else set()
        mask_unseen = ~df_new_raw["job_id"].isin(already_ids)
        df_new = df_new_raw.loc[mask_unseen].copy().reset_index(drop=True)

        if not df_new.empty:
            # Build helper text for TF-IDF dedupe
            df_new["job_title_description"] = (
                df_new["job_title"].fillna("") + " " + df_new["job_description"].fillna("")
            )

            # Ensure positional index 0..n-1 before any positional ops
            df_new = df_new.reset_index(drop=True)

            # Near-duplicate dedupe (positional-safe). If it fails, skip without crashing.
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity

                tfidf = TfidfVectorizer(stop_words="english")
                tfidf_matrix = tfidf.fit_transform(df_new["job_title_description"])
                cos_sim_matrix = cosine_similarity(tfidf_matrix)

                to_drop_pos = set()
                n = len(df_new)
                # Triangular scan; threshold 0.90 as your original logic
                for i in range(n):
                    if i in to_drop_pos:
                        continue
                    sims = cos_sim_matrix[i, i + 1 :]
                    for offset, s in enumerate(sims, start=1):
                        j = i + offset
                        if s >= 0.90:
                            to_drop_pos.add(j)

                if to_drop_pos:
                    keep_mask = np.ones(n, dtype=bool)
                    idx_to_drop = [p for p in to_drop_pos if 0 <= p < n]
                    keep_mask[idx_to_drop] = False
                    df_new = df_new.iloc[keep_mask].reset_index(drop=True)

            except Exception as e:
                print(f"⚠️ Skipping text dedup due to error: {e}")

            # Ensure extraction_date exists (fetcher should set it; fallback to today)
            if "extraction_date" not in df_new.columns or df_new["extraction_date"].isnull().all():
                df_new["extraction_date"] = pd.Timestamp.now().strftime("%Y-%m-%d")

            # Append new rows into base, drop helper col
            before = len(df_old)
            df_new = df_new.drop(columns=["job_title_description"], errors="ignore")
            df_old = pd.concat([df_old, df_new], ignore_index=True)
            added = len(df_old) - before
            print(f"✅ Appended {added} new jobs to base.")
        else:
            print("⚠️ No unseen jobs to process after job_id de-duplication.")
    else:
        print("⚠️ No new jobs retrieved. Continuing with enrichment...")

    # --- Classification
    if CLASSIFIED_PATH.exists():
        df_classified = pd.read_csv(CLASSIFIED_PATH, low_memory=False)
    else:
        df_classified = pd.DataFrame(columns=["job_id", "is_valid_job"])

    already_classified_ids = set(df_classified["job_id"]) if not df_classified.empty else set()
    need_cls_mask = (~df_old["job_id"].isin(already_classified_ids)) | (df_old["is_valid_job"].isnull())
    to_classify = df_old.loc[need_cls_mask].copy()

    if not to_classify.empty:
        print(f"🔍 Classifying {len(to_classify)} jobs missing classification...")
        new_classified = []
        for _, row in to_classify.iterrows():
            try:
                is_valid = classify_job(row.to_dict())
                new_classified.append({"job_id": row["job_id"], "is_valid_job": is_valid})
            except Exception as e:
                print(f"❌ Error classifying job {row['job_id']}: {e}")

        if new_classified:
            df_classified = (
                pd.concat([df_classified, pd.DataFrame(new_classified)], ignore_index=True)
                .drop_duplicates("job_id", keep="last")
            )
            df_classified.to_csv(CLASSIFIED_PATH, index=False)
            print("✅ Classification completed and saved.")
        else:
            print("⚠️ Classification produced no rows (all failed or none needed).")
    else:
        print("✅ No jobs missing classification.")

    # Merge classification back
    df_old = df_old.drop(columns=["is_valid_job"], errors="ignore").merge(
        df_classified, on="job_id", how="left"
    )

    # --- AI/Data enrichment
    if AI_ENRICHED_PATH.exists():
        df_ai = pd.read_csv(AI_ENRICHED_PATH, low_memory=False)
    else:
        df_ai = pd.DataFrame(columns=["job_id", "ai_mentions", "ai_details", "data_mentions", "data_details"])

    already_enriched_ids = set(df_ai["job_id"]) if not df_ai.empty else set()
    enrich_mask = (df_old["is_valid_job"] == True) & (~df_old["job_id"].isin(already_enriched_ids))
    to_enrich = df_old.loc[enrich_mask].copy()

    if not to_enrich.empty:
        print(f"🧠 Extracting AI and Data mentions for {len(to_enrich)} jobs...")
        new_enriched = []
        for idx, row in to_enrich.iterrows():
            try:
                enriched = extract_ai_mentions(row.to_dict())
                new_enriched.append({
                    "job_id": row["job_id"],
                    "ai_mentions": enriched.get("ai_mentions"),
                    "ai_details": enriched.get("ai_details"),
                    "data_mentions": enriched.get("data_mentions"),
                    "data_details": enriched.get("data_details"),
                })
                if (idx + 1) % 10 == 0 or idx == len(to_enrich) - 1:
                    print(f"  ⏳ {idx + 1}/{len(to_enrich)} completed...")
            except Exception as e:
                print(f"❌ Error enriching job {row['job_id']}: {e}")

        if new_enriched:
            df_ai = (
                pd.concat([df_ai, pd.DataFrame(new_enriched)], ignore_index=True)
                .drop_duplicates("job_id", keep="last")
            )
            df_ai.to_csv(AI_ENRICHED_PATH, index=False)
            print("✅ AI & Data enrichment completed and saved.")
        else:
            print("⚠️ Enrichment produced no rows (all failed or none needed).")
    else:
        print("✅ No jobs missing AI enrichment.")

    # Merge enrichment back
    df_old = df_old.drop(
        columns=["ai_mentions", "ai_details", "data_mentions", "data_details"],
        errors="ignore"
    ).merge(df_ai, on="job_id", how="left")

    # --- Save final
    df_old.to_csv(OUTPUT_PATH, index=False)
    print(f"📦 Final dataset saved: {len(df_old)} jobs")
    return df_old
