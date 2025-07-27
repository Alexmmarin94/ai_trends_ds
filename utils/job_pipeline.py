import os
import time
import pandas as pd
from pathlib import Path

from utils.job_fetcher import fetch_us_jobs, job_queries
from utils.job_classifier import classify_job
from utils.ai_extraction import extract_ai_mentions

# Paths
OUTPUT_PATH = Path("data/job_postings_cleaned/data_scientist_us.csv")
CLASSIFIED_PATH = Path("data/job_postings_cleaned/data_scientist_us_classified.csv")
AI_ENRICHED_PATH = Path("data/job_postings_cleaned/data_scientist_us_ai_enriched.csv")

def run_job_pipeline():
    print("🚀 Starting job pipeline...")

    # Load or create dataset
    if OUTPUT_PATH.exists():
        print("📂 Loading existing data from CSV...")
        df_old = pd.read_csv(OUTPUT_PATH)
    else:
        print("📂 No existing file found. Creating empty dataset...")
        df_old = pd.DataFrame(columns=[
            "job_id", "job_title", "job_description", "job_posted_at_datetime_utc",
            "job_highlights.Qualifications", "job_highlights.Responsibilities",
            "extraction_date", "is_valid_job",
            "ai_mentions", "ai_details", "data_mentions", "data_details"
        ])

    for col in ["is_valid_job", "ai_mentions", "ai_details", "data_mentions", "data_details"]:
        if col not in df_old.columns:
            df_old[col] = None

    print("🌐 Fetching new jobs from API...")
    try:
        df_new_raw = fetch_us_jobs(job_queries)
    except Exception as e:
        print(f"❌ Failed to fetch jobs from API: {e}")
        df_new_raw = pd.DataFrame()

    if not df_new_raw.empty:
        df_new = df_new_raw[~df_new_raw["job_id"].isin(df_old["job_id"])].copy()
        if not df_new.empty:
            df_new["job_title_description"] = df_new["job_title"].fillna("") + " " + df_new["job_description"].fillna("")
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            tfidf = TfidfVectorizer(stop_words="english")
            tfidf_matrix = tfidf.fit_transform(df_new["job_title_description"])
            cos_sim_matrix = cosine_similarity(tfidf_matrix)

            to_drop = set()
            for i in range(len(df_new)):
                for j in range(i + 1, len(df_new)):
                    if cos_sim_matrix[i, j] >= 0.9:
                        to_drop.add(j)

            df_new = df_new.drop(index=list(to_drop)).reset_index(drop=True)
            df_new["extraction_date"] = pd.Timestamp.now().strftime("%Y-%m-%d")
            df_old = pd.concat([df_old, df_new.drop(columns=["job_title_description"], errors="ignore")], ignore_index=True)
        else:
            print("⚠️ No unseen jobs to process after deduplication.")
    else:
        print("⚠️ No new jobs retrieved. Continuing with enrichment...")

    if CLASSIFIED_PATH.exists():
        df_classified = pd.read_csv(CLASSIFIED_PATH)
    else:
        df_classified = pd.DataFrame(columns=["job_id", "is_valid_job"])

    to_classify = df_old[
        ~df_old["job_id"].isin(df_classified["job_id"] if not df_classified.empty else []) |
        df_old["is_valid_job"].isnull()
    ].copy()

    if not to_classify.empty:
        print(f"🔍 Classifying {len(to_classify)} jobs missing classification...")
        new_classified = []

        for _, row in to_classify.iterrows():
            try:
                is_valid = classify_job(row.to_dict())
                new_classified.append({"job_id": row["job_id"], "is_valid_job": is_valid})
            except Exception as e:
                print(f"❌ Error classifying job {row['job_id']}: {e}")

        df_classified = pd.concat([df_classified, pd.DataFrame(new_classified)], ignore_index=True).drop_duplicates("job_id", keep="last")
        df_classified.to_csv(CLASSIFIED_PATH, index=False)
        print("✅ Classification completed and saved.")
    else:
        print("✅ No jobs missing classification.")

    df_old = df_old.drop(columns=["is_valid_job"], errors="ignore").merge(
        df_classified, on="job_id", how="left"
    )

    # Enrichment
    if AI_ENRICHED_PATH.exists():
        df_ai = pd.read_csv(AI_ENRICHED_PATH)
    else:
        df_ai = pd.DataFrame(columns=["job_id", "ai_mentions", "ai_details", "data_mentions", "data_details"])

    to_enrich = df_old[
        (df_old["is_valid_job"] == True) &
        (~df_old["job_id"].isin(df_ai["job_id"] if not df_ai.empty else [])) |
        (df_old["ai_mentions"].isnull())
    ].copy()

    if not to_enrich.empty:
        print(f"🧠 Extracting AI and Data mentions for {len(to_enrich)} jobs...")
        new_enriched = []

        for idx, row in to_enrich.iterrows():
            try:
                enriched = extract_ai_mentions(row.to_dict())
                new_enriched.append({
                    "job_id": row["job_id"],
                    "ai_mentions": enriched["ai_mentions"],
                    "ai_details": enriched["ai_details"],
                    "data_mentions": enriched["data_mentions"],
                    "data_details": enriched["data_details"]
                })
                if (idx + 1) % 10 == 0 or idx == len(to_enrich) - 1:
                    print(f"  ⏳ {idx + 1}/{len(to_enrich)} completed...")
            except Exception as e:
                print(f"❌ Error enriching job {row['job_id']}: {e}")

        df_ai = pd.concat([df_ai, pd.DataFrame(new_enriched)], ignore_index=True).drop_duplicates("job_id", keep="last")
        df_ai.to_csv(AI_ENRICHED_PATH, index=False)
        print("✅ AI & Data enrichment completed and saved.")
    else:
        print("✅ No jobs missing AI enrichment.")

    df_old = df_old.drop(columns=["ai_mentions", "ai_details", "data_mentions", "data_details"], errors="ignore").merge(
        df_ai, on="job_id", how="left"
    )

    df_old.to_csv(OUTPUT_PATH, index=False)
    print(f"📦 Final dataset saved: {len(df_old)} jobs")
    return df_old
