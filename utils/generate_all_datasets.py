# scripts/generate_all_datasets.py

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from utils.openrouter_wrapper import ChatOpenRouter
from utils.job_pipeline import run_job_pipeline

# --- Paths ---
DATA_DIR = Path("data/job_postings_cleaned")
BASE_CSV = DATA_DIR / "df_result_base.csv"
CLUSTERS_CSV = DATA_DIR / "df_result_clusters.csv"

# --- 1. Run job pipeline only if new jobs are fetched ---
print("🚀 Running base job pipeline...")
df_result = run_job_pipeline()  # Always runs, but we'll check if there are new jobs

# --- 2. Load previous base and cluster datasets if they exist ---
if BASE_CSV.exists():
    df_base_old = pd.read_csv(BASE_CSV)
else:
    df_base_old = pd.DataFrame(columns=df_result.columns)

if CLUSTERS_CSV.exists():
    df_clusters_old = pd.read_csv(CLUSTERS_CSV)
else:
    df_clusters_old = pd.DataFrame(columns=[
        "job_id", "is_valid_job", "skills_base", "context", "full_text", "cluster",
        "cluster_name", "cluster_description", "cluster_percentage"
    ])

# --- 3. Identify new jobs for base dataset ---
jobs_new_mask = ~df_result["job_id"].isin(df_base_old["job_id"])
df_jobs_new = df_result[jobs_new_mask].copy()

if not df_jobs_new.empty:
    print(f"💡 {len(df_jobs_new)} new jobs detected. Updating df_result_base.csv...")
    df_base_updated = pd.concat([df_base_old, df_jobs_new], ignore_index=True)
    df_base_updated.to_csv(BASE_CSV, index=False)
else:
    print("ℹ️ No new jobs detected. df_result_base.csv remains unchanged.")
    df_base_updated = df_base_old

# --- 4. Clustering: Only for new jobs not already clustered ---
# Filter valid jobs for clustering that are not yet in df_clusters_old or have cluster as NaN
valid_jobs = df_base_updated[df_base_updated["is_valid_job"] == True].copy()
jobs_clustered = set(df_clusters_old.loc[df_clusters_old['cluster'].notnull(), 'job_id'])
jobs_to_cluster = valid_jobs[~valid_jobs["job_id"].isin(jobs_clustered)].copy()

if not jobs_to_cluster.empty:
    print(f"🔢 {len(jobs_to_cluster)} new valid jobs to cluster...")

    # --- 4.1. Prepare cluster features for new jobs only ---
    jobs_to_cluster["skills_base"] = (
        jobs_to_cluster["data_mentions"].fillna("") + " " +
        jobs_to_cluster["ai_mentions"].fillna("")
    )
    jobs_to_cluster["context"] = (
        jobs_to_cluster["job_title"].fillna("") + " " +
        jobs_to_cluster["job_description"].fillna("")
    )
    jobs_to_cluster["full_text"] = jobs_to_cluster["skills_base"].str.strip() + " | " + jobs_to_cluster["context"].str.strip()

    # --- 4.2. Embeddings for new jobs only ---
    print("🔢 Generating embeddings for new jobs...")
    model_emb = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model_emb.encode(jobs_to_cluster["full_text"].tolist(), show_progress_bar=True)

    # --- 4.3. Cluster assignment for new jobs ---
    print("📊 Assigning clusters to new jobs...")
    # If you want to cluster all jobs together, you must refit on the whole dataset
    # Here, for pure incrementality, assign new clusters within just the new jobs
    # Optionally, you can add logic to re-cluster everything if enough new jobs accumulate

    # Select optimal k
    score_by_k = {}
    n_new = len(embeddings)
    max_k = min(10, max(2, n_new // 2))
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)
        score_by_k[k] = silhouette_score(embeddings, labels)
    opt_k = max(score_by_k, key=score_by_k.get)
    print(f"✅ Optimal K for new jobs: {opt_k}")

    # Final clustering
    kmeans = KMeans(n_clusters=opt_k, random_state=42, n_init=10)
    jobs_to_cluster["cluster"] = kmeans.fit_predict(embeddings)

    # --- 4.4. Cluster naming using OpenRouter ---
    print("📝 Describing new clusters...")
    llm = ChatOpenRouter(temperature=0)

    def describe_cluster(cluster_id, jobs_df):
        skills = jobs_df["skills_base"].dropna().tolist()
        contexts = jobs_df["context"].dropna().tolist()
        sample_skills = " || ".join(skills[:10])
        sample_contexts = " || ".join(contexts[:5])
        prompt = f"""
You are a job market analyst. Analyze the following job offers (skills and context) from one cluster.
Summarize the core profile of this group: key skills, job functions, and a short, unique cluster name.

Skills:
{sample_skills}

Context:
{sample_contexts}

Respond with:
- Cluster name (concise and unique)
- Key skills (comma-separated)
- Typical job functions
"""
        try:
            resp = llm.invoke(prompt)
            return resp.content.strip()
        except Exception as e:
            print(f"[ERROR] Cluster {cluster_id} description failed: {e}")
            return f"Cluster {cluster_id} - Description unavailable"

    # Map new cluster ids to descriptions/names (only among new jobs)
    cluster_descriptions = {}
    for cid in sorted(jobs_to_cluster["cluster"].unique()):
        sub_df = jobs_to_cluster[jobs_to_cluster["cluster"] == cid]
        cluster_descriptions[cid] = describe_cluster(cid, sub_df)

    # Assign cluster description fields
    jobs_to_cluster["cluster_description"] = jobs_to_cluster["cluster"].map(cluster_descriptions)
    # Optionally generate names programmatically or parse from description
    jobs_to_cluster["cluster_name"] = jobs_to_cluster["cluster_description"].apply(lambda x: x.split("\n")[0] if isinstance(x, str) else "Unknown Cluster")

    # Compute percentages within new jobs only
    cluster_percent = jobs_to_cluster["cluster"].value_counts(normalize=True).mul(100).round(2)
    jobs_to_cluster["cluster_percentage"] = jobs_to_cluster["cluster"].map(cluster_percent)

    # --- 4.5. Concatenate new clusterized jobs to cluster dataset ---
    df_clusters_updated = pd.concat([df_clusters_old, jobs_to_cluster], ignore_index=True)
    df_clusters_updated.to_csv(CLUSTERS_CSV, index=False)
    print(f"💾 Appended {len(jobs_to_cluster)} new clusterized jobs to df_result_clusters.csv.")

else:
    print("ℹ️ No new valid jobs to cluster. df_result_clusters.csv remains unchanged.")

print("🎯 All datasets are up to date and consistent.")
