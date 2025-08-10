# AI Trends in Data Science Job Postings

Data scientists (and analysts) are facing a new paradigm where the boundaries of our role are shifting rapidly to adapt to the surge of AI technologies. Skills, responsibilities, and expectations are evolving faster than ever, often driven by trends rather than long-term stability. This project aims to provide clarity: to understand how our roles are being reshaped by AI, and to track which AI and data-related skills are surfacing in job postings over time.

In response to this, this repository contains the codebase for **AI Trends in Data Science Job Postings**, a Streamlit app that analyzes the evolution of AI-related terms in job postings for Data Scientist and Data Analyst roles.

The system fetches job postings via the **JSearch API** (RapidAPI), classifies them using an **LLM on OpenRouter** to separate true Data Analyst/Data Scientist positions from other similar but out-of-scope roles (e.g., *Financial Data Analyst*), and enriches them with structured **AI/Data term extraction**. This extraction uses LLM reasoning to detect skills, tools, and keywords related to data and AI in the context of the job’s responsibilities, unifies synonyms, and produces a clean, deduplicated set of terms for analysis.

It is designed with a **GitHub Actions scheduled cron job** so that on the **28th of each month**, job extractions are updated with new credits available from the JSearch API, which resets on that date. This ensures fresh data for subsequent analysis.

---

### 🔗 Live Demo

You can try the live analytics app here:  
**👉 [AI Trends in Data Science – Streamlit App](https://aitrendsds-cukohcfwja8uftb8nzsjka.streamlit.app/)**  
> ⚠️ The app may appear inactive due to sleeping from inactivity.  
> This is intentional, as I avoid abusing empty auto-commits to respect fair usage of the shared Streamlit Cloud space.  
> If you'd like to wake it up to test it, feel free to contact me at: [alexbmeist@gmail.com](mailto:alexbmeist@gmail.com)

---

## What This Project Is (and Is Not)

- ❌ This is **not a generic job scraper**.  
- ✅ It is a **full pipeline for AI term analytics**, from data ingestion to LLM-powered enrichment.  
- ✅ It applies rigorous deduplication and clustering to ensure analytical robustness.  
- ✅ It combines **GenAI reasoning** with deterministic data science techniques.

---

## Folder and File Structure

```
    root/
    │
    ├── streamlit_app.py               ← Streamlit front-end (2 analysis tabs)
    ├── requirements.txt               ← Project dependencies
    │
    ├── .github/workflows/
    │   └── monthly-update.yml          ← GitHub Actions for monthly dataset updates
    │
    ├── data/job_postings_cleaned/      ← Processed datasets (see README in folder)
    │
    ├── utils/
    │   ├── job_fetcher.py              ← API call logic to JSearch (US postings for DS/DA roles)
    │   ├── job_classifier.py           ← LLM-based classification of DS/DA vs. out-of-scope roles
    │   ├── ai_extraction.py            ← LLM-based AI/Data term extraction & normalization
    │   ├── job_pipeline.py             ← Orchestrates fetching, deduplication, classification, enrichment
    │   ├── generate_all_datasets.py    ← Incremental dataset updates + clustering
    │   └── openrouter_wrapper.py       ← OpenRouter + secrets manager wrapper
    │
    ├── learnings/                      ← Notes & technical lessons
    ├── screenshots/                    ← Visual assets for “Visual Preview” section
    └── README.md                       ← This file

```

## 🧠 Data Science Contributions and Highlights

This project is built **from a data science perspective**, with emphasis on robust preprocessing, LLM-assisted enrichment, and interpretable clustering.

Key robustness measures include:

- **Deduplication:** Even if a posting has a unique `job_id`, it is excluded if its combined title and description have ≥ 0.9 cosine similarity (TF-IDF vectorization), preventing near-duplicate jobs from skewing analysis.
- **Embedding and clustering:** New valid jobs are embedded using `all-MiniLM-L6-v2` from Hugging Face. Clusters are assigned using KMeans with optimal *k* selected via silhouette score.  
- **Dimensionality reduction:** UMAP is applied to cluster features for visualization, chosen for its ability to preserve both local and global structure while producing an interpretable 2D projection for the Streamlit explorer.
- **LLM-generated cluster descriptions:** After clustering, each group is described with an LLM to summarize skills, functions, and provide a concise cluster name.

This combination of **semantic enrichment**, **statistical clustering**, and **interpretability** ensures that the insights are both data-driven and meaningful to a human analyst.

---

### 📓 Processing Pipeline Overview

| Step | Script | Purpose |
|------|--------|---------|
| **1. Job fetching** | `job_fetcher.py` | Queries the JSearch API for multiple data science/analytics-related terms in the US. |
| **2. Deduplication** | `job_pipeline.py` | Removes near-duplicate postings using TF-IDF + cosine similarity (threshold ≥ 0.9). |
| **3. Classification (LLM)** | `job_classifier.py` | Uses LLM to flag postings as true Data Analyst/Scientist roles. |
| **4. AI/Data term extraction (LLM)** | `ai_extraction.py` | Extracts and normalizes AI and Data mentions with context-specific explanations. |
| **5. Dataset enrichment** | `job_pipeline.py` | Merges classification and term extraction results. |
| **6. Clustering** | `generate_all_datasets.py` | Embeds postings with Hugging Face (`all-MiniLM-L6-v2`), optimizes *k* via silhouette score, and describes clusters with an LLM. |

---

### 🤖 Role of LLMs (OpenRouter)

LLMs are central to three critical tasks:

1. **Role Classification**  
   - Decides if a job truly fits the Data Analyst/Data Scientist scope.
   - Applies relaxed criteria when the title clearly matches.

2. **AI/Data Term Extraction & Normalization**  
   - Extracts two separate categories:
     - *Data Mentions* (traditional DS/ML/analytics stack).
     - *AI Mentions* (modern AI stack, e.g., LLMs, RAG, prompt engineering).
   - Merges synonyms into a single canonical term.
   - Deduplicates explanations while preserving distinct use cases.

3. **Cluster Description**  
   - Summarizes each cluster’s skills, functions, and assigns a concise, unique name.

These steps are **not feasible with simple keyword rules** — the LLM handles semantic understanding, context-specific filtering, and grouping.

---

### 📊 Clustering Logic

- **Embeddings:** Generated with `all-MiniLM-L6-v2` from Hugging Face (chosen for speed, quality, and OpenRouter not providing embeddings).
- **K selection:** Optimal *k* is found per update batch via silhouette score.
- **Incremental updates:** Only new valid jobs are embedded and clustered, preserving existing cluster IDs.
- **Descriptions:** Each cluster’s name and description are LLM-generated from sample skills & contexts.

---

## Streamlit Deployment

The app (`streamlit_app.py`) has two main tabs:

1. **AI Mentions Analysis**  
   - Share of postings mentioning AI over time.  
     - If fewer than 3 months of data are available, it automatically switches to **weekly aggregation** to provide more temporal variation (less robust but more granular for short series).  
     - Once ≥ 3 months of data are available, it switches to **monthly aggregation** for stability.  
   - Top 10 AI terms (excluding “Artificial Intelligence”).
   - Evolution of each top term over time.
   - Most common responsibilities/descriptions per AI term.
   - All term trends and rankings are shown **in percentages rather than absolute counts**, since the number of postings analyzed is limited by the monthly free quota of the JSearch API.

2. **Job Cluster Explorer**  
   - UMAP visualization of job clusters.
   - Percentage distribution per cluster.
   - LLM-generated cluster descriptions.

### Key Features of the UI

- Interactive Plotly charts.
- Automatic parsing of AI/Data term details.
- Adaptive time binning for trend analysis.
- Last update date shown from dataset metadata.

---

### 🖼️ Visual Preview

*(Add screenshots here if desired, following the style in the Nococinomas Assistant README)*

### 🖼️ Visual Preview (In Case Demo is Unavailable)

In case the assistant becomes temporarily inaccessible — for example, if the current OpenAI API key runs out of credits — we include **screenshot previews** of the assistant’s functionality below.

Each screenshot is fully translated into English to illustrate the assistant’s input, reasoning, and output.

---

#### 📸 Screenshot 1 – Gluten-free high-fat meals for weight gain

![Screenshot 1](screenshots/screenshot_1.png)

**User input:**  
"I'm in a weight gain phase and have celiac disease. What are your highest-fat options?"

**Assistant output:**  
Here are your gluten-free options with the highest fat content, ideal for weight gain:

1. **Keto Chocolate Hazelnut Brownie**
   - Fat: 35.8 g  
   - Protein: 7.8 g  
   - Calories: 291  
   - Price: €4.95  
   - Weight: 180 g  

2. **Chicken with Creamy Pesto**
   - Fat: 13.5 g  
   - Protein: 13.7 g  
   - Calories: 183  
   - Price: €6.90  
   - Weight: 380 g  

3. **Eggplant and Meat Casserole**
   - Fat: 13.0 g  
   - Protein: 8.8 g  
   - Calories: 168  
   - Price: €6.35  
   - Weight: 380 g  

4. **Pepper Steak**
   - Fat: 11.2 g  
   - Protein: 19.4 g  
   - Calories: 186  
   - Price: €8.20  
   - Weight: 380 g  

5. **Salmon with Yogurt and Dill Sauce**
   - Fat: 16.4 g  
   - Protein: 10.3 g  
   - Calories: 206  
   - Price: €8.40  
   - Weight: 380 g  

Would you like help planning an order with these options?

---

#### 📸 Screenshot 2 – Keto-compatible side dishes and desserts

![Screenshot 2](screenshots/screenshot_2.png)

**User input:**  
"I'm on a Keto diet. Do you have any side dishes or desserts I can add?"

**Assistant output:**  
To support your Keto plan, I recommend the following side dishes and desserts:

1. **Keto or Ketogenic Pancakes**
   - Protein: 8.4 g  
   - Calories: 150  
   - Price: €4.15  
   - Vegetarian, Keto, diabetic-friendly  

2. **Keto Cheesecake (dessert)**
   - Protein: 6.5 g  
   - Calories: 162  
   - Price: €4.65  
   - Vegetarian, Keto, diabetic-friendly  

3. **Mini Keto Cheesecake (dessert)**
   - Protein: 11.2 g  
   - Calories: 103  
   - Price: €4.95  
   - Vegetarian, Keto, diabetic-friendly  

4. **Keto Chocolate Hazelnut Brownie (dessert)**
   - Protein: 7.8 g  
   - Calories: 291  
   - Price: €4.95  
   - Vegetarian, Keto, diabetic-friendly  

These options are compatible with a Keto diet and can complement your main dishes. Would you like help planning an order?

---

#### 📸 Screenshot 3 – 2-week low-fat lunch plan under €100

![Screenshot 3](screenshots/screenshot_3.png)

**User input:**  
"Create a 2-week lunch plan, low in fat, under €100, and with at least 2 meals suitable for freezing. I don’t mind repeating meals."

**Assistant output:**  
Here’s a 2-week (14-day) lunch plan, low in fat, with at least 2 freeze-friendly dishes, under €100:

1. **Beef with Vegetables**
   - Fat: 4.2 g  
   - Price: €7.35  
   - Freezer-friendly: Yes  

2. **Chicken with Vegetables**
   - Fat: 2.7 g  
   - Price: €6.15  
   - Freezer-friendly: Yes  

3. **Hake in Green Sauce with Vegetables**
   - Fat: 0.4 g  
   - Price: €7.85  
   - Freezer-friendly: Yes  

4. **Pinto Beans with Veggies and Ham**
   - Fat: 3.6 g  
   - Price: €6.55  
   - Freezer-friendly: Yes  

5. **Cauliflower with Garlic and Paprika**
   - Fat: 3.8 g  
   - Price: €5.85  
   - Freezer-friendly: Yes  

**Suggested plan (with repetitions):**

- Day 1: Beef with Vegetables (€7.35)  
- Day 2: Chicken with Vegetables (€6.15)  
- Day 3: Hake in Green Sauce with Vegetables (€7.85)  
- Day 4: Pinto Beans with Veggies and Ham (€6.55)  
- Day 5: Cauliflower with Garlic and Paprika (€5.85)  
- Day 6: Beef with Vegetables (€7.35)  
- Day 7: Chicken with Vegetables (€6.15)  
- Day 8: Hake in Green Sauce with Vegetables (€7.85)  
- Day 9: Pinto Beans with Veggies and Ham (€6.55)  
- Day 10: Cauliflower with Garlic and Paprika (€5.85)  
- Day 11: Beef with Vegetables (€7.35)  
- Day 12: Chicken with Vegetables (€6.15)  
- Day 13: Hake in Green Sauce with Vegetables (€7.85)  
- Day 14: Pinto Beans with Veggies and Ham (€6.55)  

**Estimated total cost:** €92.95  
All meals are low in fat and freezer-friendly.  
Would you like help placing your order?

---

#### 📸 Screenshot 4 – Vegetarian meals with the most protein

![Screenshot 4](screenshots/screenshot_4.png)

**User input:**  
"What are your highest-protein vegetarian meals?"

**Assistant output:**  
Here are the vegetarian meals with the most protein:

1. **Keto or Ketogenic Bread**
   - Protein: 23.3 g  
   - Price: €7.95  
   - Calories: 304  
   - Vegan: Yes  
   - Lactose-free: Yes  

2. **Heura Tacos (includes tortillas)**
   - Protein: 11.1 g  
   - Price: €6.95  
   - Calories: 135  
   - Vegan: Yes  
   - Lactose-free: Yes  

3. **Mini Keto Cheesecake**
   - Protein: 11.2 g  
   - Price: €4.95  
   - Calories: 103  
   - Vegetarian, not vegan  
   - Contains dairy  

4. **Keto Chocolate Hazelnut Brownie**
   - Protein: 7.8 g  
   - Price: €4.95  
   - Calories: 291  
   - Vegetarian, not vegan  
   - Lactose-free  

5. **Heura Nuggets with Potatoes**
   - Protein: 7.8 g  
   - Price: €4.95  
   - Calories: 145  
   - Vegan  
   - Lactose-free  

Would you like help planning an order with any of these?

---

## Environment Configuration (`.env` or Secrets)

This project **always** loads API keys and configuration from a secure secrets system:

- **Local**: `.env` file (loaded via `python-dotenv`).
- **Streamlit Cloud**: `.streamlit/secrets.toml`.
- **GitHub Actions**: `${{ secrets.KEY_NAME }}`.

Required keys:

    OPENROUTER_API_KEY=sk-or-v1-...
    OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
    LLM_MODEL=openai/gpt-4.1-mini
    JSEARCH_API_KEY=xxxxxxxxxxxxxxxxxxxx

---

## Scheduled Updates

A GitHub Actions workflow (`.github/workflows/monthly-update.yml`) runs:

- **Monthly** (on the 28th) — aligned with API credit reset for JSearch.

It executes:

    python utils/generate_all_datasets.py

to run the **entire processing pipeline**: fetching new postings, deduplicating, classifying, extracting AI/Data terms, clustering, and saving the **ready-to-use CSV files** consumed by the app.

---

## Data Outputs

Processed CSVs are stored in `data/job_postings_cleaned/`:

| File | Description |
|------|-------------|
| `data_scientist_us.csv` | Full dataset with classification and enrichment. |
| `data_scientist_us_ai_enriched.csv` | AI/Data mentions for valid jobs. |
| `data_scientist_us_classified.csv` | Binary LLM classification results. |
| `df_result_base.csv` | Incremental base dataset (all valid jobs). |
| `df_result_clusters.csv` | Clustered dataset with embeddings and descriptions. |

See the [`data/job_postings_cleaned/README.md`](data/job_postings_cleaned/README.md) for full details.

---

## Summary

This project is not just about scraping job postings — it’s a **data science system** that:

- Integrates **LLMs via OpenRouter** for semantic classification, AI/Data term extraction, and cluster summarization.
- Uses **Hugging Face embeddings** and **unsupervised learning** for skill clustering.
- Applies **stringent deduplication** to ensure analytical robustness.
- Handles **scheduled updates** aligned with API credit cycles.
- Provides an **interactive analytics dashboard** via Streamlit.

It is an example of how to blend **GenAI reasoning** with **deterministic, reproducible analytics**, making it possible to extract deep insights from unstructured job market data.

---

Made with 🧠 by a data scientist obsessed with extracting meaning from text data.

