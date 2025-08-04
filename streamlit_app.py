# streamlit_app.py

import streamlit as st

# ============ TAB 1: AI Mentions Analysis ============
def tab1_content():
    import pandas as pd
    import numpy as np
    from ast import literal_eval
    from collections import Counter, defaultdict
    import plotly.express as px
    from datetime import timedelta

    st.title("AI Trends in Data Science Job Postings")

    df_result = pd.read_csv("data/job_postings_cleaned/df_result_base.csv")
    df_result = df_result[df_result["is_valid_job"] == True].copy()

    # Parse ai_mentions as list
    df_result["ai_mentions"] = df_result["ai_mentions"].apply(
        lambda x: literal_eval(x) if isinstance(x, str) else []
    )

    # Parse ai_details
    def parse_ai_details(x):
        if pd.isnull(x):
            return []
        return [s.strip() for s in x.split(";") if s.strip()]

    df_result["ai_details"] = df_result["ai_details"].apply(parse_ai_details)

    # Date parsing
    def parse_datetime_utc(val):
        dt = pd.to_datetime(val, errors='coerce')
        if pd.isnull(dt):
            return pd.NaT
        if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
            return dt.tz_localize('UTC')
        else:
            return dt.tz_convert('UTC')

    df_result["job_posted_at_datetime_utc"] = df_result["job_posted_at_datetime_utc"].apply(parse_datetime_utc)
    df_result["extraction_date"] = df_result["extraction_date"].apply(parse_datetime_utc)

    mask_null_posted = df_result["job_posted_at_datetime_utc"].isnull()
    df_result.loc[mask_null_posted, "job_posted_at_datetime_utc"] = (
        df_result.loc[mask_null_posted, "extraction_date"] - pd.Timedelta(days=15)
    )

    df_result["job_posted_at_datetime_utc"] = df_result["job_posted_at_datetime_utc"].dt.tz_localize(None)
    last_extraction_date = df_result["extraction_date"].max().date()

    min_date = df_result["job_posted_at_datetime_utc"].min()
    max_date = df_result["job_posted_at_datetime_utc"].max()
    date_range = max_date - min_date

    if date_range < pd.Timedelta(days=90):
        df_result["time_bin"] = df_result["job_posted_at_datetime_utc"].dt.to_period("W").dt.to_timestamp()
        freq_label = "Weekly"
    else:
        df_result["time_bin"] = df_result["job_posted_at_datetime_utc"].dt.to_period("M").dt.to_timestamp()
        freq_label = "Monthly"

    # 1. Percentage of postings mentioning AI per time_bin
    df_result["has_ai_mention"] = df_result["ai_mentions"].apply(lambda x: len(x) > 0)
    ai_mention_evolution = (
        df_result.groupby("time_bin")["has_ai_mention"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "perc_ai_mention", "count": "n_postings"})
        .reset_index()
    )
    ai_mention_evolution["perc_ai_mention"] *= 100

    fig1 = px.line(
        ai_mention_evolution,
        x="time_bin",
        y="perc_ai_mention",
        markers=True,
        title=f"Share of Data Science Job Postings Mentioning AI ({freq_label})",
        labels={"perc_ai_mention": "% with AI Mention", "time_bin": "Job Posting Date"},
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    fig1.update_layout(
        xaxis=dict(tickformat="%Y-%m-%d", tickangle=45),
        yaxis_title="% with AI Mention",
        margin=dict(l=80, r=20, t=50, b=100),
        font=dict(size=14)
    )
    fig1.update_traces(line=dict(width=3))
    st.plotly_chart(fig1, use_container_width=True)

    # 2. Top 10 AI terms: percentage of postings
    all_ai_terms = [item for sublist in df_result["ai_mentions"] for item in sublist]
    term_counts = Counter(all_ai_terms)
    term_counts = {k: v for k, v in term_counts.items() if k != "Artificial Intelligence"}
    top10 = Counter(term_counts).most_common(10)
    top10_terms = [t[0] for t in top10]
    top10_counts = [t[1] for t in top10]
    total_postings = len(df_result)
    top10_percents = [round((count / total_postings) * 100, 2) for count in top10_counts]
    df_top10 = pd.DataFrame({"AI Term": top10_terms, "Percentage of Postings": top10_percents})

    fig2 = px.bar(
        df_top10,
        x="Percentage of Postings",
        y="AI Term",
        orientation="h",
        title="Top 10 Most Mentioned AI Terms in Data Science Job Postings (Excl. 'Artificial Intelligence')",
        labels={"Percentage of Postings": "% of Postings", "AI Term": "AI Term"},
        color="AI Term",
        color_discrete_sequence=px.colors.qualitative.Plotly,
        text="Percentage of Postings"
    )
    fig2.update_layout(
        yaxis=dict(categoryorder="total ascending", title=None),
        margin=dict(l=120, r=20, t=50, b=100),
        font=dict(size=14),
        showlegend=False
    )
    st.plotly_chart(fig2, use_container_width=True)

    # 3. Evolution over time for top 10 AI terms
    evo_df = (
        df_result
        .loc[:, ["time_bin", "ai_mentions"]]
        .explode("ai_mentions")
        .assign(ai_term=lambda d: d["ai_mentions"])
        .dropna(subset=["ai_term"])
    )
    evo_df = evo_df[evo_df["ai_term"].isin(top10_terms)]
    grouped = (
        evo_df.groupby(["time_bin", "ai_term"])
        .size()
        .reset_index(name="n_mentions")
    )
    total_per_period = df_result.groupby("time_bin").size().rename("total").reset_index()
    grouped = grouped.merge(total_per_period, on="time_bin")
    grouped["perc_postings"] = grouped["n_mentions"] / grouped["total"] * 100

    fig3 = px.line(
        grouped,
        x="time_bin",
        y="perc_postings",
        color="ai_term",
        markers=True,
        title=f"Share of Job Postings Mentioning Each Top AI Term ({freq_label})",
        labels={"perc_postings": "% of Postings", "time_bin": "Job Posting Date", "ai_term": "AI Term"},
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    fig3.update_layout(
        xaxis=dict(tickformat="%Y-%m-%d", tickangle=45),
        yaxis_title="% of Postings",
        legend_title_text="AI Term",
        margin=dict(l=100, r=20, t=50, b=120),
        font=dict(size=14),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.01,
            font=dict(size=12)
        )
    )
    fig3.update_traces(line=dict(width=3))
    st.plotly_chart(fig3, use_container_width=True)

    # 4. Most common ai_details for each top10 term
    st.subheader("Most Common Functions/Descriptions for Top AI Terms")
    st.write("Below are the most frequently mentioned responsibilities, use cases, or required skills for each of the top 10 AI terms in Data & Analytics job postings:")

    term_to_details = defaultdict(list)
    for details_list in df_result["ai_details"]:
        for detail in details_list:
            if ":" in detail:
                term, explanation = detail.split(":", 1)
                term = term.strip()
                explanation = explanation.strip()
                term_to_details[term].append(explanation)
            else:
                term_to_details[term].append(detail.strip())

    details_top_terms = {}
    for term in top10_terms:
        details = term_to_details.get(term, [])
        if not details:
            details_top_terms[term] = []
        else:
            detail_counts = Counter(details)
            most_common = detail_counts.most_common(20)
            if len(most_common) < 20 and len(set(details)) == len(details):
                most_common = [(d, 1) for d in details[-20:]]
            details_top_terms[term] = [d for d, count in most_common]

    for term, details in details_top_terms.items():
        st.markdown(f"**{term}**")
        if not details:
            st.write("- No details available.")
        else:
            for d in details:
                st.write(f"- {d}")

    st.info(f"Data last updated on extraction date: {last_extraction_date}")

# ============ TAB 2: Job Cluster Explorer ============
def tab2_content():
    import pandas as pd
    import umap
    import plotly.express as px
    import re

    st.title("Data Science Job Clusters")
    df_clusters = pd.read_csv("data/job_postings_cleaned/df_result_clusters.csv")

    # Extract clean cluster name
    def extract_cluster_name(desc):
        if pd.isna(desc):
            return None
        match = re.search(r":\s*(.*?)\n", desc, re.DOTALL)
        if match:
            name = match.group(1).strip()
            name = name.replace("**", "").strip()
            return name
        return None

    df_clusters["cluster_name"] = df_clusters["cluster_description"].apply(extract_cluster_name)
    df_clusters["cluster_name"] = df_clusters["cluster_name"].astype(str)
    df_clusters["cluster"] = df_clusters["cluster"].astype(str)

    # Generate embeddings for visualization
    features = pd.get_dummies(df_clusters["cluster_name"], prefix="cluster")
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(features)
    df_clusters["embedding_x"] = embedding[:, 0]
    df_clusters["embedding_y"] = embedding[:, 1]

    # Scatter plot
    fig = px.scatter(
        df_clusters,
        x="embedding_x",
        y="embedding_y",
        color="cluster_name",
        hover_data=["job_title", "cluster_name", "cluster_percentage"],
        title="Job Posting Clusters (UMAP projection)",
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    fig.update_traces(marker=dict(size=8, opacity=0.7, line=dict(width=0)))
    st.plotly_chart(fig, use_container_width=True)

    # Percentage bar chart
    cluster_counts = (
        df_clusters.groupby(["cluster", "cluster_name"])
        .size()
        .reset_index(name="count")
    )
    total_postings = cluster_counts["count"].sum()
    cluster_counts["percentage"] = (cluster_counts["count"] / total_postings) * 100
    cluster_counts = cluster_counts.sort_values("percentage", ascending=False)
    unique_cluster_names = cluster_counts["cluster_name"].unique()
    color_map = {
        name: color
        for name, color in zip(
            unique_cluster_names,
            px.colors.qualitative.Plotly * ((len(unique_cluster_names) // len(px.colors.qualitative.Plotly)) + 1)
        )
    }
    cluster_counts["color"] = cluster_counts["cluster_name"].map(color_map)
    fig_bar = px.bar(
        cluster_counts,
        x="cluster",
        y="percentage",
        title="Percentage of Job Postings per Cluster",
        labels={"cluster": "Cluster", "percentage": "% of Postings"},
        color="cluster_name",
        color_discrete_map=color_map,
        text=cluster_counts["percentage"].round(1).astype(str) + "%",
        hover_data=["cluster_name"]
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(
        showlegend=False,
        xaxis_tickangle=0
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Cluster descriptions with bold for headers
    def extract_cluster_description(desc):
        if pd.isna(desc):
            return None
        desc_clean = re.sub(r"^.*?\n", "", desc, count=1, flags=re.DOTALL)
        desc_clean = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", desc_clean)
        desc_clean = re.sub(r"^\s*-\s*", "", desc_clean, flags=re.MULTILINE)
        desc_clean = desc_clean.strip()
        return desc_clean

    df_clusters["cluster_full_description"] = df_clusters["cluster_description"].apply(extract_cluster_description)
    cluster_info = (
        df_clusters.groupby(["cluster", "cluster_name"])["cluster_full_description"]
        .first()
        .reset_index()
        .sort_values("cluster")
    )
    st.subheader("Cluster Descriptions")
    for _, row in cluster_info.iterrows():
        st.markdown(f"""
        <div style='margin-bottom: 40px; padding: 10px; border: 1px solid #ddd; border-radius: 5px;'>
            <h3 style='margin-bottom: 8px; color: #1f77b4;'>Cluster {row["cluster"]}: {row["cluster_name"]}</h3>
            <p style='white-space: pre-line; margin-top: 0;'>{row["cluster_full_description"]}</p>
        </div>
        """, unsafe_allow_html=True)

# ======================= TABS ========================
st.set_page_config(page_title="AI Job Analytics", layout="wide")
tabs = st.tabs(["AI Mentions Analysis", "Job Cluster Explorer"])

with tabs[0]:
    tab1_content()
with tabs[1]:
    tab2_content()
