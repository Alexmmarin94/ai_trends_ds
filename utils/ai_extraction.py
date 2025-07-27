# utils/ai_extraction.py

from pydantic import BaseModel
from typing import List
from utils.openrouter_wrapper import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate

SYSTEM = """
You are an expert in analyzing job descriptions in data and AI roles.

Your task is to extract and classify any **technical terms, methods, frameworks, or tools** mentioned in data scientist and data analyst job postings.

Split your output into two clearly differentiated categories:

### 1. Data Mentions
These include all relevant terms related to **data science, statistics, analytics, and machine learning**, such as:
- Regression models, classification, supervised/unsupervised learning
- Feature engineering, data pipelines, SQL, Pandas
- Scikit-learn, XGBoost, LightGBM
- Predictive models, time-series forecasting, A/B testing
- MLOps (unless it includes GenAI model ops) and Machine Learning in general
- BI tools and dashboards

Even if the job includes ML, unless the method falls under modern AI, treat it as "data".

- Every job posting MUST have **data-related mentions**.
- A job posting **MAY or MAY NOT** have AI-related mentions.
- If no AI concepts are found, return an empty list for AI. Don't force having any AI concept if they are not clearly present.

** Not AI**: 'Machine learning', 'Bayesian models', 'XGBoost', 'scikit-learn', 'MLOps', 'Time-series models', etc.

---

### 2. AI Mentions
These are terms related specifically to **modern Artificial Intelligence**, especially those relevant to:
- Generative AI (e.g., GPT, LLMs, diffusion models)
- NLP/NLU (if explicitly connected to AI/agents/transformers)
- Prompt engineering, RAG, embeddings, LLM fine-tuning
- Deep learning (e.g., PyTorch, TensorFlow *only* if applied to neural nets)
- Vision transformers, reinforcement learning for autonomous agents
- Multi-agent systems, autonomous agents, AI planning
- MCP

Also include clearly AI-native stacks and MLOps *if they involve managing LLMs or AI pipelines*.

** Valid AI Mentions**: 'GPT', 'RAG pipelines', 'Prompt engineering', 'LLMs', 'Generative AI', 'Transformers', 'NLP with LLMs'

---
## Term Normalization (CRITICAL)

If multiple terms refer to the same concept (e.g. 'RAG pipelines', 'Retrieval-Augmented Generation'), unify under a **single, consistent name** in `term`.

Only include **one mention per conceptual entity**. Redundant phrasing or synonyms should be merged.

Output must be structured as:

```json
{{
  "data_mentions": [
    {{ "term": "...", "explanation": "..." }},
    ...
  ],
  "ai_mentions": [
    {{ "term": "...", "explanation": "..." }},
    ...
  ]
}}

Each list must include:
- `term`: a **concise, normalized name** of the concept/tool (e.g. 'Retrieval-Augmented Generation', not 'RAG pipelines')
- `explanation`: a brief sentence describing **how that term is used or why it is relevant in the context of the data-related job described. Be concise but context-aware. Do NOT explain terms generically — explain how they relate to the job.**

"""
# ⚠️ IMPORTANT:
# We must use double curly braces {{ }} inside the JSON code block in the SYSTEM message above
# because LangChain's ChatPromptTemplate interprets single braces { } as variable placeholders. 
# If we don’t escape them, it throws a missing variable error like "missing variables {'data_mentions'}".



class Mention(BaseModel):
    term: str
    explanation: str

class AIExtractionResponse(BaseModel):
    data_mentions: List[Mention]
    ai_mentions: List[Mention]

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("user",
     "JOB TITLE: {job_title}\n\n"
     "JOB DESCRIPTION: {job_description}\n\n"
     "QUALIFICATIONS: {qualifications}\n\n"
     "RESPONSIBILITIES: {responsibilities}")
])

llm = ChatOpenRouter(temperature=0)
structured = llm.with_structured_output(AIExtractionResponse, method="json_schema")
chain = prompt | structured

def extract_ai_mentions(row: dict) -> dict:
    try:
        result = chain.invoke({
            "job_title": row["job_title"],
            "job_description": row["job_description"],
            "qualifications": row.get("job_highlights.Qualifications", ""),
            "responsibilities": row.get("job_highlights.Responsibilities", "")
        })
        return {
            "ai_mentions": [m.term for m in result.ai_mentions],
            "ai_details": "; ".join(f"{m.term}: {m.explanation}" for m in result.ai_mentions),
            "data_mentions": [m.term for m in result.data_mentions],
            "data_details": "; ".join(f"{m.term}: {m.explanation}" for m in result.data_mentions),
        }
    except Exception as e:
        print(f"❌ Extraction error: {e}")
        return {
            "ai_mentions": [],
            "ai_details": "",
            "data_mentions": [],
            "data_details": ""
        }
