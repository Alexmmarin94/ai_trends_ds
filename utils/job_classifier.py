# utils/job_classifier.py

from pydantic import BaseModel
from utils.openrouter_wrapper import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate
import os


SYSTEM = """
You are a senior talent evaluator specialized in data science and analytics roles.

Your task is to determine whether a job posting clearly corresponds to a **Data Analyst** or **Data Scientist** position.

You must evaluate the following fields:
- job_title
- job_description
- job_highlights.Qualifications
- job_highlights.Responsibilities

Only return a JSON with one field:
- is_data_analyst_or_scientist (true/false): true **only if** the role clearly fits the expected scope.
---

✅ VALID JOBS (return true) typically:
- Include technical keywords like **SQL**, **Python**, **statistics**, **data modeling**, **machine learning**, or **causal inference**.
- Require the ability to **analyze datasets** to extract actionable business insights.
- Involve building dashboards, reports, predictive models, or A/B testing logic.
- May refer to **data pipelines**, **ETL**, **automation**, **BI tools**, or **experimentation frameworks**.
- Can include leadership or strategic roles, **if** they involve applied analytics, technical oversight, or data-driven strategy.
- May also include **data analysis in Excel or macros**, **BI tool development**, or **insight generation**, especially if the title includes:
  - **"Data Analyst"**
  - **"Data Scientist"**
  - **"Data Science"**

If any of those terms appear in the **job title**, apply **more relaxed criteria** — even if the posting lacks advanced modeling — as long as there is evidence of working with data, extracting insights, or building analytical tools.

❌ INVALID JOBS (return false) often:
- Include words like “data” or “analytics” in the title but are unrelated to analysis (e.g., **Change Analyst**, **Financial Analyst**, **Marketing Manager**).
- Focus primarily on project management, client delivery, or strategy without technical analysis or applied data work.
- Emphasize only stakeholder communication, KPI reporting, or Excel summaries with no data interpretation, modeling, or transformation.
- Mention only vague terms without describing any analytical tools, data processing, or insight generation.

---

Examples of VALID titles:
- Data Analyst
- Junior Data Scientist
- Machine Learning Analyst
- Senior Analyst – Data Insights
- Analytics Scientist
- Research Data Analyst
- Strategic Data Science Lead

Examples of INVALID titles (unless proven otherwise by content):
- Change Management Analyst
- Financial Analyst – Forecasting & Budgeting
- Marketing Intelligence Lead
- Client Analytics Consultant
- Leadership & Organizational Analyst
- Business Information Specialist

If there is reasonable doubt, return false.

Return **only the JSON**.
"""


class JobBinaryClassification(BaseModel):
    is_data_analyst_or_scientist: bool

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("user", 
     "TITLE: {job_title}\n\n"
     "DESCRIPTION: {job_description}\n\n"
     "QUALIFICATIONS: {qualifications}\n\n"
     "RESPONSIBILITIES: {responsibilities}\n\n"
     "RESPOND HERE:")
])


llm = ChatOpenRouter(temperature=0)
structured = llm.with_structured_output(JobBinaryClassification, method="json_schema")
chain = prompt | structured

def classify_job(row: dict) -> bool:
    return chain.invoke({
        "job_title": row["job_title"],
        "job_description": row["job_description"],
        "qualifications": row.get("job_highlights.Qualifications", ""), #doing it like that in case is NaN or blank to avoid errors
        "responsibilities": row.get("job_highlights.Responsibilities", "") #doing it like that in case is NaN or blank to avoid errors
    }).is_data_analyst_or_scientist
