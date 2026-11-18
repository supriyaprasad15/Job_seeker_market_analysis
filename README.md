# Job Market Analysis & Recommendation System

## Included
- `app.py`: Streamlit dashboard and TF-IDF content-based job recommender
- `all_upwork_jobs_2024-02-07-2024-03-24.csv`: supplied dataset (copy beside app.py)
- `requirements.txt`: Python dependencies

## Run locally
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Data and interpretation
The supplied file has 244,828 rows and columns for title, link, published date, hourly flag, hourly low/high, fixed-price budget, and country. The observed dates range from 2023-11-02 to 2024-03-24 (UTC), although the filename suggests February–March 2024. Treat this as a historical Upwork snapshot, not real-time market coverage.

The file does not contain annual salary, currency metadata, experience, skills, education, employer, or employment type. Hourly rates are reported hourly values; fixed-price budgets are project budgets, not salaries. Category labels are inferred from job-title keywords and should be reviewed before business decisions.

The recommender ranks postings using TF-IDF cosine similarity between the user's entered skills/role and the job title. It is a transparent prototype, not a trained hiring-success model.

## Scope covered
- Keyword-based job category and posting-volume analysis
- Hourly rate comparisons by country (with minimum sample thresholds and outlier filtering)
- Weekly posting trends
- Personalized content-based recommendations

## Not claimed / future work
- Live API ingestion and automatic monthly refresh
- Robust emerging-category detection and long-horizon forecasting
- Predictive model accuracy: no ground-truth future demand labels are provided
- Currency normalization and annual salary comparison
- Production API, authentication, Docker deployment, and user profile persistence

## Docker (optional)
```bash
docker compose up --build
```
Open `http://localhost:8501`.
