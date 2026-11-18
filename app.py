"""
Job Market Analysis & Recommendation System
Run: pip install -r requirements.txt
     streamlit run app.py
Dataset: all_upwork_jobs_2024-02-07-2024-03-24.csv
Note: this is a historical snapshot, not a live feed.
"""
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Job Market Explorer", page_icon="📊", layout="wide")

CATEGORY_RULES = {
    "Data & Analytics": r"\b(data analyst|data analytics|data scientist|data science|business intelligence|power bi|tableau|sql|python|analytics engineer|data engineer|etl|machine learning)\b",
    "Software Development": r"\b(developer|software|web app|web development|full.?stack|frontend|front.?end|backend|back.?end|react|node\.?js|java|php|wordpress|mobile app|ios|android|flutter)\b",
    "Design & Creative": r"\b(design|designer|graphic|logo|ui/ux|ux|ui design|figma|illustrat|photoshop|video edit|animation)\b",
    "Writing & Translation": r"\b(writer|writing|copywriting|copywriter|proofread|editing|translation|translator|content writing|blog)\b",
    "Marketing & Sales": r"\b(marketing|seo|social media|google ads|facebook ads|media buyer|lead generation|sales|email marketing)\b",
    "Admin & Support": r"\b(virtual assistant|data entry|customer support|customer service|administrative|admin support|transcription)\b",
    "Finance & Accounting": r"\b(accountant|accounting|bookkeep|finance|financial analyst|tax preparation|quickbooks)\b",
    "Engineering & Architecture": r"\b(engineer|engineering|architect|cad|autocad|3d model|solidworks|mechanical|electrical)\b"
}
def classify(title):
    value = " " + str(title).lower() + " "
    for category, pattern in CATEGORY_RULES.items():
        if re.search(pattern, value):
            return category
    return "Other"

@st.cache_data
def load_data():
    df = pd.read_csv("all_upwork_jobs_2024-02-07-2024-03-24.csv")
    df["published_date"] = pd.to_datetime(df["published_date"], utc=True, errors="coerce")
    df["title"] = df["title"].fillna("").astype(str)
    df["country"] = df["country"].fillna("Unknown").astype(str)
    df["hourly_low"] = pd.to_numeric(df["hourly_low"], errors="coerce")
    df["hourly_high"] = pd.to_numeric(df["hourly_high"], errors="coerce")
    df["budget"] = pd.to_numeric(df["budget"], errors="coerce")
    df["hourly_midpoint"] = (df["hourly_low"] + df["hourly_high"]) / 2
    df["category"] = df["title"].map(classify)
    return df

try:
    data = load_data()
except Exception as exc:
    st.error("Could not load CSV. Keep the CSV beside app.py. " + str(exc))
    st.stop()

st.title("📊 Job Market Analysis & Recommendation System")
st.caption("Exploratory analysis and content-based recommendations from the supplied Upwork dataset.")
st.warning("Dataset limitation: historical snapshot. It does not connect to a live job API; refresh the CSV to update the analysis.")

valid_dates = data["published_date"].dropna()
if not valid_dates.empty:
    st.caption(f"Observed posting dates: {valid_dates.min().date()} to {valid_dates.max().date()} (UTC)")

# Sidebar filters
st.sidebar.header("Market filters")
countries = sorted(data["country"].unique())
selected_countries = st.sidebar.multiselect("Country", countries, default=[])
cats = sorted(data["category"].unique())
selected_cats = st.sidebar.multiselect("Job category", cats, default=[])
filtered = data.copy()
if selected_countries:
    filtered = filtered[filtered.country.isin(selected_countries)]
if selected_cats:
    filtered = filtered[filtered.category.isin(selected_cats)]

c1,c2,c3,c4 = st.columns(4)
c1.metric("Job postings", f"{len(filtered):,}")
c2.metric("Countries", f"{filtered.country.nunique():,}")
c3.metric("Hourly postings", f"{int(filtered.is_hourly.sum()):,}")
c4.metric("Categories", f"{filtered.category.nunique():,}")

tab1,tab2,tab3,tab4 = st.tabs(["Market overview","Salary & geography","Demand trends","Job recommendations"])
with tab1:
    left,right=st.columns(2)
    with left:
        by_cat=filtered.category.value_counts().rename_axis("category").reset_index(name="postings")
        st.plotly_chart(px.bar(by_cat.head(12),x="postings",y="category",orientation="h",title="Postings by inferred category"),use_container_width=True)
    with right:
        by_country=filtered.country.value_counts().head(12).rename_axis("country").reset_index(name="postings")
        st.plotly_chart(px.bar(by_country,x="postings",y="country",orientation="h",title="Top posting countries"),use_container_width=True)
    st.subheader("Most common job titles")
    st.dataframe(filtered.title.value_counts().head(20).rename_axis("job title").reset_index(name="postings"),use_container_width=True)

with tab2:
    hourly=filtered[filtered.is_hourly & filtered.hourly_midpoint.notna()].copy()
    hourly=hourly[hourly.hourly_midpoint.between(1,200)] # suppress extreme outliers for readable comparison
    st.caption("Hourly midpoint = (hourly_low + hourly_high) / 2. Extreme values above $200/hour are excluded from this chart only.")
    if not hourly.empty:
        country_rates=hourly.groupby("country").agg(postings=("title","size"),median_hourly_rate=("hourly_midpoint","median"),mean_hourly_rate=("hourly_midpoint","mean")).query("postings >= 20").reset_index()
        st.plotly_chart(px.bar(country_rates.sort_values("median_hourly_rate",ascending=False).head(20),x="median_hourly_rate",y="country",orientation="h",hover_data=["postings"],title="Median hourly midpoint by country (minimum 20 hourly postings)"),use_container_width=True)
        st.metric("Overall median hourly midpoint",f"${hourly.hourly_midpoint.median():.2f}/hour")
    else:
        st.info("No hourly rate records for the selected filters.")
    fixed=filtered[~filtered.is_hourly & filtered.budget.notna()]
    st.metric("Median fixed-price project budget",f"${fixed.budget.median():,.0f}" if not fixed.empty else "N/A")
    st.caption("Fixed-price budget is a project budget, not an annual salary. Currency is not explicitly stored in the supplied file; values are shown as reported.")

with tab3:
    dated=filtered.dropna(subset=["published_date"]).copy()
    if not dated.empty:
        dated["week"]=dated["published_date"].dt.to_period("W").astype(str)
        weekly=dated.groupby(["week","category"]).size().reset_index(name="postings")
        st.plotly_chart(px.line(weekly,x="week",y="postings",color="category",markers=True,title="Weekly posting volume by inferred category"),use_container_width=True)
        st.caption("A short historical window is not sufficient to establish a durable emerging trend or reliable long-term forecast.")
    else:
        st.info("No dated postings under the selected filters.")

with tab4:
    st.write("Enter skills, preferred role, and optional location/rate constraints. Matching uses TF-IDF similarity between your profile and job titles, with transparent rule-based category and filter preferences.")
    skills=st.text_input("Your skills / keywords",placeholder="e.g. SQL, Power BI, Python, dashboard")
    preferred=st.text_input("Preferred role",placeholder="e.g. Data Analyst")
    country=st.selectbox("Preferred country",["Any"]+sorted(data.country.unique()))
    min_rate=st.number_input("Minimum hourly rate (optional, USD as reported)",min_value=0.0,value=0.0,step=1.0)
    n=st.slider("Number of recommendations",5,30,10)
    if st.button("Find matching jobs",type="primary"):
        profile=(skills+" "+preferred).strip()
        if not profile:
            st.info("Enter at least one skill or a preferred role.")
        else:
            candidates=data.copy()
            if country!="Any":
                candidates=candidates[candidates.country==country]
            if min_rate>0:
                candidates=candidates[candidates.is_hourly & candidates.hourly_midpoint.ge(min_rate)]
            candidates=candidates[candidates.title.str.strip().ne("")]
            if candidates.empty:
                st.warning("No postings meet these filters. Try a broader country or rate.")
            else:
                corpus=candidates.title.tolist()+[profile]
                vectorizer=TfidfVectorizer(ngram_range=(1,2),stop_words="english")
                matrix=vectorizer.fit_transform(corpus)
                scores=cosine_similarity(matrix[-1],matrix[:-1]).ravel()
                candidates=candidates.copy()
                candidates["match_score"]=scores
                candidates=candidates.sort_values("match_score",ascending=False).head(n)
                show=candidates[["title","category","country","is_hourly","hourly_low","hourly_high","budget","match_score","link"]].copy()
                show["match_score"]=(show["match_score"]*100).round(1)
                st.dataframe(show.rename(columns={"match_score":"similarity_score_%"}),use_container_width=True,hide_index=True)
                st.caption("Similarity is a text-match score, not a probability of getting hired. Review each posting and its requirements before applying.")

st.divider()
st.caption("Method: title-keyword categories are heuristic; hourly midpoint is calculated only where both bounds are present. No personal data, experience level, or skill requirements are present in this dataset.")
