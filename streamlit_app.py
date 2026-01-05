import streamlit as st
import pandas as pd
import s3fs
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Credit Risk Dashboard",
    layout="wide"
)

st.title("📊 Credit Risk Portfolio Dashboard")
st.markdown(
    """
    Forward-looking credit risk monitoring for post-2016 current loans.
    Data is sourced from AWS S3 and powered by a calibrated PD decision engine.
    *PD = Probability of Default*
    """
)

@st.cache_data
def load_data():
    fs = s3fs.S3FileSystem()

    loans = pd.read_parquet(
        "s3://jacobschlessel-credit-risk/dashboard/dashboard_loans.parquet",
        filesystem=fs
    )

    portfolio = pd.read_parquet(
        "s3://jacobschlessel-credit-risk/dashboard/portfolio_summary.parquet",
        filesystem=fs
    )

    policies = pd.read_parquet(
        "s3://jacobschlessel-credit-risk/dashboard/policy_summary.parquet",
        filesystem=fs
    )

    return loans, portfolio, policies

loans, portfolio, policies = load_data()

# Sidebar filter
st.sidebar.header("Filters")
year_min, year_max = int(loans["issue_year"].min()), int(loans["issue_year"].max())
year_range = st.sidebar.slider(
    "Origination Year",
    year_min,
    year_max,
    (year_min, year_max)
)

filtered = loans[
    loans["issue_year"].between(year_range[0], year_range[1])
]

# KPIs
c1, c2, c3 = st.columns(3)
c1.metric("Loans", f"{len(filtered):,}")
c2.metric("Average PD", f"{filtered['calibrated_pd'].mean():.2%}")
c3.metric("Exposure", f"${filtered['loan_amnt'].sum():,.0f}")

# PD distribution
st.subheader("Distribution of Calibrated PDs")
fig, ax = plt.subplots()
ax.hist(filtered["calibrated_pd"], bins=50)
ax.set_xlabel("Calibrated PD")
ax.set_ylabel("Loan Count")
st.pyplot(fig)

# Risk buckets
st.subheader("Risk Bucket Composition")
st.bar_chart(filtered["risk_bucket"].value_counts().sort_index())

# Policy results
st.subheader("Decision Engine Policy Comparison")
st.dataframe(policies)
