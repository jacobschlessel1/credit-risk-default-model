import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import s3fs
import joblib
import shap
from pathlib import Path
import boto3
from io import BytesIO

# Config
st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")

S3_BUCKET = "jacobschlessel-credit-risk"     
S3_PREFIX = "dashboard"                      

# S3 client
s3 = boto3.client("s3")

def read_parquet_s3(key: str) -> pd.DataFrame:
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    return pd.read_parquet(BytesIO(obj["Body"].read()))

def read_joblib_s3(key: str):
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    return joblib.load(BytesIO(obj["Body"].read()))

# Load dashboard tables
@st.cache_data
def load_dashboard_tables():
    loans = read_parquet_s3(f"{S3_PREFIX}/dashboard_loans.parquet")
    portfolio = read_parquet_s3(f"{S3_PREFIX}/portfolio_summary.parquet")
    policies = read_parquet_s3(f"{S3_PREFIX}/policy_summary.parquet")
    return loans, portfolio, policies

# Load model artifacts
@st.cache_resource
def load_model_artifacts():
    calibrated_model = read_joblib_s3(f"{S3_PREFIX}/xgb_post2016_calibrated.pkl")
    model_features = read_joblib_s3(f"{S3_PREFIX}/model_features_post2016.pkl")
    xgb_model = read_joblib_s3(f"{S3_PREFIX}/xgb_post2016.pkl")
    return calibrated_model, model_features, xgb_model

# Feature engineering
def build_model_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    joint_numeric_pred = [
        "sec_app_mths_since_last_major_derog",
        "sec_app_revol_util",
        "revol_bal_joint",
        "sec_app_open_acc",
        "sec_app_num_rev_accts",
        "sec_app_inq_last_6mths",
        "sec_app_mort_acc",
        "sec_app_open_act_il",
        "sec_app_fico_range_low",
        "sec_app_fico_range_high",
        "sec_app_chargeoff_within_12_mths",
        "sec_app_collections_12_mths_ex_med",
        "dti_joint",
        "annual_inc_joint",
    ]

    for var in joint_numeric_pred:
        df[var] = df.get(var, 0).fillna(0)
        df[f"{var}_active"] = df.get("joint_flag", 0) * df[var]

    flag_to_var = {
        "mths_since_last_record_flag": "mths_since_last_record",
        "mths_since_recent_bc_dlq_flag": "mths_since_recent_bc_dlq",
        "mths_since_last_major_derog_flag": "mths_since_last_major_derog",
        "mths_since_recent_revol_delinq_flag": "mths_since_recent_revol_delinq",
        "mths_since_last_delinq_flag": "mths_since_last_delinq",
    }

    for flag, var in flag_to_var.items():
        df[var] = df.get(var, 0).replace(999, 0).fillna(0)
        df[f"{var}_active"] = df.get(flag, 0) * df[var]

    recent_credit_vars_2016 = [
        "il_util", "mths_since_rcnt_il", "all_util",
        "open_acc_6m", "inq_last_12m", "total_cu_tl",
        "open_il_24m", "open_il_12m", "open_act_il",
        "max_bal_bc", "inq_fi", "total_bal_il",
        "open_rv_24m", "open_rv_12m",
    ]

    for col in recent_credit_vars_2016:
        df[col] = df.get(col, 0).fillna(0)

    df = pd.get_dummies(
        df,
        columns=["term", "region", "home_ownership"],
        drop_first=True
    )

    return df

# Load data from S3
@st.cache_data
def load_full_data_sample(n=5000):
    df = read_parquet_s3(f"{S3_PREFIX}/accepted_clean.parquet")
    df = df[df["issue_d"].dt.year >= 2016].copy()
    return df.sample(n=min(n, len(df)), random_state=42)

# SHAP computation
@st.cache_data
def compute_top_shap_features(_xgb_model, model_features, df_sample):
    X_full = build_model_features(df_sample)
    X = X_full.reindex(columns=model_features, fill_value=0)

    explainer = shap.TreeExplainer(_xgb_model)
    shap_vals = explainer.shap_values(X)

    return (
        pd.Series(np.abs(shap_vals).mean(axis=0), index=X.columns)
        .sort_values(ascending=False)
        .head(5)
    )

# Load
loans, portfolio, policies = load_dashboard_tables()
calibrated_model, model_features, xgb_model = load_model_artifacts()

# Tabs
tab_eda, tab_sim, tab_decision = st.tabs([
    "1) Project Summary",
    "2) Interactive Simulator",
    "3) Decision Engine Dashboard"
])

# Tab 1: Summary and EDA
with tab_eda:
    st.header("Project Summary")

    st.markdown(
        """
This project builds and deploys a **calibrated Probability of Default (PD) 
model**
to evaluate credit risk for consumer loans from 2016-2018. Data is sourced 
from LendingClub, a peer-to-peer lending platform. Important note: for this 
project, **default** is defined as a loan that is either charged off or 
defaulted, so predicted probabilities reflect this definition and may be higher than one would expect for loans that strictly defaulted only.
        """
    )

    # SHAP
    st.subheader("Most Important Features in Predicting Probability of Default")
    df_sample = load_full_data_sample()
    top5 = compute_top_shap_features(xgb_model, model_features, df_sample)

    fig, ax = plt.subplots()
    ax.barh(top5.index[::-1], top5.values[::-1])
    ax.set_xlabel("Relative Importance")
    ax.set_title("Top 5 Most Important Features")
    st.pyplot(fig)

    # Loan amount distribution
    st.subheader("Loan Amount Distribution")
    bins = np.arange(0, loans["loan_amnt"].max() + 5000, 5000)

    fig, ax = plt.subplots()
    ax.hist(loans["loan_amnt"], bins=bins)
    ax.set_xlabel("Loan Amount ($)")
    ax.set_ylabel("Number of Loans")
    st.pyplot(fig)

    # Number of loans by year
    loans_by_year = (
        df_sample
        .assign(issue_year=df_sample["issue_d"].dt.year)
        .groupby("issue_year")
        .size()
    )

    fig, ax = plt.subplots()
    ax.bar(loans_by_year.index, loans_by_year.values)
    ax.set_xticks(loans_by_year.index)
    ax.set_xticklabels(loans_by_year.index, rotation=45)
    ax.set_xlabel("Origination Year")
    ax.set_ylabel("Number of Loans")
    ax.set_title("Number of Loans by Origination Year")
    st.pyplot(fig)

    # Pie chart: region
    st.subheader("Loan Distribution by Region")

    region_counts = df_sample["region"].value_counts()

    fig, ax = plt.subplots()
    ax.pie(region_counts, labels=region_counts.index, autopct="%1.1f%%")
    ax.set_title("Loans by Region")
    st.pyplot(fig)


# Simulator
with tab_sim:
    st.header("🔧 Interactive Simulator")
    st.markdown(
        "Adjust the important borrower characteristics to see how the " \
        "chance of default " \
        "changes. Note: all other features are held constant at a " \
        "representative borrower's values."
    )

    base_loan = load_full_data_sample(n=1)

    loan_amnt = st.slider("Loan Amount ($)", 1000, 50000, int(base_loan["loan_amnt"].iloc[0]), 500)
    fico = st.slider("FICO Score", 550, 850, int(base_loan["fico_range_low"].iloc[0]), 5)
    dti = st.slider("Debt-to-Income Ratio", 0.0, 50.0, float(base_loan["dti"].iloc[0]), 0.5)

    sim_loan = base_loan.copy()
    sim_loan["loan_amnt"] = loan_amnt
    sim_loan["fico_range_low"] = fico
    sim_loan["fico_range_high"] = fico + 4
    sim_loan["dti"] = dti

    X_base = build_model_features(base_loan).reindex(columns=model_features, fill_value=0)
    X_sim = build_model_features(sim_loan).reindex(columns=model_features, fill_value=0)

    pd_base = calibrated_model.predict_proba(X_base)[:, 1][0]
    pd_sim = calibrated_model.predict_proba(X_sim)[:, 1][0]

    st.metric("Reference PD", f"{pd_base:.2%}")
    st.metric("Simulated PD", f"{pd_sim:.2%}", delta=f"{pd_sim - pd_base:.2%}")


# Decision engine dashboard
with tab_decision:
    st.header("Decision Engine Dashboard")

    st.markdown(
        "In creating policies for loan approvals, it's important to "
        "understand the overall risk profile of the portfolio. This "
        "dashboard shows important implications of different risk "
        "tolerances in deciding to accept or reject loans based on their "
        "odds of defaulting. Here, **risk buckets** are used to show the "
        "proportion of loans that fall into specific ranges of chances of "
        "defaulting."
    )


    c1, c2, c3 = st.columns(3)
    c1.metric("Loans", f"{len(loans):,}")
    c2.metric("Avg PD", f"{loans['calibrated_pd'].mean():.2%}")
    c3.metric("Exposure", f"${loans['loan_amnt'].sum():,.0f}")

    st.subheader("Distribution of Probability of Default (PD)")
    pd_hist = (
        pd.cut(loans["calibrated_pd"], bins=10)
        .value_counts()
        .sort_index()
    )
    pd_hist.index = pd_hist.index.astype(str)
    st.bar_chart(pd_hist)

    
    st.subheader("Risk Bucket Composition")
    st.bar_chart(loans["risk_bucket"].value_counts().sort_index())

    st.subheader("Portfolio Summary")
    st.dataframe(portfolio)

    st.subheader("Policy Comparison")
    st.dataframe(policies)

