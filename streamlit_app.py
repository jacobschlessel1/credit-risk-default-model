import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import s3fs
import joblib

# Page config
st.set_page_config(
    page_title="Credit Risk Dashboard",
    layout="wide"
)

st.title("📊 Credit Risk Portfolio Dashboard")

st.markdown(
    """
    Forward-looking credit risk monitoring and scenario analysis for post-2016 loans.
    Model outputs are stored in AWS S3 and served through an interactive decision engine.
    """
)
#Load full dataset
@st.cache_data
def load_base_loan():
    df_full = pd.read_parquet(
        "data/processed/accepted_clean.parquet",
        engine="pyarrow"
    )

    df_full = df_full.loc[
        (df_full["issue_d"].dt.year >= 2016) &
        (df_full["current_flag"] == 0)
    ].copy()

    return df_full.sample(1, random_state=42)

base_loan = load_base_loan()

# Load data from S3
@st.cache_data
def load_dashboard_data():
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

loans, portfolio, policies = load_dashboard_data()


# Load model artifacts
@st.cache_resource
def load_model():
    calibrated_model = joblib.load("artifacts/xgb_post2016_calibrated.pkl")
    model_features = joblib.load("artifacts/model_features_post2016.pkl")
    return calibrated_model, model_features

calibrated_model, model_features = load_model()

# Feature engineering function

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
        "annual_inc_joint"
    ]

    for var in joint_numeric_pred:
        df[var] = df[var].fillna(0)
        df[f"{var}_active"] = df["joint_flag"] * df[var]

    flag_to_var = {
        "mths_since_last_record_flag": "mths_since_last_record",
        "mths_since_recent_bc_dlq_flag": "mths_since_recent_bc_dlq",
        "mths_since_last_major_derog_flag": "mths_since_last_major_derog",
        "mths_since_recent_revol_delinq_flag": "mths_since_recent_revol_delinq",
        "mths_since_last_delinq_flag": "mths_since_last_delinq",
    }

    for flag, var in flag_to_var.items():
        df[var] = df[var].replace(999, 0).fillna(0)
        df[f"{var}_active"] = df[flag] * df[var]

    recent_credit_vars_2016 = [
        "il_util", "mths_since_rcnt_il", "all_util",
        "open_acc_6m", "inq_last_12m", "total_cu_tl",
        "open_il_24m", "open_il_12m", "open_act_il",
        "max_bal_bc", "inq_fi", "total_bal_il",
        "open_rv_24m", "open_rv_12m"
    ]

    for col in recent_credit_vars_2016:
        df[col] = df[col].fillna(0)

    df = pd.get_dummies(
        df,
        columns=["term", "region", "home_ownership"],
        drop_first=True
    )

    return df

# Sidebar filters
st.sidebar.header("📅 Portfolio Filters")

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

# Risk Buckets
st.subheader("Risk Bucket Composition")
st.bar_chart(filtered["risk_bucket"].value_counts().sort_index())

# Policy table
st.subheader("Decision Engine – Policy Comparison")
st.dataframe(policies)

# What-IF Simulator
st.markdown("---")
st.header("🔧 What-If Probability of Default Simulator")


st.sidebar.header("What-If Inputs")

loan_amnt = st.sidebar.slider(
    "Loan Amount ($)",
    1000, 50000,
    int(base_loan["loan_amnt"].iloc[0]),
    step=500
)

fico = st.sidebar.slider(
    "FICO Score",
    550, 850,
    int(base_loan["fico_range_low"].iloc[0]),
    step=5
)

dti = st.sidebar.slider(
    "Debt-to-Income Ratio",
    0.0, 50.0,
    float(base_loan["dti"].iloc[0]),
    step=0.5
)

term = st.sidebar.selectbox(
    "Loan Term",
    [" 36 months", " 60 months"]
)

sim_loan = base_loan.copy()
sim_loan["loan_amnt"] = loan_amnt
sim_loan["fico_range_low"] = fico
sim_loan["fico_range_high"] = fico + 4
sim_loan["dti"] = dti
sim_loan["term"] = term

X_sim = build_model_features(sim_loan)
X_sim = X_sim.reindex(columns=model_features, fill_value=0)

pd_sim = calibrated_model.predict_proba(X_sim)[:, 1][0]

# Compute baseline PD from the unmodified base loan
X_base = build_model_features(base_loan)
X_base = X_base.reindex(columns=model_features, fill_value=0)

pd_orig = calibrated_model.predict_proba(X_base)[:, 1][0]

delta = pd_sim - pd_orig

st.metric(
    label="Simulated Probability of Default",
    value=f"{pd_sim:.2%}",
    delta=f"{delta:.2%}"
)

st.caption(
    "Simulated PDs are model-based sensitivity estimates and do not imply causal guarantees."
)
