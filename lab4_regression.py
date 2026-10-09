from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="EC2 Pricing Analysis and Regression", layout="wide")
st.title("EC2 Pricing Analysis and Regression")
st.caption("Activity: Analyzing Amazon EC2 Instance Costs · Prices come from the supplied CSV.")

PRICE_COLUMNS = [
    "On Demand", "Linux Reserved cost", "Linux Spot Minimum cost",
    "Windows On Demand cost", "Windows Reserved cost",
]

@st.cache_data
def load_data():
    raw = pd.read_csv(Path(__file__).with_name("ec2dataset.csv"))
    data = raw.copy()
    for column in PRICE_COLUMNS:
        cleaned = (data[column].astype("string")
                   .str.replace("$", "", regex=False)
                   .str.replace(",", "", regex=False)
                   .str.replace("hourly", "", regex=False).str.strip())
        data[column] = pd.to_numeric(cleaned, errors="coerce")
    data["Instance Memory"] = pd.to_numeric(
        raw["Instance Memory"].str.extract(r"([\d.]+)", expand=False), errors="coerce")
    data["vCPUs"] = pd.to_numeric(
        raw["vCPUs"].str.extract(r"(\d+)", expand=False), errors="coerce")
    data["Family"] = data["API Name"].str.split(".").str[0]
    return raw, data

raw, data = load_data()
overview, analysis, families, regression = st.tabs([
    "Dataset", "Cost Analysis", "Instance Families", "Regression",
])

def show_boxplot(frame, title, log_scale=False):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=frame[PRICE_COLUMNS], ax=ax, palette="Set2", showmeans=True)
    ax.set_title(title)
    ax.set_ylabel("Hourly cost (USD)")
    ax.tick_params(axis="x", labelrotation=30)
    if log_scale:
        ax.set_yscale("log")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with overview:
    st.subheader("Load and Explore")
    c1, c2, c3 = st.columns(3)
    c1.metric("Records", len(raw))
    c2.metric("Original Columns", len(raw.columns))
    c3.metric("Raw Missing Cells", int(raw.isna().sum().sum()))
    st.dataframe(raw)
    st.subheader("Data Types")
    st.dataframe(pd.DataFrame({"Original": raw.dtypes.astype(str),
                              "Cleaned": data.dtypes.astype(str)}))
    st.subheader("Missing Prices After Cleaning")
    st.dataframe(data[PRICE_COLUMNS].isna().sum().rename("Missing Prices"))
    st.caption("Unavailable price text becomes missing, not zero. Memory is in GiB; CPU counts use the first number.")

with analysis:
    st.subheader("Pricing Summary Statistics")
    st.dataframe(data[PRICE_COLUMNS].describe())
    log_cost = st.checkbox("Logarithmic boxplot scale", value=True)
    show_boxplot(data, "EC2 Hourly Price Distributions", log_cost)
    st.caption("The box spans the middle 50%; the line is the median. Points beyond the whiskers are potential outliers.")
    st.subheader("On-Demand Outliers Using IQR")
    q1, q3 = data["On Demand"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = data[(data["On Demand"] < lower) | (data["On Demand"] > upper)]
    st.write(f"Q1: ${q1:.4f} · Q3: ${q3:.4f} · IQR: ${iqr:.4f}")
    st.write(f"Bounds: ${lower:.4f} to ${upper:.4f} · {len(outliers)} potential outliers")
    st.dataframe(outliers.sort_values("On Demand", ascending=False))
    st.caption("Bounds are calculated on original dollar values. Statistical outliers can be valid specialized instances.")
    st.subheader("On-Demand vs Linux Reserved")
    comparison = data[["Name", "API Name", "On Demand", "Linux Reserved cost"]].dropna()
    comparison = comparison[comparison["On Demand"] > 0].copy()
    comparison["Hourly Difference"] = comparison["On Demand"] - comparison["Linux Reserved cost"]
    comparison["Difference (%)"] = 100 * comparison["Hourly Difference"] / comparison["On Demand"]
    st.dataframe(comparison.sort_values("On Demand"))
    st.caption("These are differences in the supplied rates. Purchase terms and full deployment costs are not specified by this table.")

with families:
    st.subheader("Compare Exact Instance Families")
    chosen = st.multiselect("Instance families", sorted(data["Family"].unique()), default=["t2", "t3"])
    for family in chosen:
        subset = data[data["Family"] == family]
        st.subheader(f"{family}: {len(subset)} instance types")
        st.dataframe(subset[PRICE_COLUMNS].describe())
        show_boxplot(subset, f"{family} Hourly Prices", log_cost)
    family_comparison = data[data["Family"].isin(chosen)][
        ["Name", "API Name", "Family", "On Demand", "Linux Reserved cost"]
    ].dropna(subset=["On Demand", "Linux Reserved cost"]).sort_values("On Demand")
    st.subheader("Lowest-Cost Family Comparisons")
    st.dataframe(family_comparison.head(10))
    st.caption("Exact API prefixes keep t3 separate from t3a. No family selected produces an empty comparison.")

with regression:
    st.subheader("Predict On-Demand Hourly Price")
    st.write("Predictors: memory in GiB and vCPU count. Target: Linux on-demand USD/hour.")
    training_data = data.dropna(subset=["On Demand", "Instance Memory", "vCPUs"])
    training_data = training_data[training_data["On Demand"] > 0]
    X = np.log(training_data[["Instance Memory", "vCPUs"]])
    y = training_data["On Demand"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = TransformedTargetRegressor(
        regressor=LinearRegression(), func=np.log, inverse_func=np.exp,
    ).fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    st.write(f"Usable rows: {len(X)} · Training: {len(X_train)} · Test: {len(X_test)} · Seed: 42")
    fitted = model.regressor_
    st.code(f"Hourly price = exp({fitted.intercept_:.6f} + ({fitted.coef_[0]:.6f} × log(memory GiB)) + ({fitted.coef_[1]:.6f} × log(vCPUs)))")
    st.caption("The model fits log(price) from log(memory) and log(vCPUs), then exponentiates predictions to keep prices positive. Evaluation below uses USD/hour predictions.")
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE (USD/hour)", f"{mae:.4f}")
    c2.metric("MSE (squared USD/hour)", f"{mse:.4f}")
    c3.metric("RMSE (USD/hour)", f"{rmse:.4f}")
    st.caption("MAE measures average absolute error. RMSE emphasizes larger errors. Coefficients describe this fitted dataset, not a causal pricing rule.")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, predictions, alpha=0.6)
    low, high = min(y_test.min(), predictions.min()), max(y_test.max(), predictions.max())
    ax.plot([low, high], [low, high], "r--", label="Perfect prediction")
    ax.set(xlabel="Actual hourly price (USD)", ylabel="Predicted hourly price (USD)", title="Actual vs Predicted Test Prices")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    results = pd.DataFrame({"API Name": training_data.loc[X_test.index, "API Name"],
                            "Actual": y_test, "Predicted": predictions})
    results["Absolute Error"] = (results["Actual"] - results["Predicted"]).abs()
    st.dataframe(results.sort_values("Absolute Error", ascending=False))
    st.subheader("Predict a New Configuration")
    memory = st.number_input("Memory (GiB)", min_value=0.5, value=4.0, step=0.5)
    cpus = st.number_input("vCPUs", min_value=1, value=2, step=1)
    new = pd.DataFrame({"Instance Memory": [memory], "vCPUs": [cpus]})
    prediction = float(model.predict(np.log(new))[0])
    st.metric("Predicted On-Demand Cost (USD/hour)", f"${prediction:.4f}")
    st.caption("This two-feature baseline omits instance family, architecture, accelerators, storage, and network. Predictions are educational estimates, not verified AWS quotes.")
    st.download_button("Download Test Predictions", results.to_csv(index=False), "ec2_test_predictions.csv", "text/csv")
