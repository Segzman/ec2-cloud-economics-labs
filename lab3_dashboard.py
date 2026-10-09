import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(layout="wide")

st.title("EC2 Instance EDA Dashboard")

raw_df = pd.read_csv(Path(__file__).with_name("ec2dataset.csv"))
df = raw_df.copy()

df["Memory_GiB"] = (
    df["Instance Memory"]
    .str.extract(r"([\d.]+)")
    .astype(float)
)

df["vCPU_Count"] = (
    df["vCPUs"]
    .str.extract(r"(\d+)")
    .astype(float)
)

def clean_price(value):
    if pd.isna(value):
        return None
    value = str(value)
    if "unavailable" in value.lower():
        return None
    return float(
        value.replace("$", "")
        .replace(" hourly", "")
        .strip()
    )


price_columns = [
    "On Demand",
    "Linux Reserved cost",
    "Linux Spot Minimum cost",
    "Windows On Demand cost",
    "Windows Reserved cost",
]

for column in price_columns:
    df[column + "_USD"] = df[column].apply(clean_price)

df["Monthly_On_Demand"] = df["On Demand_USD"] * 730
df["Cost_Per_GiB"] = df["On Demand_USD"] / df["Memory_GiB"]
df["Cost_Per_vCPU"] = df["On Demand_USD"] / df["vCPU_Count"]

st.sidebar.header("Filters")
max_memory = float(df["Memory_GiB"].max())
memory_filter = st.sidebar.slider(
    "Maximum Memory (GiB)",
    min_value=0.5,
    max_value=max_memory,
    value=max_memory,
)

cpu_values = sorted(df["vCPU_Count"].dropna().unique())
selected_cpu = st.sidebar.multiselect(
    "vCPU Count",
    options=cpu_values,
    default=cpu_values,
)

network_values = sorted(df["Network Performance"].dropna().unique())
selected_network = st.sidebar.multiselect(
    "Network Performance",
    options=network_values,
    default=network_values,
)

max_hourly_cost = float(df["On Demand_USD"].max())
price_filter = st.sidebar.slider(
    "Maximum Hourly Cost (USD)",
    min_value=0.0,
    max_value=max_hourly_cost,
    value=max_hourly_cost,
    step=0.0001,
    format="$%.4f",
)
st.sidebar.caption("The hourly cost limit excludes instances with unavailable on-demand prices.")

filtered_df = df[
    (df["Memory_GiB"] <= memory_filter)
    & (df["vCPU_Count"].isin(selected_cpu))
    & (df["Network Performance"].isin(selected_network))
    & (df["On Demand_USD"] <= price_filter)
]

st.subheader("EC2 Summary")
col1, col2, col3, col4 = st.columns(4)

col1.metric("Instances", len(filtered_df))
col2.metric(
    "Avg Memory",
    f"{filtered_df['Memory_GiB'].mean():.2f} GiB" if not filtered_df.empty else "N/A",
)
col3.metric(
    "Avg vCPUs",
    f"{filtered_df['vCPU_Count'].mean():.1f}" if not filtered_df.empty else "N/A",
)
average_cost = filtered_df["On Demand_USD"].mean()
col4.metric(
    "Avg Hourly Cost",
    f"${average_cost:.4f}" if pd.notna(average_cost) else "N/A",
)

overview_tab, eda_tab, cost_tab, pricing_tab, export_tab = st.tabs(
    ["Overview", "EDA", "Cost Analysis", "Pricing Comparison", "Export"]
)

with overview_tab:
    st.write("Dataset Preview")
    st.dataframe(raw_df)

    st.subheader("Dataset Information")
    col1, col2, col3 = st.columns(3)

    col1.metric("Number of Instances", len(raw_df))
    col2.metric("Number of Columns", len(raw_df.columns))
    col3.metric("Raw Missing Values", raw_df.isna().sum().sum())

    st.subheader("Dataset Structure")
    st.write("Columns:")
    st.write(raw_df.columns.tolist())

    st.subheader("Data Types")
    st.write(raw_df.dtypes.astype(str))

    st.subheader("Filtered EC2 Instances")
    st.write(f"{len(filtered_df)} instances found")
    st.dataframe(filtered_df)

    with st.expander("Data structure and cleaning walkthrough"):
        st.subheader("Memory Cleaning Preview")
        st.dataframe(df[["Instance Memory", "Memory_GiB"]].head(10))

        st.subheader("vCPU Cleaning Preview")
        st.dataframe(df[["vCPUs", "vCPU_Count"]].head(10))


        st.subheader("Pricing Cleaning Preview")
        st.dataframe(df[["On Demand", "On Demand_USD"]].head(10))

        st.write("Missing prices after cleaning:")
        st.dataframe(df[[column + "_USD" for column in price_columns]].isna().sum())


with eda_tab:
    st.subheader("Memory Distribution")
    fig = px.histogram(
        filtered_df,
        x="Memory_GiB",
        nbins=30,
        title="Distribution of EC2 Memory",
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("vCPU Distribution")
    fig = px.histogram(
        filtered_df,
        x="vCPU_Count",
        title="Distribution of vCPUs",
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("Memory vs vCPUs")
    log_scale = st.checkbox("Use logarithmic axes", value=True)
    st.caption(
        "Log axes spread out smaller instances. Drag a rectangle to zoom; "
        "double-click to reset. Instances with identical memory and CPU counts overlap."
    )
    fig = px.scatter(
        filtered_df,
        x="vCPU_Count",
        y="Memory_GiB",
        hover_name="API Name",
        hover_data=["On Demand_USD"],
        title="EC2 Memory vs vCPU",
        log_x=log_scale,
        log_y=log_scale,
        height=650,
    )
    fig.update_traces(marker=dict(size=7, opacity=0.65, line=dict(width=0.5, color="white")))
    fig.update_layout(dragmode="zoom", hovermode="closest")
    st.plotly_chart(fig, width="stretch", config={"scrollZoom": True})

    st.subheader("Memory vs On-Demand Cost")
    cost_plot_df = filtered_df.dropna(subset=["Memory_GiB", "On Demand_USD", "vCPU_Count"])
    fig = px.scatter(
        cost_plot_df,
        x="Memory_GiB",
        y="On Demand_USD",
        hover_name="API Name",
        size="vCPU_Count",
        size_max=24,
        title="Memory vs EC2 On-Demand Cost",
        labels={"Memory_GiB": "Memory (GiB)", "On Demand_USD": "Hourly Cost (USD)"},
        log_x=log_scale,
        log_y=log_scale,
        height=650,
    )
    fig.update_traces(marker=dict(opacity=0.65, line=dict(width=0.5, color="white")))
    fig.update_layout(dragmode="zoom", hovermode="closest")
    st.plotly_chart(fig, width="stretch", config={"scrollZoom": True})
    st.caption("Dot size represents vCPU count. Instances with unavailable prices are omitted.")


with cost_tab:
    st.subheader("Monthly On-Demand Cost")
    st.dataframe(
        filtered_df[
            [
                "Name",
                "API Name",
                "Memory_GiB",
                "vCPU_Count",
                "On Demand_USD",
                "Monthly_On_Demand",
            ]
        ]
    )

    st.subheader("Lowest-Cost EC2 Instances")
    cheapest = (
        filtered_df.dropna(subset=["On Demand_USD"])
        .sort_values("On Demand_USD")
        [
            [
                "Name",
                "API Name",
                "Memory_GiB",
                "vCPU_Count",
                "On Demand_USD",
                "Monthly_On_Demand",
            ]
        ]
        .head(10)
    )
    st.dataframe(cheapest)

    st.subheader("Highest-Cost EC2 Instances")
    most_expensive = (
        filtered_df.dropna(subset=["On Demand_USD"])
        .sort_values("On Demand_USD", ascending=False)
        [
            [
                "Name",
                "API Name",
                "Memory_GiB",
                "vCPU_Count",
                "On Demand_USD",
                "Monthly_On_Demand",
            ]
        ]
        .head(10)
    )
    st.dataframe(most_expensive)

    st.subheader("Cost per GiB of Memory")
    efficiency = (
        filtered_df.dropna(subset=["Cost_Per_GiB"])
        .sort_values("Cost_Per_GiB")
        [
            [
                "Name",
                "API Name",
                "Memory_GiB",
                "vCPU_Count",
                "On Demand_USD",
                "Cost_Per_GiB",
            ]
        ]
        .head(15)
    )
    st.caption(
        "Cost per GiB is hourly on-demand USD divided by memory in GiB. "
        "Lower values mean cheaper memory capacity; CPU, storage, network, "
        "and workload requirements also matter."
    )
    st.dataframe(efficiency)

    st.subheader("Cost per vCPU")
    cpu_efficiency = (
        filtered_df.dropna(subset=["Cost_Per_vCPU"])
        .sort_values("Cost_Per_vCPU")
        [["Name", "API Name", "Memory_GiB", "vCPU_Count", "On Demand_USD", "Cost_Per_vCPU"]]
        .head(15)
    )
    st.caption("Hourly on-demand USD per vCPU. This compares price per CPU, not measured CPU performance.")
    st.dataframe(cpu_efficiency)


with pricing_tab:
    pricing = filtered_df[
        [
            "Name",
            "API Name",
            "On Demand_USD",
            "Linux Reserved cost_USD",
            "Linux Spot Minimum cost_USD",
            "Windows On Demand cost_USD",
            "Windows Reserved cost_USD",
        ]
    ]

    st.subheader("EC2 Pricing Comparison")
    st.caption("All prices are hourly USD values from the dataset. Blank cells indicate unavailable prices.")
    st.dataframe(pricing)

    st.subheader("Instance Pricing Comparison")
    if filtered_df.empty:
        st.info("Adjust the filters to select an EC2 instance.")
    else:
        selected_instance = st.selectbox(
            "Select an EC2 Instance",
            filtered_df["API Name"].unique(),
        )
        instance = filtered_df[
            filtered_df["API Name"] == selected_instance
        ].iloc[0]

        pricing_data = pd.DataFrame({
            "Pricing Model": ["On Demand", "Linux Reserved", "Linux Spot"],
            "Hourly Cost": [
                instance["On Demand_USD"],
                instance["Linux Reserved cost_USD"],
                instance["Linux Spot Minimum cost_USD"],
            ],
        }).dropna(subset=["Hourly Cost"])

        if pricing_data.empty:
            st.info("No Linux prices are available for this instance in the dataset.")
        else:
            fig = px.bar(
                pricing_data,
                x="Pricing Model",
                y="Hourly Cost",
                title=f"Pricing Comparison: {selected_instance}",
                labels={"Hourly Cost": "Hourly Cost (USD)"},
            )
            st.plotly_chart(fig, width="stretch")

            pricing_data["Monthly Cost"] = pricing_data["Hourly Cost"] * 730
            st.caption("Monthly estimates use 730 hours at the dataset's hourly rate.")
            st.dataframe(pricing_data)


with export_tab:
    st.subheader("Data Export")
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="Download Filtered Dataset",
        data=csv,
        file_name="filtered_ec2_instances.csv",
        mime="text/csv",
    )
