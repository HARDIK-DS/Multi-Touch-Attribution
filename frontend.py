import streamlit as st
import pandas as pd
import requests
import json
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# Backend URLs
# ---------------------------
FASTAPI_URL = "http://127.0.0.1:8000"
EDA_URL = f"{FASTAPI_URL}/eda"
PREDICT_CSV_URL = f"{FASTAPI_URL}/predict_from_csv"
PREDICT_MANUAL_URL = f"{FASTAPI_URL}/predict_from_manual"

# ---------------------------
# Helper Function for File Upload
# ---------------------------
def upload_file(url, uploaded_file):
    if uploaded_file is not None:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
        try:
            response = requests.post(url, files=files, timeout=20)
            return response
        except requests.exceptions.RequestException as e:
            st.error(f"Connection error: {e}")
            return None
    else:
        st.warning("Please upload a CSV file.")
        return None

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("📊 Revenue Predictor Dashboard - Multi-Touch Attribution")

# Sidebar navigation
st.sidebar.header("Navigation")
option = st.sidebar.radio("Choose an option", ["EDA on CSV", "Predict from CSV", "Manual Prediction"])

# --------------------- EDA on CSV ---------------------
if option == "EDA on CSV":
    st.header("Exploratory Data Analysis (EDA)")
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    if uploaded_file:
        response = upload_file(EDA_URL, uploaded_file)
        if response and response.status_code == 200:
            eda_data = response.json()

            # Extract processed data from backend
            shape = eda_data.get("Shape", None)
            missing_values = eda_data.get("Missing_Values", None)
            numeric_summary = eda_data.get("Numeric_Summary", None)
            top_ad_groups = eda_data.get("Top_Ad_Groups", None)
            top_months = eda_data.get("Top_Months", None)
            correlation_data = eda_data.get("Correlation", None)

            # Dataset Shape
            if shape:
                st.subheader("Dataset Shape")
                st.write(f"Rows: {shape[0]}, Columns: {shape[1]}")

            # Missing Values
            if missing_values:
                st.subheader("Missing Values")
                st.write(pd.DataFrame(missing_values, index=["Count"]).T)

            # Numeric Summary
            if numeric_summary:
                st.subheader("Numeric Summary")
                st.write(pd.DataFrame(numeric_summary))

            # Top Ad Groups
            if top_ad_groups:
                st.subheader("Top Ad Groups")
                st.bar_chart(pd.DataFrame(list(top_ad_groups.items()), columns=["Ad_Group", "Count"]).set_index("Ad_Group"))

            # Top Months
            if top_months:
                st.subheader("Top Months")
                st.bar_chart(pd.DataFrame(list(top_months.items()), columns=["Month", "Count"]).set_index("Month"))

            # Correlation Heatmap
            if correlation_data:
                st.subheader("Correlation Heatmap")
                corr_df = pd.DataFrame(correlation_data)
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.heatmap(corr_df, cmap="coolwarm", annot=True, ax=ax)
                st.pyplot(fig)
        else:
            st.error(f"Error: {response.status_code if response else 'No response'}")

# --------------------- Predict from CSV ---------------------
if option == "Predict from CSV":
    st.subheader("Predict Revenue from CSV")
    uploaded_file = st.file_uploader("Upload CSV for Prediction", type=["csv"], key="predict_csv")
    if st.button("Predict from CSV"):
        if uploaded_file:
            response = upload_file(PREDICT_CSV_URL, uploaded_file)
            if response and response.status_code == 200:
                pred_result = response.json()
                st.success("Prediction Results:")
                st.write(pred_result)
            else:
                st.error(f"Error: {response.json() if response else 'No response'}")

# --------------------- Manual Prediction ---------------------
if option == "Manual Prediction":
    st.subheader("Manual Revenue Prediction")

    # User inputs
    impressions = st.number_input("Impressions", min_value=0)
    clicks = st.number_input("Clicks", min_value=0)
    ctr = st.number_input("CTR (%)", min_value=0.0, format="%.2f")
    conversions = st.number_input("Conversions", min_value=0)
    conv_rate = st.number_input("Conversion Rate (%)", min_value=0.0, format="%.2f")
    cost = st.number_input("Cost", min_value=0.0, format="%.2f")
    cpc = st.number_input("CPC", min_value=0.0, format="%.2f")
    sale_amount = st.number_input("Sale Amount", min_value=0.0, format="%.2f")
    pnl = st.number_input("PnL (Profit/Loss)", format="%.2f")  # allows negative
    ad_group = st.text_input("Ad Group")
    month = st.text_input("Month")

    if st.button("Predict Manually"):
        input_data = {
            "Impressions": impressions,
            "Clicks": clicks,
            "CTR": ctr,
            "Conversions": conversions,
            "Conv_Rate": conv_rate,
            "Cost": cost,
            "CPC": cpc,
            "Sale_Amount": sale_amount,
            "PnL": pnl,
            "Ad_Group": ad_group,
            "Month": month
        }
        try:
            response = requests.post(PREDICT_MANUAL_URL, json=input_data, timeout=20)
            if response.status_code == 200:
                st.success("Manual Prediction Result:")
                st.write(response.json())
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Connection error: {e}")
