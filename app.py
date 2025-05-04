# AF3005 - Programming for Finance
# Assignment 3: Machine Learning on Financial Data with Streamlit
# Instructor: Dr. Usama Arshad
# Student: [Your Name]
# Section: [BSFT06A/B/C]

# How to Run:
# Save this script as 'AF3005_Assignment_3.py'
# Then run the app using:
# streamlit run AF3005_Assignment_3.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from io import BytesIO

# Set page configuration
st.set_page_config(page_title="Finance ML App", layout="wide", page_icon="💹")

# --- Welcome Interface ---
st.title("📊 Machine Learning on Financial Data")
st.image("https://media.giphy.com/media/QpVUMRUJGokfqXyfa1/giphy.gif", use_container_width=True)
st.markdown("""
Welcome to the **AF3005 Finance ML App**.
Upload your dataset or fetch real-time financial data from Yahoo Finance to begin.
""")

# --- Sidebar ---
st.sidebar.header("📁 Load Data")
data_source = st.sidebar.radio("Choose Data Source:", ["Kragle Dataset (via URL)", "Yahoo Finance"])

df = None
if data_source == "Kragle Dataset (via URL)":
    dataset_url = st.sidebar.text_input("Enter the URL of the Kragle dataset (CSV format):")
    if dataset_url:
        try:
            df = pd.read_csv(dataset_url)
            st.success("✅ Kragle dataset loaded successfully!")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"❌ Failed to load data: {e}")
elif data_source == "Yahoo Finance":
    ticker = st.sidebar.text_input("Enter Ticker Symbol (e.g., AAPL, GOOGL)", value="AAPL")
    start_date = st.sidebar.date_input("Start Date")
    end_date = st.sidebar.date_input("End Date")
    if st.sidebar.button("Fetch Data"):
        df = yf.download(ticker, start=start_date, end=end_date)
        if not df.empty:
            df.reset_index(inplace=True)
            st.success("✅ Yahoo Finance data loaded!")
            st.dataframe(df.head())
        else:
            st.warning("⚠️ No data fetched. Check ticker and dates.")

# --- ML Workflow ---
st.header("🔄 Step-by-Step Machine Learning Pipeline")

if df is not None and not df.empty:
    # Step 1: Preprocessing
    if st.button("1️⃣ Preprocessing"):
        st.info("Running preprocessing...")
        st.write("Missing values per column:")
        st.dataframe(df.isnull().sum())
        df.dropna(inplace=True)
        st.success("✅ Missing values removed.")

    # Step 2: Feature Engineering
    if st.button("2️⃣ Feature Engineering"):
        st.info("Selecting numeric features...")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        features = st.multiselect("Select feature columns:", options=numeric_cols, default=numeric_cols[:-1])
        target = st.selectbox("Select target column:", options=numeric_cols, index=numeric_cols.index('Close') if 'Close' in numeric_cols else len(numeric_cols)-1)
        st.success(f"✅ Selected features: {features}, Target: {target}")

    # Step 3: Train-Test Split
    if st.button("3️⃣ Train/Test Split"):
        if features and target:
            X = df[features]
            y = df[target]
            if not X.empty:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                st.success("✅ Data split successfully!")
                split_data = {"Train": len(X_train), "Test": len(X_test)}
                fig = px.pie(values=split_data.values(), names=split_data.keys(), title="Train/Test Split")
                st.plotly_chart(fig)
        else:
            st.warning("⚠️ Please select features and target first.")

    # Step 4: Model Training
    if st.button("4️⃣ Train Model"):
        model = LinearRegression()
        model.fit(X_train, y_train)
        st.success("✅ Linear Regression model trained.")

    # Step 5: Evaluation
    if st.button("5️⃣ Evaluate Model"):
        y_pred = model.predict(X_test)
        st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred))
        st.write("R-squared Score:", r2_score(y_test, y_pred))
        results_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
        fig2 = px.scatter(results_df, x="Actual", y="Predicted", color_discrete_sequence=["blue"], title="Actual vs Predicted")
        st.plotly_chart(fig2, use_container_width=True)

    # Step 6: Results Visualization & Download
    if st.button("6️⃣ Show Predictions & Download"):
        results_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
        st.dataframe(results_df.head())
        buffer = BytesIO()
        results_df.to_csv(buffer, index=False)
        st.download_button("📥 Download Predictions", data=buffer.getvalue(), file_name="predictions.csv")
else:
    st.warning("⚠️ Please load a dataset to begin.")
