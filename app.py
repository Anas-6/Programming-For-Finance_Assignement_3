# AF3005 - Programming for Finance
# Assignment 3: Machine Learning on Financial Data with Streamlit
# Instructor: Dr. Usama Arshad
# Student: Muhammad Anas
# Section: BSFT06B

# How to Run:
# Convert this notebook to a .py file using:
# !jupyter nbconvert --to script AF3005_Assignment_3.ipynb
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
import os

st.set_page_config(page_title="Finance ML App", layout="wide", page_icon="💹")

# --- Initialize session state variables ---
if 'df' not in st.session_state:
    st.session_state.df = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'features' not in st.session_state:
    st.session_state.features = []
if 'target' not in st.session_state:
    st.session_state.target = None

# --- Welcome Interface ---
st.title("📊 Machine Learning on Financial Data")
st.image("https://media.giphy.com/media/QpVUMRUJGokfqXyfa1/giphy.gif", use_column_width=True)
st.markdown("""
Welcome to the **AF3005 Finance ML App**.
Upload your dataset or fetch real-time financial data from Yahoo Finance to begin.
""")

# --- Sidebar ---
st.sidebar.header("📁 Load Data")
data_source = st.sidebar.radio("Choose Data Source:", ["Upload Kragle Dataset", "Yahoo Finance"])

if data_source == "Upload Kragle Dataset":
    uploaded_file = st.sidebar.file_uploader("Upload Kragle Dataset (.csv)", type=["csv"])
    if uploaded_file:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.success("✅ Kragle dataset uploaded successfully!")
        st.dataframe(st.session_state.df.head())
elif data_source == "Yahoo Finance":
    ticker = st.sidebar.text_input("Enter Ticker Symbol (e.g., AAPL, GOOGL)", value="AAPL")
    start_date = st.sidebar.date_input("Start Date")
    end_date = st.sidebar.date_input("End Date")
    if st.sidebar.button("Fetch Data"):
        df = yf.download(ticker, start=start_date, end=end_date)
        if not df.empty:
            df.reset_index(inplace=True)
            st.session_state.df = df
            st.success("✅ Yahoo Finance data loaded!")
            st.dataframe(df.head())
        else:
            st.warning("⚠️ No data fetched. Check ticker and dates.")

# --- ML Workflow ---
st.header("🔄 Step-by-Step Machine Learning Pipeline")

df = st.session_state.df
if df is not None and not df.empty:
    # Step 1: Preprocessing
    if st.button("1️⃣ Preprocessing"):
        st.info("Running preprocessing...")
        st.write("Missing values per column:")
        st.dataframe(df.isnull().sum())
        df.dropna(inplace=True)
        st.session_state.df = df
        st.success("✅ Missing values removed.")

    # Step 2: Feature Engineering
    if st.button("2️⃣ Feature Engineering"):
        st.info("Selecting numeric features...")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        st.session_state.features = st.multiselect("Select feature columns:", options=numeric_cols, default=numeric_cols[:-1])
        st.session_state.target = st.selectbox("Select target column:", options=numeric_cols, index=len(numeric_cols)-1)
        st.success(f"✅ Selected features: {st.session_state.features}, Target: {st.session_state.target}")

    # Step 3: Train-Test Split
    if st.button("3️⃣ Train/Test Split"):
        if st.session_state.features and st.session_state.target:
            X = df[st.session_state.features]
            y = df[st.session_state.target]
            st.session_state.X_train, st.session_state.X_test, st.session_state.y_train, st.session_state.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            st.success("✅ Data split successfully!")
            split_data = {"Train": len(st.session_state.X_train), "Test": len(st.session_state.X_test)}
            fig = px.pie(values=split_data.values(), names=split_data.keys(), title="Train/Test Split")
            st.plotly_chart(fig)
        else:
            st.warning("⚠️ Please select features and target first.")

    # Step 4: Model Training
    if st.button("4️⃣ Train Model"):
        st.session_state.model = LinearRegression()
        st.session_state.model.fit(st.session_state.X_train, st.session_state.y_train)
        st.success("✅ Linear Regression model trained.")

    # Step 5: Evaluation
    if st.button("5️⃣ Evaluate Model"):
        y_pred = st.session_state.model.predict(st.session_state.X_test)
        st.write("Mean Squared Error:", mean_squared_error(st.session_state.y_test, y_pred))
        st.write("R-squared Score:", r2_score(st.session_state.y_test, y_pred))
        fig2 = px.scatter(x=st.session_state.y_test, y=y_pred, labels={"x": "Actual", "y": "Predicted"}, title="Actual vs Predicted")
        st.plotly_chart(fig2)

    # Step 6: Results Visualization & Download
    if st.button("6️⃣ Show Predictions & Download"):
        y_pred = st.session_state.model.predict(st.session_state.X_test)
        results_df = pd.DataFrame({"Actual": st.session_state.y_test, "Predicted": y_pred})
        st.dataframe(results_df.head())
        buffer = BytesIO()
        results_df.to_csv(buffer, index=False)
        st.download_button("📥 Download Predictions", data=buffer.getvalue(), file_name="predictions.csv")
else:
    st.warning("⚠️ Please load a dataset to begin.")
