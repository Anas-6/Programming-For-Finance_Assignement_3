# AF3005 - Programming for Finance
# Assignment 3: Machine Learning on Financial Data with Streamlit
# Instructor: Dr. Usama Arshad
# Student: [Your Name]
# Section: [BSFT06A/B/C]

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

st.set_page_config(page_title="Finance ML App", layout="wide", page_icon="\ud83d\udcc9")

# --- Welcome Interface ---
st.title("\ud83d\udcca Machine Learning on Financial Data")
st.image("https://media.giphy.com/media/QpVUMRUJGokfqXyfa1/giphy.gif", use_container_width=True)
st.markdown("""
Welcome to the **AF3005 Finance ML App**.
Upload your dataset or fetch real-time financial data from Yahoo Finance to begin.
""")

# --- Initialize session state ---
if "df" not in st.session_state:
    st.session_state.df = None
if "features" not in st.session_state:
    st.session_state.features = []
if "target" not in st.session_state:
    st.session_state.target = None
if "model" not in st.session_state:
    st.session_state.model = None

# --- Sidebar ---
st.sidebar.header("\ud83d\udcc1 Load Data")
data_source = st.sidebar.radio("Choose Data Source:", ["Kragle Dataset (via URL)", "Yahoo Finance"])

if data_source == "Kragle Dataset (via URL)":
    dataset_url = st.sidebar.text_input("Enter direct link to CSV file")
    if dataset_url:
        try:
            df = pd.read_csv(dataset_url)
            st.session_state.df = df
            st.success("\u2705 Kragle dataset loaded successfully!")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"\u274c Failed to load data: {e}")
elif data_source == "Yahoo Finance":
    ticker = st.sidebar.text_input("Enter Ticker Symbol (e.g., AAPL, GOOGL)", value="AAPL")
    start_date = st.sidebar.date_input("Start Date")
    end_date = st.sidebar.date_input("End Date")
    if st.sidebar.button("Fetch Data"):
        df = yf.download(ticker, start=start_date, end=end_date)
        if not df.empty:
            df.reset_index(inplace=True)
            st.session_state.df = df
            st.success("\u2705 Yahoo Finance data loaded!")
            st.dataframe(df.head())
        else:
            st.warning("\u26a0\ufe0f No data fetched. Check ticker and dates.")

# --- Main ML Workflow ---
df = st.session_state.df

st.header("\ud83d\udd04 Step-by-Step Machine Learning Pipeline")

# Step 1: Preprocessing
if st.button("1\ufe0f\u20e3 Preprocessing"):
    if df is not None:
        st.info("Running preprocessing...")
        st.write("Missing values per column:")
        st.dataframe(df.isnull().sum())
        st.session_state.df = df.dropna()
        st.success("\u2705 Missing values removed.")
    else:
        st.warning("\u26a0\ufe0f Load data first.")

# Step 2: Feature Engineering
if st.button("2\ufe0f\u20e3 Feature Engineering"):
    df = st.session_state.df
    if df is not None:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        features = st.multiselect("Select feature columns:", options=numeric_cols, default=numeric_cols[:-1])
        default_target = "Close" if "Close" in numeric_cols else numeric_cols[-1]
        target = st.selectbox("Select target column:", options=numeric_cols, index=numeric_cols.index(default_target))
        st.session_state.features = features
        st.session_state.target = target
        st.success(f"\u2705 Selected features: {features}, Target: {target}")
    else:
        st.warning("\u26a0\ufe0f Load and preprocess data first.")

# Step 3: Train-Test Split
if st.button("3\ufe0f\u20e3 Train/Test Split"):
    df = st.session_state.df
    features = st.session_state.features
    target = st.session_state.target
    if df is not None and features and target:
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.success("\u2705 Data split successfully!")
        split_data = {"Train": len(X_train), "Test": len(X_test)}
        fig = px.pie(values=split_data.values(), names=split_data.keys(), title="Train/Test Split")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("\u26a0\ufe0f Complete feature selection first.")

# Step 4: Model Training
if st.button("4\ufe0f\u20e3 Train Model"):
    if "X_train" in st.session_state:
        model = LinearRegression()
        model.fit(st.session_state.X_train, st.session_state.y_train)
        st.session_state.model = model
        st.success("\u2705 Linear Regression model trained.")
    else:
        st.warning("\u26a0\ufe0f Run Train/Test Split first.")

# Step 5: Evaluation
if st.button("5\ufe0f\u20e3 Evaluate Model"):
    if "model" in st.session_state:
        y_pred = st.session_state.model.predict(st.session_state.X_test)
        mse = mean_squared_error(st.session_state.y_test, y_pred)
        r2 = r2_score(st.session_state.y_test, y_pred)
        st.write("**Mean Squared Error:**", mse)
        st.write("**R-squared Score:**", r2)
        fig2 = px.scatter(
            x=st.session_state.y_test,
            y=y_pred,
            labels={"x": "Actual", "y": "Predicted"},
            title="Actual vs Predicted",
            color_discrete_sequence=["red"]
        )
        fig2.add_scatter(x=st.session_state.y_test, y=st.session_state.y_test, mode='lines', name='Ideal Fit')
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("\u26a0\ufe0f Train the model first.")

# Step 6: Results & Download
if st.button("6\ufe0f\u20e3 Show Predictions & Download"):
    if "model" in st.session_state:
        y_pred = st.session_state.model.predict(st.session_state.X_test)
        results_df = pd.DataFrame({
            "Actual": st.session_state.y_test,
            "Predicted": y_pred
        })
        st.dataframe(results_df.head())
        buffer = BytesIO()
        results_df.to_csv(buffer, index=False)
        st.download_button("\ud83d\udcc5 Download Predictions", data=buffer.getvalue(), file_name="predictions.csv")
    else:
        st.warning("\u26a0\ufe0f Train the model first.")

# Fallback warning
if st.session_state.df is None:
    st.warning("\u26a0\ufe0f Please load a dataset to begin.")
