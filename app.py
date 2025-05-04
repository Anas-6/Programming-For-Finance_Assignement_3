# AF3005 – Programming for Finance | Assignment 3
# Student: Muhammad Anas
# Instructor: Dr. Usama Arshad
# Section: BSFT06B
# Streamlit App: ML on Financial Data

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
from io import BytesIO
import datetime

# ---------------- Streamlit Config ----------------
st.set_page_config(page_title="Finance ML Dashboard", layout="wide", page_icon="📈")

# ---------------- Welcome UI ----------------
st.title("📊 ML on Financial Data – AF3005 Assignment 3")
st.image("https://media.giphy.com/media/QpVUMRUJGokfqXyfa1/giphy.gif", use_container_width=True)
st.markdown("""
Welcome to the **Finance ML Dashboard**! 
This app allows you to upload Kragle datasets or fetch Yahoo Finance data, perform preprocessing, apply ML models, and visualize results.
""")

# ---------------- Sidebar ----------------
st.sidebar.title("📁 Data Options")

data_source = st.sidebar.radio("Select Data Source", ["Kragle Dataset (URL)", "Yahoo Finance"])

df = None

if data_source == "Kragle Dataset (URL)":
    url = st.sidebar.text_input("Enter Direct CSV File URL:")
    if st.sidebar.button("Load Kragle Data"):
        try:
            df = pd.read_csv(url)
            st.sidebar.success("✅ Kragle data loaded!")
        except Exception as e:
            st.sidebar.error(f"❌ Error: {e}")

elif data_source == "Yahoo Finance":
    ticker = st.sidebar.text_input("Ticker (e.g., AAPL)", value="AAPL")
    start = st.sidebar.date_input("Start Date", datetime.date(2020, 1, 1))
    end = st.sidebar.date_input("End Date", datetime.date.today())
    if st.sidebar.button("Fetch Yahoo Data"):
        df = yf.download(ticker, start=start, end=end)
        if not df.empty:
            df.reset_index(inplace=True)
            st.sidebar.success("✅ Yahoo data loaded!")
        else:
            st.sidebar.warning("⚠️ No data found. Try changing the ticker or dates.")

if df is not None and not df.empty:
    st.subheader("📄 Raw Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    # Session State to persist selections
    if "df" not in st.session_state:
        st.session_state.df = df.copy()

    df = st.session_state.df

    # -------- Step 1: Preprocessing --------
    st.header("🔧 Step 1: Preprocessing")

    if st.button("Run Preprocessing"):
        st.info("Cleaning missing values...")
        df.dropna(inplace=True)
        st.write("✅ Dropped missing values.")

        # Remove outliers using Z-score
        numeric_cols = df.select_dtypes(include=np.number).columns
        z_scores = np.abs((df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std())
        df = df[(z_scores < 3).all(axis=1)]
        st.success("✅ Outliers removed using Z-score method.")
        st.dataframe(df.describe())

        st.session_state.df = df  # Save processed df

    # -------- Step 2: Feature Engineering --------
    st.header("🧠 Step 2: Feature Engineering")

    if st.button("Run Feature Engineering"):
        df['Daily Return'] = df['Close'].pct_change()
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['Volatility'] = df['Close'].rolling(window=10).std()
        df.dropna(inplace=True)
        st.success("✅ Features added: Daily Return, MA10, Volatility")
        st.line_chart(df[['Close', 'MA10']], use_container_width=True)

        st.session_state.df = df

    # -------- Step 3: Train-Test Split --------
    st.header("📉 Step 3: Train/Test Split")

    features = st.multiselect("Select feature columns:", df.select_dtypes(include=np.number).columns.tolist(), default=["Daily Return", "MA10", "Volatility"])
    target = st.selectbox("Select target column:", options=["Close"])

    if st.button("Split Data"):
        X = df[features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test

        pie_data = {"Train": len(X_train), "Test": len(X_test)}
        st.success("✅ Data split successfully!")
        fig = px.pie(names=pie_data.keys(), values=pie_data.values(), title="Train vs Test Split")
        st.plotly_chart(fig)

    # -------- Step 4: Model Selection & Training --------
    st.header("⚙️ Step 4: Machine Learning Models (Choose One)")

    model_type = st.selectbox("Choose a model:", ["Linear Regression", "Logistic Regression", "KMeans Clustering"])

    if st.button("Train Model"):
        if model_type == "Linear Regression":
            model = LinearRegression()
            model.fit(st.session_state.X_train, st.session_state.y_train)
            y_pred = model.predict(st.session_state.X_test)
            st.session_state.y_pred = y_pred
            st.session_state.model = model
            st.success("✅ Linear Regression model trained.")

        elif model_type == "Logistic Regression":
            y_binary = (st.session_state.y_train > st.session_state.y_train.median()).astype(int)
            model = LogisticRegression()
            model.fit(st.session_state.X_train, y_binary)
            y_pred = model.predict(st.session_state.X_test)
            st.session_state.y_pred = y_pred
            st.session_state.model = model
            st.success("✅ Logistic Regression model trained.")

        elif model_type == "KMeans Clustering":
            model = KMeans(n_clusters=3, random_state=42)
            model.fit(st.session_state.X_train)
            cluster_labels = model.predict(st.session_state.X_test)
            st.session_state.y_pred = cluster_labels
            st.session_state.model = model
            st.success("✅ KMeans clustering applied.")

    # -------- Step 5: Evaluation --------
    st.header("📈 Step 5: Evaluation")

    if st.button("Evaluate Model"):
        y_test = st.session_state.y_test
        y_pred = st.session_state.y_pred

        if model_type == "Linear Regression":
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.write("📊 Mean Squared Error:", round(mse, 2))
            st.write("📊 R-squared Score:", round(r2, 4))
            fig = px.scatter(x=y_test, y=y_pred, labels={"x": "Actual", "y": "Predicted"}, title="Actual vs Predicted", color_discrete_sequence=["blue"])
            fig.add_scatter(x=y_test, y=y_test, mode='lines', name='Ideal Fit', line=dict(color='green'))
            st.plotly_chart(fig)

        elif model_type == "Logistic Regression":
            y_binary_test = (y_test > y_test.median()).astype(int)
            acc = accuracy_score(y_binary_test, y_pred)
            st.write("📊 Accuracy:", round(acc, 3))
            st.write("📊 Confusion Matrix:")
            st.write(confusion_matrix(y_binary_test, y_pred))

        elif model_type == "KMeans Clustering":
            fig = px.scatter_matrix(st.session_state.X_test, color=st.session_state.y_pred, title="KMeans Cluster Visualization")
            st.plotly_chart(fig)

    # -------- Step 6: Results Download --------
    st.header("📥 Step 6: Results")

    if st.button("Download Predictions"):
        result_df = pd.DataFrame({"Actual": st.session_state.y_test, "Predicted": st.session_state.y_pred})
        st.dataframe(result_df.head())

        buffer = BytesIO()
        result_df.to_csv(buffer, index=False)
        st.download_button("Download CSV", data=buffer.getvalue(), file_name="results.csv", mime="text/csv")

else:
    st.warning("⚠️ Please load a dataset to start the workflow.")
