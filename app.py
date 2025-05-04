# AF3005 - Programming for Finance
# Assignment 3: Machine Learning on Financial Data with Streamlit
# Instructor: Dr. Usama Arshad
# Student: Muhammad Anas
# Section: BSFT06B

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Finance ML Dashboard", layout="wide", page_icon="💹")

# ----------------- Initialize Session State -----------------
if 'df' not in st.session_state:
    st.session_state.df = None
if 'features' not in st.session_state:
    st.session_state.features = []
if 'target' not in st.session_state:
    st.session_state.target = None
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
if 'y_pred' not in st.session_state:
    st.session_state.y_pred = None

# ----------------- Header -----------------
st.title("💹 Machine Learning on Financial Data")
st.image("https://media.giphy.com/media/QpVUMRUJGokfqXyfa1/giphy.gif", use_container_width=True)
st.markdown("Welcome to the **AF3005 Finance ML Dashboard**. Upload Kragle data or fetch live data from Yahoo Finance to build and evaluate ML models.")

# ----------------- Sidebar -----------------
st.sidebar.header("📁 Load Data")
data_option = st.sidebar.radio("Choose Data Source", ["Kragle CSV Link", "Yahoo Finance"])

if data_option == "Kragle CSV Link":
    csv_link = st.sidebar.text_input("Paste Kragle CSV Link:")
    if st.sidebar.button("Load Kragle Data"):
        try:
            st.session_state.df = pd.read_csv(csv_link)
            st.success("✅ Kragle dataset loaded successfully!")
        except Exception as e:
            st.error(f"Error loading dataset: {e}")

elif data_option == "Yahoo Finance":
    ticker = st.sidebar.text_input("Enter Ticker Symbol (e.g. AAPL)", value="AAPL")
    start_date = st.sidebar.date_input("Start Date")
    end_date = st.sidebar.date_input("End Date")
    if st.sidebar.button("Fetch Yahoo Finance Data"):
        df_yahoo = yf.download(ticker, start=start_date, end=end_date)
        if not df_yahoo.empty:
            df_yahoo.reset_index(inplace=True)
            st.session_state.df = df_yahoo
            st.success("✅ Yahoo Finance data fetched successfully!")
        else:
            st.warning("⚠️ No data returned from Yahoo Finance.")

# ----------------- Display Loaded Data -----------------
if st.session_state.df is not None:
    st.subheader("📊 Loaded Data Preview")
    st.dataframe(st.session_state.df.head())

# ----------------- ML Pipeline Buttons -----------------
st.header("⚙️ Step-by-Step ML Workflow")

# Step 1: Preprocessing
if st.button("1️⃣ Preprocessing"):
    df = st.session_state.df.copy()
    st.write("Missing values per column:")
    st.dataframe(df.isnull().sum())
    df.dropna(inplace=True)
    st.session_state.df = df
    st.success("✅ Missing values dropped.")

# Step 2: Feature Engineering
if st.button("2️⃣ Feature Engineering"):
    df = st.session_state.df
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    st.session_state.features = st.multiselect("Select feature columns:", options=numeric_cols, default=numeric_cols[:-1])
    st.session_state.target = st.selectbox("Select target column:", options=numeric_cols, index=numeric_cols.index("Close") if "Close" in numeric_cols else len(numeric_cols) - 1)
    
    st.success(f"✅ Selected features: {st.session_state.features}")
    st.success(f"✅ Target variable: {st.session_state.target}")

# Step 3: Train/Test Split
if st.button("3️⃣ Train/Test Split"):
    df = st.session_state.df
    if st.session_state.features and st.session_state.target:
        X = df[st.session_state.features]
        y = df[st.session_state.target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test

        split_sizes = {"Train": len(X_train), "Test": len(X_test)}
        fig = px.pie(names=split_sizes.keys(), values=split_sizes.values(), title="Train/Test Split")
        st.plotly_chart(fig)
        st.success("✅ Data split completed.")
    else:
        st.warning("⚠️ Please select features and target first.")

# ---------------- Step 4: Model Selection ----------------
st.subheader("⚙️ Step 4: Machine Learning Models (Choose One)")

# Bind the selectbox directly to session_state['model_choice']
model_choice = st.selectbox(
    "Select an ML Model",
    ["Linear Regression", "Logistic Regression", "K-Means Clustering"],
    key="model_choice"
)

# ---------------- Step 5: Train Model ----------------
if st.button("5️⃣ Train Model"):
    # Ensure Train/Test split has been done
    if st.session_state.X_train is None or st.session_state.y_train is None:
        st.warning("⚠️ Please complete Train/Test Split first.")
    else:
        X_train = st.session_state.X_train
        y_train = st.session_state.y_train

        if model_choice == "Linear Regression":
            model = LinearRegression()
            model.fit(X_train, y_train)

        elif model_choice == "Logistic Regression":
            # Binarize around median
            y_bin = (y_train > y_train.median()).astype(int)
            model = LogisticRegression(max_iter=200)
            model.fit(X_train, y_bin)
            st.session_state.y_train_binary = y_bin

        elif model_choice == "K-Means Clustering":
            model = KMeans(n_clusters=3, random_state=42)
            model.fit(X_train)

        else:
            st.error(f"Unknown model: {model_choice}")
            st.stop()

        st.session_state.model = model
        st.success(f"✅ {model_choice} trained successfully!")

# Step 6: Evaluation
if st.button("6️⃣ Evaluate Model"):
    model = st.session_state.model
    y_pred = model.predict(st.session_state.X_test) if isinstance(model, (LinearRegression, LogisticRegression)) else model.predict(st.session_state.X_test)
    st.session_state.y_pred = y_pred

    if isinstance(model, (LinearRegression, LogisticRegression)):
        mse = mean_squared_error(st.session_state.y_test, y_pred)
        r2 = r2_score(st.session_state.y_test, y_pred)
        st.metric("📉 Mean Squared Error", f"{mse:,.2f}")
        st.metric("📈 R-squared Score", f"{r2:.4f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=st.session_state.y_test, y=y_pred, mode='markers', name="Predicted", marker=dict(color='blue')))
        fig.add_trace(go.Scatter(x=st.session_state.y_test, y=st.session_state.y_test, mode='lines', name="Ideal Line", line=dict(color='green')))
        fig.update_layout(title="Actual vs Predicted", xaxis_title="Actual", yaxis_title="Predicted")
        st.plotly_chart(fig)
    else:
        # K-Means Clustering visualization
        st.write("K-Means Clustering Result:")
        st.write(f"Cluster Centers: {model.cluster_centers_}")
        st.write(f"Labels: {model.labels_}")

        # Visualize the clusters
        df_with_labels = st.session_state.X_test.copy()
        df_with_labels['Cluster'] = model.labels_
        fig = px.scatter(df_with_labels, x=df_with_labels.columns[0], y=df_with_labels.columns[1], color='Cluster', title="K-Means Clusters")
        st.plotly_chart(fig)

# Step 7: Download Results
if st.button("7️⃣ Show Predictions & Download"):
    results_df = pd.DataFrame({
        "Actual": st.session_state.y_test,
        "Predicted": st.session_state.y_pred
    })
    st.dataframe(results_df.head())
    st.download_button("📥 Download Results as CSV", data=results_df.to_csv(index=False), file_name="predictions.csv")
