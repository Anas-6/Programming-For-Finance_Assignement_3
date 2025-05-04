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
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Finance ML Dashboard", layout="wide", page_icon="📉")

# ----------------- Initialize Session State -----------------
for key in ['df', 'features', 'target', 'X_train', 'X_test', 'y_train', 'y_test', 'model', 'y_pred']:
    if key not in st.session_state:
        st.session_state[key] = None

# ----------------- Header -----------------
st.title("📉 Machine Learning on Financial Data")
st.image("https://media.giphy.com/media/QpVUMRUJGokfqXyfa1/giphy.gif", use_container_width=True)
st.markdown("Welcome to the **AF3005 Finance ML Dashboard**. Upload Kaggle data or fetch live data from Yahoo Finance to build and evaluate ML models.")

# ----------------- Sidebar -----------------
st.sidebar.header("📁 Load Data")
data_option = st.sidebar.radio("Choose Data Source", ["Kaggle Dataset Link", "Yahoo Finance"])

if data_option == "Kaggle Dataset Link":
    kaggle_link = st.sidebar.text_input("Paste Kaggle Raw CSV File URL:")
    if st.sidebar.button("Load Kaggle Data"):
        try:
            st.session_state.df = pd.read_csv(kaggle_link)
            st.success("✅ Kaggle dataset loaded successfully!")
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
    st.markdown("---\n### 🧹 Preprocessing: Removing Missing Values and Scaling")
    df = st.session_state.df.copy()
    st.write("Missing values per column:")
    st.dataframe(df.isnull().sum())
    df.dropna(inplace=True)

    numeric_cols = df.select_dtypes(include=np.number).columns
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    st.session_state.df = df
    st.success("✅ Missing values dropped and data scaled using StandardScaler.")

# Step 2: Feature Engineering
if st.button("2️⃣ Feature Engineering"):
    st.markdown("---\n### 🏗️ Feature Engineering")
    df = st.session_state.df
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    default_features = st.session_state.features if st.session_state.features else numeric_cols[:-1]
    default_target = st.session_state.target if st.session_state.target else "Close"

    st.session_state.features = st.multiselect("Select feature columns:", options=numeric_cols, default=default_features)
    st.session_state.target = st.selectbox("Select target column:", options=numeric_cols, index=numeric_cols.index(default_target) if default_target in numeric_cols else len(numeric_cols) - 1)

    st.success(f"✅ Selected features: {st.session_state.features}")
    st.success(f"✅ Target variable: {st.session_state.target}")

# Step 3: Train/Test Split
if st.button("3️⃣ Train/Test Split"):
    st.markdown("---\n### 🧪 Splitting Data into Training & Testing Sets")
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
if st.button("4️⃣ Select Model"):

    model_choice = st.selectbox(
    "Select an ML Model",
    ["Linear Regression", "Logistic Regression", "K-Means Clustering"],
    key="model_choice"
)

# ---------------- Step 5: Train Model ----------------
if st.button("5️⃣ Train Model"):
    st.markdown("---\n### 🚀 Training the Selected Model")
    if st.session_state.X_train is None or st.session_state.y_train is None:
        st.warning("⚠️ Please complete Train/Test Split first.")
    else:
        X_train = st.session_state.X_train
        y_train = st.session_state.y_train

        if model_choice == "Linear Regression":
            model = LinearRegression()
            model.fit(X_train, y_train)

        elif model_choice == "Logistic Regression":
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
    st.markdown("---\n### 📊 Evaluating Model Performance")
    model = st.session_state.model
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    y_pred = model.predict(X_test)
    st.session_state.y_pred = y_pred

    if isinstance(model, LinearRegression):
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.metric("📉 Mean Squared Error", f"{mse:,.2f}")
        st.metric("📈 R-squared Score", f"{r2:.4f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name="Predicted", marker=dict(color='blue')))
        fig.add_trace(go.Scatter(x=y_test, y=y_test, mode='lines', name="Ideal Line", line=dict(color='green')))
        fig.update_layout(title="Actual vs Predicted", xaxis_title="Actual", yaxis_title="Predicted")
        st.plotly_chart(fig)

        coeffs = pd.DataFrame({"Feature": X_test.columns, "Coefficient": model.coef_})
        st.dataframe(coeffs)

        residuals = y_test - y_pred
        fig = px.histogram(residuals, nbins=50, title="Residuals Distribution")
        st.plotly_chart(fig)

    elif isinstance(model, LogisticRegression):
        y_true_bin = (y_test > y_test.median()).astype(int)
        acc = accuracy_score(y_true_bin, y_pred)
        st.metric("✅ Accuracy", f"{acc*100:.2f}%")
        st.text("Classification Report:")
        st.text(classification_report(y_true_bin, y_pred))

    else:
        st.write("K-Means Clustering Result:")
        st.write(f"Cluster Centers: {model.cluster_centers_}")
        st.write(f"Labels: {model.labels_}")

        df_with_labels = X_test.copy()
        df_with_labels['Cluster'] = model.predict(X_test)
        fig = px.scatter(df_with_labels, x=df_with_labels.columns[0], y=df_with_labels.columns[1], color='Cluster', title="K-Means Clusters")
        st.plotly_chart(fig)

# Step 7: Download Results
if st.button("7️⃣ Show Predictions & Download"):
    st.markdown("---\n### 💾 Show and Download Predictions")
    results_df = pd.DataFrame({
        "Actual": st.session_state.y_test,
        "Predicted": st.session_state.y_pred
    })
    st.dataframe(results_df.head())
    st.download_button("📥 Download Results as CSV", data=results_df.to_csv(index=False), file_name="predictions.csv")
