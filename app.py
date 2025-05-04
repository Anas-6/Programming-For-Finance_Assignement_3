# AF3005 – Programming for Finance | Assignment 3
# Student: Muhammad Anas
# Instructor: Dr. Usama Arshad
# Section: BSFT06B
# Streamlit App: ML on Financial Data

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from io import BytesIO

st.set_page_config(page_title="Finance ML App", layout="wide", page_icon="💹")

# --- Sidebar Setup ---
st.sidebar.header("📁 Load Data")
data_source = st.sidebar.radio("Choose Data Source:", ["Upload CSV", "Yahoo Finance"])

df = None
if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload Kragle Dataset (.csv)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state['df'] = df  # Storing the dataframe in session state
        st.success("✅ Kragle dataset uploaded successfully!")
        st.dataframe(df.head())
elif data_source == "Yahoo Finance":
    ticker = st.sidebar.text_input("Enter Ticker Symbol (e.g., AAPL, GOOGL)", value="AAPL")
    start_date = st.sidebar.date_input("Start Date")
    end_date = st.sidebar.date_input("End Date")
    if st.sidebar.button("Fetch Data"):
        df = yf.download(ticker, start=start_date, end=end_date)
        if not df.empty:
            df.reset_index(inplace=True)
            st.session_state['df'] = df  # Storing the dataframe in session state
            st.success("✅ Yahoo Finance data loaded!")
            st.dataframe(df.head())
        else:
            st.warning("⚠️ No data fetched. Check ticker and dates.")

# --- Step-by-Step ML Pipeline ---
if 'df' in st.session_state and st.session_state['df'] is not None:
    df = st.session_state['df']

    st.header("🔄 Step-by-Step Machine Learning Pipeline")

    # --- Step 1: Preprocessing ---
    if st.button("1️⃣ Preprocessing"):
        st.info("Running preprocessing...")

        # Handle Missing Values
        st.write("Missing values per column:")
        st.dataframe(df.isnull().sum())

        # Drop rows with missing values in essential columns (e.g., 'Close')
        df.dropna(subset=['Close'], inplace=True)

        # Remove outliers using the Interquartile Range (IQR) method
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

        st.success("✅ Data cleaned (missing values handled, outliers removed).")

        # Show cleaned data preview
        st.dataframe(df.head())

    # --- Step 2: Feature Engineering ---
    if st.button("2️⃣ Feature Engineering"):
        st.info("Performing feature engineering...")

        # Adding new features (e.g., moving averages)
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()

        # Selecting numerical features
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        # Feature Scaling
        scaler = StandardScaler()
        df_scaled = df.copy()
        df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])

        # Show transformed data preview
        st.dataframe(df_scaled.head())

        st.success("✅ Feature engineering completed (new features added, data scaled).")

    # --- Step 3: Train/Test Split ---
    if st.button("3️⃣ Train/Test Split"):
        st.info("Splitting data into training and testing sets...")

        # Select features (X) and target (y)
        X = df[['SMA_50', 'SMA_200']]  # Using moving averages as features
        y = df['Close']  # Close price as target

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        st.session_state['X_train'] = X_train
        st.session_state['X_test'] = X_test
        st.session_state['y_train'] = y_train
        st.session_state['y_test'] = y_test

        st.success("✅ Data split into train and test sets.")

    # --- Step 4: Model Selection ---
    if st.button("4️⃣ Model Selection"):
        model_choice = st.selectbox("Select Model", ["Linear Regression", "Decision Tree", "Random Forest"])
        st.session_state['model_choice'] = model_choice
        st.info(f"Selected model: {model_choice}")

    # --- Step 5: Model Training ---
    if st.button("5️⃣ Train Model"):
        model_choice = st.session_state.get('model_choice', 'Linear Regression')

        if model_choice == "Linear Regression":
            model = LinearRegression()
        elif model_choice == "Decision Tree":
            model = DecisionTreeRegressor()
        elif model_choice == "Random Forest":
            model = RandomForestRegressor()

        # Train the model
        model.fit(st.session_state['X_train'], st.session_state['y_train'])
        st.session_state['model'] = model

        st.success(f"✅ {model_choice} model trained.")

    # --- Step 6: Evaluation ---
    if st.button("6️⃣ Evaluate Model"):
        model = st.session_state.get('model', None)
        if model is None:
            st.warning("⚠️ Please train a model first.")
        else:
            y_pred = model.predict(st.session_state['X_test'])
            mse = mean_squared_error(st.session_state['y_test'], y_pred)
            r2 = r2_score(st.session_state['y_test'], y_pred)

            st.write("Mean Squared Error:", mse)
            st.write("R-squared Score:", r2)

            # Plot Actual vs Predicted
            fig = px.scatter(x=st.session_state['y_test'], y=y_pred, labels={"x": "Actual", "y": "Predicted"},
                             title="Actual vs Predicted")
            st.plotly_chart(fig)

    # --- Step 7: Show Results & Download ---
    if st.button("7️⃣ Show Results & Download"):
        model = st.session_state.get('model', None)
        if model is None:
            st.warning("⚠️ Please train a model first.")
        else:
            y_pred = model.predict(st.session_state['X_test'])
            results_df = pd.DataFrame({"Actual": st.session_state['y_test'], "Predicted": y_pred})
            st.dataframe(results_df)

            # Download predictions
            buffer = BytesIO()
            results_df.to_csv(buffer, index=False)
            st.download_button("📥 Download Predictions", data=buffer.getvalue(), file_name="predictions.csv")

else:
    st.warning("⚠️ Please load a dataset to begin.")
