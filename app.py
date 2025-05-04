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
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
from io import BytesIO
import datetime

st.set_page_config(page_title="Finance ML Dashboard", layout="wide", page_icon="📈")

st.title("📊 ML on Financial Data – AF3005 Assignment 3")
st.image("https://media.giphy.com/media/QpVUMRUJGokfqXyfa1/giphy.gif", use_container_width=True)
st.markdown("Welcome to the **Finance ML Dashboard**!")

# ---------------- Session Initialization ----------------
for var in ['df', 'X_train', 'X_test', 'y_train', 'y_test', 'y_pred', 'model']:
    if var not in st.session_state:
        st.session_state[var] = None

# ---------------- Sidebar: Data Source ----------------
st.sidebar.title("📁 Data Options")
data_source = st.sidebar.radio("Select Data Source", ["Kragle Dataset (URL)", "Yahoo Finance"])

if data_source == "Kragle Dataset (URL)":
    url = st.sidebar.text_input("Enter Direct CSV File URL:")
    if st.sidebar.button("Load Kragle Data"):
        try:
            df = pd.read_csv(url)
            st.session_state.df = df.copy()
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
            st.session_state.df = df.copy()
            st.sidebar.success("✅ Yahoo data loaded!")
        else:
            st.sidebar.warning("⚠️ No data found.")

df = st.session_state.df

if df is not None:
    st.subheader("📄 Raw Data Preview")
    st.dataframe(df.head(), use_container_width=True)
    # ---------------- Preprocessing ----------------
    st.header("🔧 Step 1: Data Preprocessing")

    if st.button("Preprocess Data"):
        df_cleaned = df.copy()

        # Convert date to datetime if needed
        if 'Date' in df_cleaned.columns:
            df_cleaned['Date'] = pd.to_datetime(df_cleaned['Date'])

        # Drop rows with NaNs
        df_cleaned.dropna(inplace=True)

        # Remove outliers based on Z-score
        from scipy.stats import zscore
        numeric_cols = df_cleaned.select_dtypes(include=np.number).columns
        df_cleaned = df_cleaned[(np.abs(zscore(df_cleaned[numeric_cols])) < 3).all(axis=1)]

        st.session_state.df = df_cleaned
        st.success("✅ Data Preprocessing Complete!")
        st.dataframe(df_cleaned.head(), use_container_width=True)

    # ---------------- Feature Engineering ----------------
    st.header("🛠️ Step 2: Feature Engineering")

    if st.button("Generate Features"):
        df_fe = st.session_state.df.copy()
        df_fe['Daily Return'] = df_fe['Close'].pct_change()
        df_fe['MA10'] = df_fe['Close'].rolling(window=10).mean()
        df_fe['Volatility'] = df_fe['Close'].rolling(window=10).std()
        df_fe.dropna(inplace=True)

        st.session_state.df = df_fe
        st.success("✅ Feature Engineering Applied!")
        st.dataframe(df_fe.head(), use_container_width=True)

        st.plotly_chart(px.line(df_fe, x='Date' if 'Date' in df_fe.columns else df_fe.index, y='Close', title="📈 Closing Price Over Time"), use_container_width=True)
        st.plotly_chart(px.line(df_fe, x='Date' if 'Date' in df_fe.columns else df_fe.index, y='Daily Return', title="📉 Daily Return"), use_container_width=True)

    # ---------------- Train-Test Split ----------------
    st.header("✂️ Step 3: Train-Test Split")

    if st.button("Split Data"):
        df_model = st.session_state.df.copy()
        df_model.dropna(inplace=True)

        # Ensure features and target are available
        features = df_model.select_dtypes(include=np.number).drop(columns=['Close'])
        target = df_model['Close']

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test

        st.success("✅ Data split into Train/Test sets")
        st.write("Train shape:", X_train.shape)
        st.write("Test shape:", X_test.shape)
    # ---------------- ML Model Selection ----------------
    st.header("🤖 Step 4: Machine Learning Models (Choose One)")

    model_choice = st.selectbox("Select a Machine Learning Model", ["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor"])

    model = None
    if st.button("Train Model"):
        X_train = st.session_state.X_train
        X_test = st.session_state.X_test
        y_train = st.session_state.y_train
        y_test = st.session_state.y_test

        if model_choice == "Linear Regression":
            model = LinearRegression()
        elif model_choice == "Decision Tree Regressor":
            model = DecisionTreeRegressor(random_state=42)
        elif model_choice == "Random Forest Regressor":
            model = RandomForestRegressor(n_estimators=100, random_state=42)

        model.fit(X_train, y_train)
        st.session_state.model = model

        y_pred = model.predict(X_test)
        st.session_state.y_pred = y_pred

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.success("✅ Model Trained Successfully")
        st.write("📊 Mean Squared Error:", round(mse, 2))
        st.write("📈 R-squared Score:", round(r2, 4))

        # Store for plotting
        pred_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
        st.session_state.pred_df = pred_df

    # ---------------- Plot Actual vs Predicted ----------------
    st.header("📍 Step 5: Actual vs Predicted Plot")

    if 'pred_df' in st.session_state:
        pred_df = st.session_state.pred_df.copy()
        fig = px.scatter(pred_df, x="Actual", y="Predicted",
                         color_discrete_sequence=["#636EFA"],
                         title="Actual vs Predicted Closing Prices")
        fig.add_shape(type='line', line=dict(dash='dash'),
                      x0=pred_df["Actual"].min(), y0=pred_df["Actual"].min(),
                      x1=pred_df["Actual"].max(), y1=pred_df["Actual"].max())
        st.plotly_chart(fig, use_container_width=True)
    # ---------------- Show Predictions Table ----------------
    st.header("📄 Step 6: Show Predictions & Download")

    if 'pred_df' in st.session_state:
        st.subheader("Sample Predictions")
        st.dataframe(st.session_state.pred_df.head(), use_container_width=True)

        buffer = BytesIO()
        st.session_state.pred_df.to_csv(buffer, index=False)
        st.download_button("📥 Download Predictions as CSV", data=buffer.getvalue(),
                           file_name="predictions.csv", mime="text/csv")

    # ---------------- Extra Visualization ----------------
    st.header("📈 Bonus Visualizations")

    with st.expander("📊 Distribution of Target Variable (Close Price)"):
        fig_close = px.histogram(st.session_state.df, x="Close", nbins=50,
                                 title="Distribution of Closing Price", marginal="box")
        st.plotly_chart(fig_close, use_container_width=True)

    with st.expander("📉 Correlation Heatmap"):
        corr = st.session_state.df.select_dtypes(include=np.number).corr()
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Matrix")
        st.plotly_chart(fig_corr, use_container_width=True)

    with st.expander("📌 Feature Importance (Only for Tree-Based Models)"):
        if model_choice in ["Decision Tree Regressor", "Random Forest Regressor"]:
            importance_df = pd.DataFrame({
                "Feature": st.session_state.X_train.columns,
                "Importance": st.session_state.model.feature_importances_
            }).sort_values(by="Importance", ascending=False)

            fig_imp = px.bar(importance_df, x="Feature", y="Importance", title="Feature Importances")
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("ℹ️ Feature importance only available for Decision Tree / Random Forest models.")

    # ---------------- Footer ----------------
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; font-size: 16px;'>
        Built with ❤️ for AF3005 – Programming for Finance <br>
        <strong>Instructor:</strong> Dr. Usama Arshad <br>
        <strong>Student:</strong> [Your Name] | Section: BSFT06[A/B/C]
    </div>
    """, unsafe_allow_html=True)
