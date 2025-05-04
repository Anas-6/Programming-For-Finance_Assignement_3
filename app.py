import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
from io import BytesIO

# Machine Learning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

st.image("https://media.giphy.com/media/26ufdipQqU2lhNA4g/giphy.gif", 
         caption="Let's analyze some data!", 
         use_column_width=True)

# --------------- 🎨 Enhanced Dark Theme with Custom Components ---------------
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
    color: white;
}
[data-testid="stSidebar"] {
    background-color: #1a1a1a;
    border-right: 1px solid #444;
}
h1, h2, h3, h4, h5, h6 {
    color: #ffffff;
}
.stButton>button {
    border: 1px solid #4CAF50;
    border-radius: 4px;
    background-color: #2E7D32;
    color: white;
    padding: 8px 16px;
    transition: all 0.3s;
}
.stButton>button:hover {
    background-color: #4CAF50;
    color: white;
}
.stProgress > div > div > div > div {
    background-color: #4CAF50;
}
.css-1aumxhk {
    background-color: #1a1a1a;
    border: 1px solid #444;
    border-radius: 4px;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# --------------- 📌 Advanced Session State Management ---------------
def init_session_state():
    session_defaults = {
        'df': None,
        'original_df': None,
        'models': {},
        'X_train': None,
        'X_test': None,
        'y_train': None,
        'y_test': None,
        'y_pred': None,
        'preprocessor': None,
        'feature_selector': None,
        'current_model': None,
        'metrics': {},
        'feature_importance': None,
        'preprocessing_done': False,
        'preprocessing_active': False,
        'missing_values': "Drop rows",
        'outlier_threshold': 1.5,
        'scaling_method': "None"
    }
    
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# --------------- 🎬 Enhanced Welcome Interface ---------------
st.title("🚀 Advanced Financial ML Dashboard")
st.markdown("""
    <div style="background-color: #1a1a1a; padding: 15px; border-radius: 10px; border-left: 5px solid #4CAF50;">
        <h3 style="color: #4CAF50; margin-top: 0;">Interactive Financial Analysis Platform</h3>
        <p>Upload your financial dataset and explore powerful machine learning capabilities with just a few clicks.</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar with enhanced upload and settings
# --------------- Sidebar Data Configuration ---------------
# --------------- Sidebar Data Configuration ---------------
with st.sidebar:
    st.header("📂 Data Configuration")
    
    # Data source selection
    data_source = st.radio("Data Source:", 
                         ["Upload File", "Kaggle Dataset", "Yahoo Finance"],
                         key="data_source")
    
    # File upload option (original feature)
    if data_source == "Upload File":
        uploaded_file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"],
                                       key="file_uploader")
    
    # Kaggle dataset option (new feature)
    elif data_source == "Kaggle Dataset":
        with st.expander("Kaggle API Setup", expanded=True):
            st.markdown("""
            1. Get your API key from [Kaggle Account Settings](https://www.kaggle.com/settings)
            2. Accept competition/dataset rules if required
            """)
            kaggle_username = st.text_input("Kaggle Username", key="kaggle_user")
            kaggle_key = st.text_input("Kaggle API Key", type="password", key="kaggle_key")
            dataset_url = st.text_input("Dataset URL (format: owner/dataset)", 
                                      placeholder="username/dataset-name",
                                      key="kaggle_dataset")
            
    # Yahoo Finance option (new feature)
    elif data_source == "Yahoo Finance":
        with st.expander("Stock Data Settings", expanded=True):
            ticker = st.text_input("Ticker Symbol", value="AAPL", key="yahoo_ticker")
            start_date = st.date_input("Start Date", value=pd.to_datetime('2020-01-01'), 
                                     key="yahoo_start")
            end_date = st.date_input("End Date", value=pd.to_datetime('today'),
                                   key="yahoo_end")
            interval = st.selectbox("Interval", 
                                  ["1d", "1wk", "1mo"], 
                                  key="yahoo_interval")

    # Original global settings from your version
    st.markdown("---")
    st.header("⚙️ Global Settings")
    test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2, 0.05, key="test_size")
    random_state = st.number_input("Random State", 0, 100, 42, key="random_state")
    target_col = st.text_input("Target Column (leave blank for last column)", "", key="target_col")
    
    st.markdown("---")
    st.header("🔍 Feature Selection")
    feature_selection_method = st.selectbox(
        "Method",
        ["None", "SelectKBest", "PCA"],
        key="feature_selection_method"
    )
    
    if feature_selection_method != "None":
        n_features = st.number_input("Number of Features", 1, 20, 5, key="n_features")
    
    st.markdown("---")
    st.header("📊 Visualization Settings")
    chart_theme = st.selectbox(
        "Chart Theme",
        ["plotly", "plotly_dark", "ggplot2", "seaborn"],
        key="chart_theme"
    )

# --------------- 1️⃣ Enhanced Data Loading (Combined) ---------------
def load_data():
    try:
        if st.session_state.data_source == "Upload File":
            if st.session_state.file_uploader is not None:
                if st.session_state.file_uploader.name.endswith('.csv'):
                    df = pd.read_csv(st.session_state.file_uploader)
                else:
                    df = pd.read_excel(st.session_state.file_uploader)
            else:
                st.warning("⚠️ Please upload a file")
                return None
                
        elif st.session_state.data_source == "Kaggle Dataset":
            if not all([st.session_state.kaggle_user, 
                       st.session_state.kaggle_key,
                       st.session_state.kaggle_dataset]):
                st.error("❌ Missing Kaggle credentials or dataset URL")
                return None
                
            try:
                import os
                os.environ['KAGGLE_USERNAME'] = st.session_state.kaggle_user
                os.environ['KAGGLE_KEY'] = st.session_state.kaggle_key
                from kaggle.api.kaggle_api_extended import KaggleApi
                
                api = KaggleApi()
                api.authenticate()
                with st.spinner(f"Downloading {st.session_state.kaggle_dataset}..."):
                    api.dataset_download_files(st.session_state.kaggle_dataset, unzip=True)
                    dataset_name = st.session_state.kaggle_dataset.split('/')[-1]
                    df = pd.read_csv(f"{dataset_name}.csv")
                    
            except Exception as e:
                st.error(f"❌ Kaggle Error: {str(e)}")
                return None
                
        elif st.session_state.data_source == "Yahoo Finance":
            try:
                import yfinance as yf
                with st.spinner(f"Fetching {st.session_state.yahoo_ticker} data..."):
                    df = yf.download(
                        tickers=st.session_state.yahoo_ticker,
                        start=st.session_state.yahoo_start,
                        end=st.session_state.yahoo_end,
                        interval=st.session_state.yahoo_interval
                    ).reset_index()
                    
            except Exception as e:
                st.error(f"❌ Yahoo Finance Error: {str(e)}")
                return None
                
        # Store in session state (original feature)
        st.session_state.original_df = df.copy()
        st.session_state.df = df.copy()
        st.session_state.preprocessing_done = False
        st.success(f"✅ Data loaded successfully! Shape: {df.shape}")
        
        # Original data preview features
        with st.expander("🔍 Data Preview (First 5 Rows)"):
            st.dataframe(df.head().style.set_properties(**{
                'background-color': '#1a0a2e',
                'color': '#e0aaff',
                'border-color': '#7b2cbf'
            }))
        
        # Original statistics display
        with st.expander("📊 Data Summary (Statistics)"):
            st.write(df.describe().style.format("{:.2f}").set_properties(**{
                'background-color': '#1a0a2e',
                'color': '#e0aaff',
                'border-color': '#7b2cbf'
            }))
        
        # Original data information display
        with st.expander("🧐 Data Information (dtypes/memory)"):
            buffer = BytesIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue().decode('utf-8'))
        
        # Original visualization features
        if len(df.select_dtypes(include=np.number).columns) > 1:
            sample_size = min(100, len(df))
            sample_df = df.sample(sample_size) if len(df) > 100 else df
            fig = px.scatter_matrix(
                sample_df,
                dimensions=df.select_dtypes(include=np.number).columns[:4],
                title="Scatter Matrix of First 4 Numeric Columns",
                color_discrete_sequence=['#9d4edd']
            )
            fig.update_layout({
                'plot_bgcolor': 'rgba(15, 5, 36, 1)',
                'paper_bgcolor': 'rgba(15, 5, 36, 1)',
                'font': {'color': '#e0aaff'}
            })
            st.plotly_chart(fig)
        else:
            st.warning("⚠️ Not enough numeric columns for visualization")
        
        return df
        
    except pd.errors.EmptyDataError:
        st.error("🕳️ The file is a black hole (empty)!")
        return None
    except Exception as e:
        st.error(f"💥 Cosmic radiation interference! Error: {str(e)}")
        return None

# Original load button with enhanced functionality
if st.sidebar.button("1️⃣ Load & Explore Data"):
    load_data()
# --------------- 2️⃣ Advanced Preprocessing ---------------
def update_preprocessing():
    """Callback function to update preprocessing results"""
    if st.session_state.df is None:
        return
    
    df = st.session_state.original_df.copy()
    
    # 1. Convert date columns to datetime and extract features
    date_cols = []
    for col in df.columns:
        try:
            # Try converting to datetime
            df[col] = pd.to_datetime(df[col], errors='ignore')
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                date_cols.append(col)
                # Extract date features
                df[f'{col}_year'] = df[col].dt.year
                df[f'{col}_month'] = df[col].dt.month
                df[f'{col}_day'] = df[col].dt.day
        except:
            pass
    
    # Drop original date columns
    if date_cols:
        df = df.drop(columns=date_cols)
    
    # 2. Convert remaining columns to numeric
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "").str.replace("%", ""), errors="ignore")
        except:
            pass
    
    # 3. Handle missing values
    missing_options = st.session_state.get('missing_values', "Drop rows")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if missing_options == "Drop rows":
        df = df.dropna(subset=numeric_cols)
    elif missing_options == "Drop columns":
        df = df.dropna(axis=1, subset=numeric_cols)
    elif missing_options == "Fill with mean":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif missing_options == "Fill with median":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # 4. Outlier handling
    outlier_threshold = st.session_state.get('outlier_threshold', 1.5)
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (outlier_threshold * iqr)
        upper_bound = q3 + (outlier_threshold * iqr)
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
    
    # 5. Feature scaling
    scaling_method = st.session_state.get('scaling_method', "None")
    if scaling_method == "Standard":
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        st.session_state.preprocessor = scaler
    elif scaling_method == "MinMax":
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        st.session_state.preprocessor = scaler
    
    # Store results
    st.session_state.df = df
    st.session_state.preprocessing_done = True

def show_preprocessing_ui():
    """Display preprocessing UI components"""
    if st.session_state.df is None:
        st.error("❌ Please load data first")
        return
    
    with st.expander("⚙️ Preprocessing Options", expanded=True):
        # Missing values handling
        st.radio(
            "Handle Missing Values",
            ["Drop rows", "Drop columns", "Fill with mean", "Fill with median"],
            key='missing_values',
            on_change=update_preprocessing
        )
        
        # Outlier handling
        st.slider(
            "Outlier Threshold (IQR multiplier)",
            0.0, 5.0, 1.5, 0.1,
            key='outlier_threshold',
            on_change=update_preprocessing
        )
        
        # Feature scaling
        st.selectbox(
            "Feature Scaling",
            ["None", "Standard", "MinMax"],
            key='scaling_method',
            on_change=update_preprocessing
        )
    
    # Show results
    if st.session_state.get('preprocessing_done', False):
        st.success(f"✅ Preprocessing complete. Final shape: {st.session_state.df.shape}")
        
        # Visualization
        if len(st.session_state.df.columns) > 0:
            col_to_plot = st.selectbox(
                "Select column to visualize distribution",
                st.session_state.df.columns,
                key='viz_column'
            )
            fig = px.histogram(st.session_state.df, x=col_to_plot, nbins=50, 
                             title=f"Distribution of {col_to_plot}")
            st.plotly_chart(fig)

if st.sidebar.button("2️⃣ Preprocess Data"):
    st.session_state.preprocessing_active = True

if st.session_state.get('preprocessing_active', False):
    show_preprocessing_ui()

    # --------------- 3️⃣ Feature Engineering & Selection ---------------
# --------------- 3️⃣ Feature Engineering & Selection ---------------
def convert_financial_abbreviations(value):
    """Convert financial abbreviations (K, M, B) to numeric values"""
    if isinstance(value, str):
        value = value.replace(',', '').upper()
        if 'K' in value:
            return float(value.replace('K', '')) * 1_000
        elif 'M' in value:
            return float(value.replace('M', '')) * 1_000_000
        elif 'B' in value:
            return float(value.replace('B', '')) * 1_000_000_000
        elif '%' in value:
            return float(value.replace('%', '')) / 100
    return value

def feature_engineering():
    if st.session_state.df is not None:
        df = st.session_state.df.copy()
        
        with st.expander("🔧 Feature Engineering Options"):
            # Convert financial abbreviations to numeric values
            for col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].apply(convert_financial_abbreviations)
                    try:
                        df[col] = pd.to_numeric(df[col], errors='ignore')
                    except:
                        pass
            
            # Date features (if any datetime columns)
            date_cols = [col for col in df.columns if df[col].dtype == 'datetime64[ns]']
            if date_cols:
                st.write("Date columns detected. Creating temporal features...")
                for col in date_cols:
                    df[f'{col}_year'] = df[col].dt.year
                    df[f'{col}_month'] = df[col].dt.month
                    df[f'{col}_day'] = df[col].dt.day
                    df[f'{col}_dayofweek'] = df[col].dt.dayofweek
                df = df.drop(columns=date_cols)
            
            # Interaction features
            if len(df.columns) > 1:
                num_cols = df.select_dtypes(include=[np.number]).columns
                if st.checkbox("Create interaction features"):
                    for i in range(len(num_cols)):
                        for j in range(i+1, len(num_cols)):
                            col1 = num_cols[i]
                            col2 = num_cols[j]
                            df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                            df[f'{col1}_div_{col2}'] = df[col1] / (df[col2].replace(0, np.nan))
            
            # Polynomial features
            if st.checkbox("Create polynomial features (degree 2)"):
                from sklearn.preprocessing import PolynomialFeatures
                poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                poly_features = poly.fit_transform(df[numeric_cols])
                poly_col_names = poly.get_feature_names_out(numeric_cols)
                df_poly = pd.DataFrame(poly_features, columns=poly_col_names)
                df = pd.concat([df, df_poly], axis=1)
        
        # Feature selection
        if feature_selection_method != "None":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if target_col and target_col in df.columns:
                y = df[target_col]
                X = df.drop(columns=[target_col])
            else:
                y = df.iloc[:, -1]
                X = df.iloc[:, :-1]
            
            # Ensure all features are numeric before selection
            X = X.select_dtypes(include=[np.number])
            
            if feature_selection_method == "SelectKBest":
                selector = SelectKBest(f_regression, k=n_features)
                X_selected = selector.fit_transform(X, y)
                selected_features = X.columns[selector.get_support()]
                df = pd.concat([pd.DataFrame(X_selected, columns=selected_features), y], axis=1)
                st.session_state.feature_selector = selector
                st.info(f"Selected features: {', '.join(selected_features)}")
            
            elif feature_selection_method == "PCA":
                pca = PCA(n_components=n_features)
                X_pca = pca.fit_transform(X)
                df = pd.concat([pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(n_features)]), y], axis=1)
                st.session_state.feature_selector = pca
                
                # Plot explained variance
                fig = px.bar(x=[f"PC{i+1}" for i in range(n_features)], 
                            y=pca.explained_variance_ratio_,
                            labels={'x': 'Principal Component', 'y': 'Explained Variance Ratio'},
                            title="PCA Explained Variance")
                st.plotly_chart(fig)
        
        st.session_state.df = df
        st.success(f"✅ Feature engineering complete. Final shape: {df.shape}")
        
        # Correlation matrix
        if len(df.columns) > 1:
            corr = df.corr()
            fig = go.Figure(data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                colorscale='Viridis'
            ))
            fig.update_layout(title="Feature Correlation Matrix")
            st.plotly_chart(fig)
    else:
        st.error("❌ Please load and preprocess data first")

if st.sidebar.button("3️⃣ Feature Engineering"):
    feature_engineering()

# --------------- 4️⃣ Train/Test Split with Enhanced Options ---------------
def train_test_split_data():
    if st.session_state.df is not None:
        df = st.session_state.df
        
        if len(df.columns) < 2:
            st.error("❌ Not enough columns for train/test split")
            return
        
        # Target selection
        if target_col and target_col in df.columns:
            y = df[target_col]
            X = df.drop(columns=[target_col])
        else:
            y = df.iloc[:, -1]
            X = df.iloc[:, :-1]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        
        st.success(f"✅ Data split complete (Train: {len(X_train)}, Test: {len(X_test)})")
        
        # Visualize split
        fig = px.pie(values=[len(X_train), len(X_test)], 
                     names=["Train", "Test"], 
                     title="Train-Test Split",
                     hole=0.4)
        st.plotly_chart(fig)
        
        # Show sample of training data
        with st.expander("👀 View Training Data Sample"):
            st.dataframe(pd.concat([X_train, y_train], axis=1).head())
    else:
        st.error("❌ Please complete previous steps first")

if st.sidebar.button("4️⃣ Train/Test Split"):
    train_test_split_data()

# --------------- 5️⃣ Advanced Model Training with Multiple Algorithms ---------------
# --------------- 5️⃣ Advanced Model Training with State Persistence ---------------
def update_model_params():
    """Update model parameters when selections change"""
    if 'X_train' not in st.session_state:
        return
    
    model_type = st.session_state.get('selected_model', "Linear Regression")
    
    # Initialize model with current parameters
    if model_type == "Ridge Regression":
        st.session_state.model = Ridge(alpha=st.session_state.get('ridge_alpha', 1.0))
    elif model_type == "Lasso Regression":
        st.session_state.model = Lasso(alpha=st.session_state.get('lasso_alpha', 1.0))
    elif model_type == "Random Forest":
        st.session_state.model = RandomForestRegressor(
            n_estimators=st.session_state.get('rf_n_estimators', 100),
            max_depth=st.session_state.get('rf_max_depth', 10),
            random_state=random_state
        )
    elif model_type == "Gradient Boosting":
        st.session_state.model = GradientBoostingRegressor(
            n_estimators=st.session_state.get('gb_n_estimators', 100),
            learning_rate=st.session_state.get('gb_learning_rate', 0.1),
            random_state=random_state
        )
    elif model_type == "Support Vector Machine":
        st.session_state.model = SVR(
            kernel=st.session_state.get('svm_kernel', "rbf"),
            C=st.session_state.get('svm_c', 1.0)
        )
    else:  # Linear Regression
        st.session_state.model = LinearRegression()

def train_model_ui():
    """Display model training UI with persistent state"""
    if 'X_train' not in st.session_state:
        st.error("❌ Please split the data first")
        return
    
    model_options = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(),
        "Random Forest": RandomForestRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "Support Vector Machine": SVR()
    }
    
    # Model selection with persistence
    st.selectbox(
        "Select Model",
        list(model_options.keys()),
        key='selected_model',
        on_change=update_model_params
    )
    
    # Dynamic parameter controls
    with st.expander("⚙️ Hyperparameters", expanded=True):
        if st.session_state.selected_model == "Ridge Regression":
            st.slider(
                "Alpha (Regularization)", 
                0.01, 10.0, 1.0, 0.1,
                key='ridge_alpha',
                on_change=update_model_params
            )
        elif st.session_state.selected_model == "Lasso Regression":
            st.slider(
                "Alpha (Regularization)", 
                0.01, 10.0, 1.0, 0.1,
                key='lasso_alpha',
                on_change=update_model_params
            )
        elif st.session_state.selected_model == "Random Forest":
            st.slider(
                "Number of Trees", 
                10, 500, 100, 10,
                key='rf_n_estimators',
                on_change=update_model_params
            )
            st.slider(
                "Max Depth", 
                1, 50, 10, 1,
                key='rf_max_depth',
                on_change=update_model_params
            )
        elif st.session_state.selected_model == "Gradient Boosting":
            st.slider(
                "Number of Trees", 
                10, 500, 100, 10,
                key='gb_n_estimators',
                on_change=update_model_params
            )
            st.slider(
                "Learning Rate", 
                0.01, 1.0, 0.1, 0.01,
                key='gb_learning_rate',
                on_change=update_model_params
            )
        elif st.session_state.selected_model == "Support Vector Machine":
            st.selectbox(
                "Kernel", 
                ["linear", "poly", "rbf", "sigmoid"],
                key='svm_kernel',
                on_change=update_model_params
            )
            st.slider(
                "C (Regularization)", 
                0.1, 10.0, 1.0, 0.1,
                key='svm_c',
                on_change=update_model_params
            )
    
    # Train button - now properly connected to the current model
    if st.button("🚀 Train Model"):
        if hasattr(st.session_state, 'model'):
            with st.spinner(f"Training {st.session_state.selected_model}..."):
                try:
                    model = st.session_state.model
                    model.fit(st.session_state.X_train, st.session_state.y_train)
                    st.session_state.models[st.session_state.selected_model] = model
                    st.session_state.current_model = st.session_state.selected_model
                    st.success(f"✅ {st.session_state.selected_model} trained successfully!")
                    
                    # Feature importance
                    if hasattr(model, 'feature_importances_'):
                        importance = model.feature_importances_
                        features = st.session_state.X_train.columns
                        st.session_state.feature_importance = pd.DataFrame({
                            'Feature': features,
                            'Importance': importance
                        }).sort_values('Importance', ascending=False)
                        
                        fig = px.bar(st.session_state.feature_importance,
                                    x='Feature',
                                    y='Importance',
                                    title="Feature Importance")
                        st.plotly_chart(fig)
                    elif hasattr(model, 'coef_'):
                        coef = model.coef_
                        features = st.session_state.X_train.columns
                        st.session_state.feature_importance = pd.DataFrame({
                            'Feature': features,
                            'Coefficient': coef
                        }).sort_values('Coefficient', ascending=False)
                        
                        fig = px.bar(st.session_state.feature_importance,
                                    x='Feature',
                                    y='Coefficient',
                                    title="Feature Coefficients")
                        st.plotly_chart(fig)
                except Exception as e:
                    st.error(f"❌ Error training model: {str(e)}")
        else:
            st.error("❌ No model initialized. Please select a model type first.")

# Initialize model parameters when loading data
def load_data():
    # ... [your existing load_data function] ...
    if uploaded_file is not None:
        # ... [your existing loading code] ...
        # Add these initialization lines:
        st.session_state.selected_model = "Linear Regression"
        st.session_state.ridge_alpha = 1.0
        st.session_state.lasso_alpha = 1.0
        st.session_state.rf_n_estimators = 100
        st.session_state.rf_max_depth = 10
        st.session_state.gb_n_estimators = 100
        st.session_state.gb_learning_rate = 0.1
        st.session_state.svm_kernel = "rbf"
        st.session_state.svm_c = 1.0
        update_model_params()  # Initialize the model

# Call the model training UI
if st.sidebar.button("5️⃣ Train Model"):
    st.session_state.model_training_active = True

if st.session_state.get('model_training_active', False):
    train_model_ui()

# --------------- 6️⃣ Comprehensive Model Evaluation ---------------
# --------------- 6️⃣ Comprehensive Model Evaluation (Fixed) ---------------
def evaluate_model():
    """Evaluate the currently selected model with proper state management"""
    if not st.session_state.get('current_model') or st.session_state.current_model not in st.session_state.models:
        st.error("❌ Please train a model first")
        return
    
    model = st.session_state.models[st.session_state.current_model]
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    
    try:
        # Make predictions
        y_pred = model.predict(X_test)
        st.session_state.y_pred = y_pred
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Store metrics with model name
        model_name = st.session_state.current_model
        st.session_state.metrics[model_name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
        
        # Display metrics
        st.success(f"✅ {model_name} Evaluation Complete")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mean Squared Error", f"{mse:.4f}")
        col2.metric("Root Mean Squared Error", f"{rmse:.4f}")
        col3.metric("Mean Absolute Error", f"{mae:.4f}")
        col4.metric("R² Score", f"{r2:.4f}")
        
        # Actual vs Predicted plot
        fig1 = px.scatter(x=y_test, y=y_pred,
                         labels={'x': 'Actual', 'y': 'Predicted'},
                         title=f"{model_name} - Actual vs Predicted Values",
                         trendline="ols")
        fig1.add_shape(type="line", x0=y_test.min(), y0=y_test.min(),
                      x1=y_test.max(), y1=y_test.max(),
                      line=dict(color="Red", width=2, dash="dot"))
        st.plotly_chart(fig1)
        
        # Residual plot
        residuals = y_test - y_pred
        fig2 = px.scatter(x=y_pred, y=residuals,
                         labels={'x': 'Predicted', 'y': 'Residuals'},
                         title=f"{model_name} - Residual Plot")
        fig2.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig2)
        
        # Error distribution
        fig3 = px.histogram(residuals, nbins=50,
                           title=f"{model_name} - Distribution of Residuals",
                           labels={'value': 'Residuals'})
        st.plotly_chart(fig3)
        
        # Model comparison (if multiple models exist)
        if len(st.session_state.models) > 1:
            st.markdown("### 📊 Model Comparison")
            comparison_df = pd.DataFrame.from_dict(st.session_state.metrics, orient='index')
            st.dataframe(comparison_df.style.format("{:.4f}").highlight_min(axis=0, color='#4CAF50'))
            
    except Exception as e:
        st.error(f"❌ Error evaluating model: {str(e)}")

# Initialize evaluation when training completes
def train_model():
    # ... [your existing train_model code] ...
    # After successful training, add:
    st.session_state.metrics[st.session_state.current_model] = {}  # Initialize metrics entry

# Evaluation button
if st.sidebar.button("6️⃣ Evaluate Model"):
    evaluate_model()

# --------------- 7️⃣ Prediction Interface ---------------
# --------------- 7️⃣ Prediction Interface (Fixed) ---------------
def prediction_interface():
    """Prediction interface with proper model state management"""
    if not st.session_state.get('current_model') or st.session_state.current_model not in st.session_state.models:
        st.error("❌ Please train a model first")
        return
    
    model = st.session_state.models[st.session_state.current_model]
    feature_names = st.session_state.X_train.columns
    
    st.header(f"🔮 {st.session_state.current_model} Predictions")
    
    # Option 1: Manual input
    with st.expander("✍️ Manual Input", expanded=True):
        input_data = {}
        cols = st.columns(3)
        for i, feature in enumerate(feature_names):
            with cols[i % 3]:
                # Store inputs in session state
                input_key = f"pred_input_{feature}"
                if input_key not in st.session_state:
                    st.session_state[input_key] = 0.0
                
                input_data[feature] = st.number_input(
                    feature, 
                    value=st.session_state[input_key],
                    key=input_key
                )
        
        if st.button("Predict", key="manual_predict_btn"):
            try:
                input_df = pd.DataFrame([input_data])
                prediction = model.predict(input_df)
                st.success(f"Predicted Value: {prediction[0]:.4f}")
                # Store prediction in session state
                st.session_state.last_prediction = {
                    'model': st.session_state.current_model,
                    'input': input_data,
                    'output': prediction[0]
                }
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
    
    # Option 2: CSV upload for batch prediction
    with st.expander("📁 Batch Prediction (CSV)", expanded=True):
        pred_file = st.file_uploader("Upload CSV for prediction", 
                                    type=["csv"],
                                    key="batch_pred_uploader")
        
        if pred_file is not None:
            try:
                pred_df = pd.read_csv(pred_file)
                missing_features = set(feature_names) - set(pred_df.columns)
                
                if not missing_features:
                    predictions = model.predict(pred_df[feature_names])
                    pred_df['Prediction'] = predictions
                    st.success(f"✅ {len(predictions)} predictions generated using {st.session_state.current_model}")
                    
                    # Store batch results in session state
                    st.session_state.last_batch_prediction = {
                        'model': st.session_state.current_model,
                        'data': pred_df,
                        'timestamp': datetime.now()
                    }
                    
                    st.dataframe(pred_df)
                    
                    # Download predictions
                    csv = pred_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions",
                        data=csv,
                        file_name=f"predictions_{st.session_state.current_model.replace(' ', '_')}.csv",
                        mime="text/csv",
                        key="pred_download_btn"
                    )
                else:
                    st.error(f"❌ Missing required features: {', '.join(missing_features)}")
            except Exception as e:
                st.error(f"❌ Batch prediction failed: {str(e)}")
    
    # Show last prediction if available
    if 'last_prediction' in st.session_state and st.session_state.last_prediction['model'] == st.session_state.current_model:
        with st.expander("⏱️ Last Prediction", expanded=False):
            st.json(st.session_state.last_prediction)

# Initialize prediction interface when model is trained
def train_model():
    # ... [your existing train_model code] ...
    # After successful training, add:
    st.session_state.prediction_active = True  # Activate prediction tab

# Call the prediction interface
if st.sidebar.button("7️⃣ Make Predictions"):
    st.session_state.prediction_active = True

if st.session_state.get('prediction_active', False):
    prediction_interface()

# --------------- 8️⃣ Model Management & Download ---------------
def model_management():
    st.header("💾 Model Management")
    
    if st.session_state.models:
        selected_model = st.selectbox("Select Model to Manage", list(st.session_state.models.keys()))
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Save model
            model_filename = st.text_input("Model filename", f"{selected_model.lower().replace(' ', '_')}.pkl")
            if st.button("💾 Save Model"):
                model = st.session_state.models[selected_model]
                joblib.dump(model, model_filename)
                st.success(f"Model saved as {model_filename}")
        
        with col2:
            # Load model
            uploaded_model = st.file_uploader("Upload Model (.pkl)", type=["pkl"])
            if uploaded_model is not None:
                try:
                    model = joblib.load(uploaded_model)
                    model_name = uploaded_model.name.replace(".pkl", "").replace("_", " ").title()
                    st.session_state.models[model_name] = model
                    st.success(f"Model '{model_name}' loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
    
    # Download processed data
    if st.session_state.df is not None:
        st.markdown("---")
        st.header("📥 Download Processed Data")
        csv = st.session_state.df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="processed_data.csv",
            mime="text/csv"
        )

if st.sidebar.button("8️⃣ Model Management"):
    model_management()

# --------------- 📜 Help/About Section ---------------
# --------------- 📜 Help/About Section ---------------
with st.sidebar.expander("ℹ️ Help & About", expanded=False):
    st.markdown("""
    **Advanced Financial ML Dashboard**
    
    This application provides a comprehensive platform for financial data analysis and predictive modeling.
    
    **Workflow:**
    1. Upload your financial dataset
    2. Preprocess and clean the data
    3. Engineer new features
    4. Split into train/test sets
    5. Train machine learning models
    6. Evaluate model performance
    7. Make predictions
    8. Manage and save models
    
    **Features:**
    - Multiple regression algorithms
    - Hyperparameter tuning
    - Feature selection
    - Comprehensive visualizations
    - Model persistence
    
    **Connect with the Developer:**
    - [LinkedIn Profile](https://www.linkedin.com/in/abdul-hadi-cheema-238562220/)
    - [GitHub Portfolio](https://github.com/AHC62)
    """, unsafe_allow_html=True)
