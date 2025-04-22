import streamlit as st

# Page configuration
st.set_page_config(page_title="Stock Volatility Prediction Dashboard", layout="wide")

import joblib
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import matplotlib.pyplot as plt
import yfinance as yf

# Load trained volatility prediction model
import os
import requests

def download_file(url, local_path):
    if not os.path.exists(local_path):
        with requests.get(url, stream=True) as r:
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

# Hugging Face model URL 
model_url = "https://huggingface.co/avadar-kedavra/fyp_models/resolve/main/stack_model_rf.pkl"
model_path = "stack_model_rf.pkl"

# Download model
download_file(model_url, model_path)

# Load model
try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.stop()

# Load scaler for Volume and Daily Return
try:
    scaler = joblib.load("minmax_scaler.pkl")
except FileNotFoundError:
    st.error("❌ Error: Scaler file 'minmax_scaler.pkl' not found.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading scaler: {str(e)}")
    st.stop()

# Extract min and max values from the scaler for scaling optional inputs
volume_min, daily_return_min = scaler.data_min_
volume_max, daily_return_max = scaler.data_max_

# Define a helper function to scale raw values manually
def scale_value(raw, min_val, max_val):
    """Scale a raw value using the min-max scaling formula and clip to [0, 1]."""
    if max_val == min_val:  # Prevent division by zero
        return 0.5  # Default to midpoint if range is zero
    scaled = (raw - min_val) / (max_val - min_val)
    return np.clip(scaled, 0, 1)

# Load FinBERT tokenizer
finbert_path = "finbert_fintuned"
finbert_url = "https://huggingface.co/avadar-kedavra/fyp_models/resolve/main/finbert_fintuned"

# Download FinBERT folder (using Hugging Face API)
from transformers import BertTokenizer, BertForSequenceClassification

try:
    tokenizer = BertTokenizer.from_pretrained(finbert_url)
except Exception as e:
    st.error(f"❌ Error loading FinBERT tokenizer: {str(e)}")
    st.stop()

@st.cache_resource()
def load_finbert():
    try:
        return BertForSequenceClassification.from_pretrained(finbert_url)
    except Exception as e:
        st.error(f"❌ Error loading FinBERT model: {str(e)}")
        st.stop()


# Define footer content
footer = """
<div style="text-align: center; padding: 10px; color: #888;">
    Developed by Ong Kang Hao, Asia Pacific University of Technology & Innovation
</div>
"""

# Initialize session state for navigation
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

# Sidebar for Navigation
st.sidebar.title("Navigation")
if st.sidebar.button("🏠 Home"):
    st.session_state.page = 'Home'
if st.sidebar.button("🔮 Predict Volatility"):
    st.session_state.page = 'Predict Volatility'
if st.sidebar.button("📊 Feature Importance"):
    st.session_state.page = 'Feature Importance'
if st.sidebar.button("📈 Stock Market Trends"):
    st.session_state.page = 'Stock Market Trends'

# Mapping of sectors to encoded values
sector_mapping = {
    "Unknown": -1,
    "Basic Materials": 0,
    "Communication Services": 1,
    "Consumer Cyclical": 2,
    "Consumer Defensive": 3,
    "Energy": 4,
    "Financial Services": 5,
    "Healthcare": 6,
    "Industrials": 7,
    "Technology": 8,
    "Utilities": 9,
}

# Function to predict sentiment using FinBERT
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = finbert_model(**inputs)
    
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1)[0].tolist()
    
    predicted_class = torch.argmax(logits, dim=1).item()
    sentiment_labels = {0: "Positive", 1: "Neutral", 2: "Negative"}

    return predicted_class, sentiment_labels[predicted_class], probabilities

# Home Page
if st.session_state.page == "Home":
    st.header("📌 Welcome to the Stock Volatility Prediction Dashboard!")
    st.write("Use this tool to predict stock volatility based on market indicators, sentiment, and sector analysis.")

    # Use HTML to highlight project aim
    st.markdown("""
        <div style="background-color: #e6ffe6; padding: 15px; border-radius: 10px; border: 1px solid #2ca02c; margin: 10px 0;">
            <h3 style="color: #2ca02c;">🎯 Project Aim</h3>
            <p style="font-size: 16px; font-weight: bold; color: #333;">
                The aim of this project is to enhance investor decision-making and promote sustainable economic growth by developing a comprehensive predictive system that 
                leverages advanced sentiment analysis and historical stock data to forecast stock market volatility, ultimately supporting informed and responsible financial decisions.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    **How It Works:**  
    - 🔮 **Prediction:** Enter market parameters and a headline to forecast volatility for the next 7 days.  
    - 📈 **Feature Importance:** See which factors or models impact the final predictions.  
    - 📊 **Stock Market Trends:** View live stock data.  
    """)

    # Load gif
    st.image("homepagegif.gif", width=600)

    st.markdown(footer, unsafe_allow_html=True)  # footer

# Prediction Page
elif st.session_state.page == "Predict Volatility":
    st.header("🔮 Predict Stock Volatility")
    st.write("Enter stock market parameters and a news headline to predict volatility.")

    # Required Inputs
    st.subheader("Required Inputs")
    raw_volume = st.number_input("📌 Trading Volume (raw)", min_value=0.0, step=1.0, value=0.0)
    raw_daily_return = st.number_input("📌 Daily Return (%)", min_value=-1000.0, max_value=1000.0, value=0.0, step=0.000001, format="%.6f")
    headline = st.text_area("📰 Enter a News Headline")
    selected_sector = st.selectbox("📌 Select Sector", list(sector_mapping.keys()))

    # Optional Inputs with raw values and defaults set to midpoints
    st.subheader("Optional Inputs (raw values)")
    st.write("Enter raw values for these optional features. Defaults are set to the midpoint of 'Volume' or 'Daily Return' training ranges.")
    default_lagged_volume = (volume_min + volume_max) / 2
    default_lagged_return = (daily_return_min + daily_return_max) / 2

    lagged_volume_1_raw = st.number_input("📌 Lagged Volume 1 (raw)", min_value=0.0, value=float(default_lagged_volume), step=1.0)
    lagged_volume_2_raw = st.number_input("📌 Lagged Volume 2 (raw)", min_value=0.0, value=float(default_lagged_volume), step=1.0)
    lagged_return_1_raw = st.number_input("📌 Lagged Return 1 (%)", min_value=-1000.0, max_value=1000.0, value=0.0, step=0.000001, format="%.6f")
    lagged_return_2_raw = st.number_input("📌 Lagged Return 2 (%)", min_value=-1000.0, max_value=1000.0, value=0.0, step=0.000001, format="%.6f")
    rolling_volume_5_raw = st.number_input("📌 Rolling Volume 5 (raw)", min_value=0.0, value=float(default_lagged_volume), step=1.0)
    rolling_return_3_raw = st.number_input("📌 Rolling Return 3 (%)", min_value=-1000.0, max_value=1000.0, value=0.0, step=0.000001, format="%.6f")

    # Scale required inputs using the scaler
    try:
        scaled_inputs = scaler.transform([[raw_volume, raw_daily_return]])
        scaled_volume, scaled_daily_return = scaled_inputs[0]
    except Exception as e:
        st.error(f"❌ Error scaling required inputs: {str(e)}")
        st.stop()

    # Scale optional inputs using the custom scaling function
    scaled_lagged_volume_1 = scale_value(lagged_volume_1_raw, volume_min, volume_max)
    scaled_lagged_volume_2 = scale_value(lagged_volume_2_raw, volume_min, volume_max)
    scaled_lagged_return_1 = scale_value(lagged_return_1_raw, daily_return_min, daily_return_max)
    scaled_lagged_return_2 = scale_value(lagged_return_2_raw, daily_return_min, daily_return_max)
    scaled_rolling_volume_5 = scale_value(rolling_volume_5_raw, volume_min, volume_max)
    scaled_rolling_return_3 = scale_value(rolling_return_3_raw, daily_return_min, daily_return_max)

    # Display scaled values and training ranges
    with st.expander("Show Scaled Values and Training Range"):
        st.write(f"🔹 **Scaled Volume:** {scaled_volume:.7f} (Training range: {volume_min:.0f} to {volume_max:.0f})")
        st.write(f"🔹 **Scaled Daily Return:** {scaled_daily_return:.7f} (Training range: {daily_return_min:.2f}% to {daily_return_max:.2f}%)")
        st.write(f"🔹 **Scaled Lagged Volume 1:** {scaled_lagged_volume_1:.7f} (Scaled using Volume's range: {volume_min:.0f} to {volume_max:.0f})")
        st.write(f"🔹 **Scaled Lagged Volume 2:** {scaled_lagged_volume_2:.7f} (Scaled using Volume's range: {volume_min:.0f} to {volume_max:.0f})")
        st.write(f"🔹 **Scaled Lagged Return 1:** {scaled_lagged_return_1:.7f} (Scaled using Daily Return's range: {daily_return_min:.2f}% to {daily_return_max:.2f}%)")
        st.write(f"🔹 **Scaled Lagged Return 2:** {scaled_lagged_return_2:.7f} (Scaled using Daily Return's range: {daily_return_min:.2f}% to {daily_return_max:.2f}%)")
        st.write(f"🔹 **Scaled Rolling Volume 5:** {scaled_rolling_volume_5:.7f} (Scaled using Volume's range: {volume_min:.0f} to {volume_max:.0f})")
        st.write(f"🔹 **Scaled Rolling Return 3:** {scaled_rolling_return_3:.7f} (Scaled using Daily Return's range: {daily_return_min:.2f}% to {daily_return_max:.2f}%)")
        st.markdown("""
        **Note:** The model expects scaled values between 0 and 1. Inputs outside the training range are clipped to 0 or 1.
        Optional features are scaled based on 'Volume' or 'Daily Return' ranges.
        """)

    # Warning for extreme daily returns
    if abs(raw_daily_return) > 500:
        st.warning("⚠️ Daily Return exceeds 500%, which is rare. Please verify your input.")

    # Prediction Button
    if st.button("🚀 Predict Volatility"):
        if headline.strip() == "":
            st.warning("⚠️ Please enter a headline before predicting.")
        else:
            # Predict sentiment using FinBERT
            sentiment_encoded, sentiment_label, sentiment_probs = predict_sentiment(headline)
            st.write(f"📝 **Predicted Sentiment:** {sentiment_label} (Encoded: {sentiment_encoded})")
            st.write(f"Confidence Scores: Positive: {sentiment_probs[0]:.2f}, Neutral: {sentiment_probs[1]:.2f}, Negative: {sentiment_probs[2]:.2f}")

            # Encode sector
            sector_encoded = sector_mapping[selected_sector]
            # Compute Sentiment-Sector Interaction
            sentiment_sector = sentiment_encoded * sector_encoded

            # Prepare input for model
            input_data = np.array([[
                scaled_volume, scaled_daily_return, scaled_lagged_volume_1, scaled_lagged_volume_2,
                scaled_lagged_return_1, scaled_lagged_return_2, scaled_rolling_volume_5, scaled_rolling_return_3,
                sector_encoded, sentiment_sector
            ]])
            # Predict probability
            prediction_probs = model.predict_proba(input_data)[0]
            confidence = max(prediction_probs)
            prediction = np.argmax(prediction_probs)

            # Display results
            col1, col2 = st.columns([2, 1])
            with col1:
                if prediction == 1:
                    st.success(f"🔺 Predicted Volatility: **Increase** (Confidence: {confidence:.2f})")
                else:
                    st.warning(f"🔻 Predicted Volatility: **Decrease** (Confidence: {confidence:.2f})")
            with col2:
                fig, ax = plt.subplots(figsize=(4, 4))
                labels = ["Decrease", "Increase"]
                sizes = [prediction_probs[0], prediction_probs[1]]
                colors = ['lightcoral', 'lightgreen']
                ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
                ax.set_title("Prediction Confidence", fontsize=12)
                st.pyplot(fig)

    st.markdown(footer, unsafe_allow_html=True)  # footer

# Feature Importance Page
elif st.session_state.page == "Feature Importance":
    st.header("📊 Feature Importance Analysis")
    st.write("Explore how features and base learners influence the stacking model's predictions.")
    st.write("""
    - **Base Learners**: Importance of each input feature in XGBoost, LightGBM, and Random Forest predictions.
    - **Meta-Learner**: How much the final Random Forest relies on each base learner’s predictions.
    """)

    # Define training features
    feature_names = [
        "Scaled_Volume", "Scaled_Daily_Return", "Lagged_Volume_1", "Lagged_Volume_2",
        "Lagged_Return_1", "Lagged_Return_2", "Rolling_Volume_5", "Rolling_Return_3",
        "Sector_Encoded", "Sentiment_Sector"
    ]

    # Access stacking model components
    estimator_names = [name for name, _ in model.estimators]  # Get names from model.estimators
    base_estimators = model.estimators_  # Get estimators
    meta_learner = model.final_estimator_

    # Base learner names for meta-learner
    base_learner_names = ["XGBoost Prediction", "LightGBM Prediction", "Random Forest Prediction"]

    # Extract importances for base estimators
    base_importances = {}
    for name, est in zip(estimator_names, base_estimators):
        if hasattr(est, 'feature_importances_'):
            base_importances[name] = est.feature_importances_
        else:
            st.write(f"{name} does not have feature_importances_ attribute.")

    # Extract importances for meta-learner
    if hasattr(meta_learner, 'feature_importances_'):
        meta_importances = meta_learner.feature_importances_
    else:
        st.write("Meta-learner does not have feature_importances_ attribute.")
        meta_importances = None

    # Create tabs for each base learner and the meta-learner
    tabs = st.tabs(estimator_names + ["Meta-Learner"])

    # Plot base learners feature importances
    for i, name in enumerate(estimator_names):
        with tabs[i]:
            st.subheader(f"Feature Importances for {name.capitalize()}")
            if name in base_importances:
                importances = base_importances[name]
                sorted_idx = np.argsort(importances)[::-1]
                sorted_features = [feature_names[idx] for idx in sorted_idx]
                sorted_importances = importances[sorted_idx]

                # Assign color palette for different models
                name_lower = name.lower()
                if "xgb" in name_lower:
                    palette = "viridis"
                elif "lgb" in name_lower or "lightgbm" in name_lower:
                    palette = "plasma"  
                elif "rf" in name_lower or "random" in name_lower:
                    palette = "inferno"
                else:
                    palette = "viridis"
                
                cmap = plt.get_cmap(palette)
                colors = cmap(np.linspace(0, 1, len(sorted_importances)))

                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.barh(sorted_features, sorted_importances, color=colors)
                for bar, value in zip(bars, sorted_importances):
                    ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{value:.3f}', 
                            va='center', fontsize=10)
                ax.set_xlabel("Importance", fontsize=12)
                ax.set_title(f"Feature Importances for {name.capitalize()}", fontsize=14)
                ax.grid(axis='x', linestyle='--', alpha=0.7)
                ax.invert_yaxis()
                st.pyplot(fig)
            else:
                st.write(f"No feature importances available for {name}.")

    # Plot meta-learner's base learner importances
    with tabs[-1]:
        st.subheader("Importance of Base Learners in Meta-Learner")
        if meta_importances is not None:
            # Number of base learners: 3
            n_base_learners = len(base_learner_names)
            # Use only importances corresponding to the base learners
            if len(meta_importances) >= n_base_learners:
                base_meta_importances = meta_importances[:n_base_learners]
            else:
                st.error("Meta-learner has fewer features than base learners!")
                base_meta_importances = meta_importances

            # Sort and plot the base learner importances
            sorted_idx = np.argsort(base_meta_importances)[::-1]
            sorted_base_learners = [base_learner_names[idx] for idx in sorted_idx]
            sorted_meta_importances = base_meta_importances[sorted_idx]

            # Meta-learner plot
            meta_palette = "cividis"
            cmap_meta = plt.get_cmap(meta_palette)
            meta_colors = cmap_meta(np.linspace(0, 1, len(sorted_meta_importances)))

            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.barh(sorted_base_learners, sorted_meta_importances, color=meta_colors)
            for bar, value in zip(bars, sorted_meta_importances):
                ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{value:.3f}', 
                        va='center', fontsize=10)
            ax.set_xlabel("Importance", fontsize=12)
            ax.set_title("Importance of Base Learners in Meta-Learner", fontsize=14)
            ax.grid(axis='x', linestyle='--', alpha=0.7)
            ax.invert_yaxis()
            st.pyplot(fig)
        else:
            st.write("Meta-learner does not support feature importances.")

    # Heatmap for Sentiment-Sector Effects
    st.subheader("Effect of Sentiment and Sector on Volatility Prediction")
    st.write("This heatmap shows the predicted probability of volatility increase for different sectors and sentiments, with other features fixed at 0.5.")

    # Define baseline for the first 8 features
    baseline = [0.5] * 8
    sentiments = [0, 1, 2]
    sentiment_labels = ["Positive", "Neutral", "Negative"]
    sectors = list(sector_mapping.values())
    sector_names = list(sector_mapping.keys())
    pred_grid = np.zeros((len(sectors), len(sentiments)))

    for i, sector in enumerate(sectors):
        for j, sentiment in enumerate(sentiments):
            sentiment_sector = sentiment * sector
            # Input data excludes Sentiment_Encoded
            input_data = baseline + [sector, sentiment_sector]
            input_data = np.array([input_data])
            pred_prob = model.predict_proba(input_data)[0][1]
            pred_grid[i, j] = pred_prob

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.pcolormesh(pred_grid, cmap='RdYlGn', edgecolor='white', linewidth=0.5)
    ax.set_xticks(np.arange(len(sentiments)) + 0.5)
    ax.set_yticks(np.arange(len(sectors)) + 0.5)
    ax.set_xticklabels(sentiment_labels)
    ax.set_yticklabels(sector_names)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.colorbar(cax, label='Probability of Volatility Increase')
    for i in range(len(sectors)):
        for j in range(len(sentiments)):
            ax.text(j + 0.5, i + 0.5, f'{pred_grid[i, j]:.2f}', 
                    ha='center', va='center', color='black', fontsize=10)
    ax.set_title('Effect of Sentiment and Sector on Volatility', fontsize=14)
    plt.tight_layout()
    st.pyplot(fig)

    # Partial Dependence Plots
    st.subheader("Partial Dependence of Sentiment and Sector on Volatility")
    st.write("These plots show how the predicted probability of volatility increase changes with sentiment and sector, with other features fixed at 0.5.")

    # Baseline for the first 8 features
    baseline = [0.5] * 8

    # Mapping sentiment labels to encoded values
    sentiment_mapping = {"Positive": 0, "Neutral": 1, "Negative": 2}
    sentiment_labels = list(sentiment_mapping.keys())

    # Effect of Sentiment for a chosen Sector
    st.write("### Effect of Sentiment for a Selected Sector")
    chosen_sector_name = st.selectbox(
        "Select a sector to see how different sentiments affect volatility:",
        list(sector_mapping.keys()),
        index=5  # Default to index 5, "Financial Services"
    )
    chosen_sector_encoded = sector_mapping[chosen_sector_name]
    st.write(f"**Selected Sector:** {chosen_sector_name} (Encoded: {chosen_sector_encoded})")

    sentiment_probs = []
    for sent_label, sent_val in sentiment_mapping.items():
        sentiment_sector = sent_val * chosen_sector_encoded
        input_data = baseline + [chosen_sector_encoded, sentiment_sector]
        input_data = np.array([input_data])
        pred_prob = model.predict_proba(input_data)[0][1]  # Probability of volatility increase
        sentiment_probs.append(pred_prob)

    # Plot bar chart of sentiment probabilities
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = plt.get_cmap("Set2")(np.linspace(0, 1, len(sentiment_probs)))
    for i, (label, prob) in enumerate(zip(sentiment_labels, sentiment_probs)):
        ax.bar(label, prob, color=colors[i], alpha=0.8)

    ax.set_xlabel("Sentiment", fontsize=12)
    ax.set_ylabel("Probability of Volatility Increase", fontsize=12)
    ax.set_title(f"Effect of Sentiment on Volatility (Sector: {chosen_sector_name})", fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig)

    # Effect of Sector for a chosen Sentiment class
    st.write("### Effect of Sector for a Selected Sentiment")
    chosen_sentiment_label = st.selectbox(
        "Select a sentiment to see how different sectors affect volatility:",
        sentiment_labels,
        index=1  # Default to "Neutral"
    )
    chosen_sentiment_encoded = sentiment_mapping[chosen_sentiment_label]
    st.write(f"**Selected Sentiment:** {chosen_sentiment_label} (Encoded: {chosen_sentiment_encoded})")

    sector_values = list(sector_mapping.values())
    sector_names = list(sector_mapping.keys())
    sector_probs = []

    for sector_name, sector_val in sector_mapping.items():
        sentiment_sector = chosen_sentiment_encoded * sector_val
        input_data = baseline + [sector_val, sentiment_sector]
        input_data = np.array([input_data])
        pred_prob = model.predict_proba(input_data)[0][1]
        sector_probs.append(pred_prob)

    # Plot bar chart of sector probabilities
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.get_cmap("Set2")(np.linspace(0, 1, len(sector_probs)))
    for i, (name, prob) in enumerate(zip(sector_names, sector_probs)):
        ax.bar(name, prob, color=colors[i], alpha=0.8)

    ax.set_xlabel("Sector", fontsize=12)
    ax.set_ylabel("Probability of Volatility Increase", fontsize=12)
    ax.set_title(f"Effect of Sector on Volatility (Sentiment: {chosen_sentiment_label})", fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

    st.markdown(footer, unsafe_allow_html=True)  # footer

# Stock Market Trends Page
elif st.session_state.page == "Stock Market Trends":
    # use plotly for interactive charts
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots 

    @st.cache_data(ttl=600) # time to live: 10 minutes, cache the searched data, avoid repeatedly fetching data
    def fetch_stock_data(ticker, period):
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            info = stock.info
            if hist.empty:
                return f"No data found for {ticker.upper()} in the selected period.", None
            return hist, info
        except Exception as e:
            return str(e), None

    # Function to calculate RSI
    def calculate_rsi(data, periods=14):
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Function to calculate MACD
    def calculate_macd(data, slow=26, fast=12, signal=9):
        exp1 = data['Close'].ewm(span=fast, adjust=False).mean()
        exp2 = data['Close'].ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line

    # Function to calculate Bollinger Bands 
    def calculate_bollinger_bands(data, window=20, num_std=2):
        rolling_mean = data['Close'].rolling(window=window).mean()
        rolling_std = data['Close'].rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return rolling_mean, upper_band, lower_band

    st.title("📈 Stock Market Trends")

    # Define periods
    period_options = {
        "1 Day": "1d", "5 Days": "5d", "1 Month": "1mo", "3 Months": "3mo", "6 Months": "6mo",
        "1 Year": "1y", "2 Years": "2y", "5 Years": "5y", "10 Years": "10y", "Year to Date": "ytd", "Maximum": "max"
    }
    selected_period = st.selectbox("Select period", list(period_options.keys()), index=2)
    period = period_options[selected_period]

    # Text input
    ticker = st.text_input("🔍 Enter a stock ticker (e.g., AAPL, TSLA, MSFT):")
    if st.button("Fetch Stock Data"):
        if ticker:
            # Fetch data from yfinance
            data, info = fetch_stock_data(ticker, period)
            # Handle errors
            if isinstance(data, str):
                if "No data found" in data or "symbol may be delisted" in data:
                    st.error(
                        f"❌ Ticker '{ticker.upper()}' not found or has no data for the selected period. "
                        f"Please check the symbol and try again. You can search for valid tickers on "
                        f"[Yahoo Finance](https://finance.yahoo.com/)."
                    )
                else:
                    st.error(f"❌ An error occurred: {data}")
            else:
                # Company Overview
                st.subheader(f"🏢 Company Overview: {info.get('longName', ticker.upper())}")
                st.markdown(f"""
                **Sector:** {info.get('sector', 'N/A')}  
                **Industry:** {info.get('industry', 'N/A')}  
                **Description:** {info.get('longBusinessSummary', 'No description available.')}
                """)

                # Key Metrics
                st.subheader("📊 Key Metrics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    yesterday_close = data['Close'].iloc[-1] if not data.empty else "N/A"
                    st.metric("Yesterday's Close", f"${yesterday_close:.2f}" if yesterday_close != "N/A" else "N/A")
                with col2:
                    yesterday_volume = data['Volume'].iloc[-1] if not data.empty else "N/A"
                    st.metric("Yesterday's Volume", f"{int(yesterday_volume):,}" if yesterday_volume != "N/A" else "N/A")
                with col3:
                    market_cap = info.get('marketCap', 'N/A')
                    st.metric("Market Cap", f"${market_cap:,}" if market_cap != "N/A" else "N/A")

                col4, col5, col6 = st.columns(3)
                with col4:
                    pe_ratio = info.get('trailingPE', 'N/A')
                    st.metric("P/E Ratio", f"{pe_ratio:.2f}" if pe_ratio != "N/A" else "N/A")
                with col5:
                    week_high = info.get('fiftyTwoWeekHigh', 'N/A')
                    st.metric("52-Week High", f"${week_high:.2f}" if week_high != "N/A" else "N/A")
                with col6:
                    week_low = info.get('fiftyTwoWeekLow', 'N/A')
                    st.metric("52-Week Low", f"${week_low:.2f}" if week_low != "N/A" else "N/A")

                # Charts and Indicators in Tabs
                st.subheader("📉 Charts and Indicators")
                tabs = st.tabs(["Price Chart", "Volume", "RSI", "MACD"])

                # Tab 1: Price Chart with Moving Averages and Bollinger Bands with Plotly
                with tabs[0]:
                    st.write(f"**Stock Price with Moving Averages and Bollinger Bands ({selected_period})**")
                    # Calculate indicators
                    ma50 = data['Close'].rolling(window=50).mean()
                    ma200 = data['Close'].rolling(window=200).mean()
                    rolling_mean, upper_band, lower_band = calculate_bollinger_bands(data)

                    # Create Plotly figure
                    fig = go.Figure()

                    # Add Close Price
                    fig.add_trace(go.Scatter(
                        x=data.index, y=data['Close'], mode='lines', name='Close Price', line=dict(color='blue')
                    ))

                    # Add 50-day MA
                    fig.add_trace(go.Scatter(
                        x=data.index, y=ma50, mode='lines', name='50-Day MA', line=dict(color='orange')
                    ))

                    # Add 200-day MA
                    fig.add_trace(go.Scatter(
                        x=data.index, y=ma200, mode='lines', name='200-Day MA', line=dict(color='green')
                    ))

                    # Add Bollinger Bands (middle, upper, and lower)
                    fig.add_trace(go.Scatter(
                        x=data.index, y=rolling_mean, mode='lines', name='20-Day MA (Bollinger)',
                        line=dict(color='purple', dash='dash')
                    ))
                    fig.add_trace(go.Scatter(
                        x=data.index, y=upper_band, mode='lines', name='Upper Bollinger Band',
                        line=dict(color='red', dash='dash'), opacity=0.5
                    ))
                    fig.add_trace(go.Scatter(
                        x=data.index, y=lower_band, mode='lines', name='Lower Bollinger Band',
                        line=dict(color='red', dash='dash'), opacity=0.5,
                        fill='tonexty', fillcolor='rgba(255, 0, 0, 0.1)'
                    ))

                    # Update layout
                    fig.update_layout(
                        title=f"{ticker.upper()} Price with Indicators",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        hovermode="x unified",
                        showlegend=True,
                        template="plotly",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Tab 2: Volume Chart 
                with tabs[1]:
                    st.write(f"**Trading Volume ({selected_period})**")
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=data.index, y=data['Volume'], name='Volume', marker_color='gray', opacity=0.6
                    ))
                    fig.update_layout(
                        title=f"{ticker.upper()} Volume",
                        xaxis_title="Date",
                        yaxis_title="Volume",
                        hovermode="x unified",
                        showlegend=True,
                        template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Tab 3: RSI
                with tabs[2]:
                    st.write(f"**Relative Strength Index (RSI, 14-Day) ({selected_period})**")
                    rsi = calculate_rsi(data)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=data.index, y=rsi, mode='lines', name='RSI', line=dict(color='purple')
                    ))
                    # overbought/oversold lines
                    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, annotation_text="Overbought (70)")
                    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, annotation_text="Oversold (30)")
                    fig.update_layout(
                        title=f"{ticker.upper()} RSI",
                        xaxis_title="Date",
                        yaxis_title="RSI",
                        hovermode="x unified",
                        showlegend=True,
                        template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Tab 4: MACD 
                with tabs[3]:
                    st.write(f"**MACD and Signal Line ({selected_period})**")
                    macd, signal_line = calculate_macd(data)

                    # Subplot with MACD & Signal on top, Histogram on bottom
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                                        subplot_titles=("MACD", "Histogram"), row_heights=[0.7, 0.3])

                    # Add MACD and Signal Line
                    fig.add_trace(go.Scatter(
                        x=data.index, y=macd, mode='lines', name='MACD', line=dict(color='blue')
                    ), row=1, col=1)
                    fig.add_trace(go.Scatter(
                        x=data.index, y=signal_line, mode='lines', name='Signal Line', line=dict(color='orange')
                    ), row=1, col=1)

                    # Add Histogram
                    histogram = macd - signal_line
                    fig.add_trace(go.Bar(
                        x=data.index, y=histogram, name='Histogram', marker_color='gray', opacity=0.3
                    ), row=2, col=1)

                    # Update layout
                    fig.update_layout(
                        title=f"{ticker.upper()} MACD",
                        xaxis2_title="Date",
                        yaxis_title="MACD",
                        yaxis2_title="Histogram",
                        hovermode="x unified",
                        showlegend=True,
                        template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Recent Trading Data
                st.subheader("📋 Recent Trading Data")
                st.dataframe(data.tail(5))


        # No ticker entered
        else:
            st.warning("⚠️ Please enter a stock ticker.")
    
    st.markdown(footer, unsafe_allow_html=True)  # footer
