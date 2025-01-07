import streamlit as st
from data_handler import fetch_data, calculate_technical_indicators
from model_handler import prepare_data, train_model, evaluate_model
from visualization import plot_predictions

st.title("Feature Engineering for Stock Prediction")

# Inputs
stock_index = st.text_input("Enter Stock Index (e.g., ^NSEI):", "^NSEI")
start_date = st.date_input("Start Date:", value="2013-02-01")
end_date = st.date_input("End Date:", value="2024-02-01")
features = st.multiselect("Select Features for Training:", 
                           ["SMA", "EMA", "RSI", "MACD", "ADX", "CCI", "ATR", "Stochastic", 
                            "Bollinger_High", "Bollinger_Low", "Momentum", "TRIX", "MFI", 
                            "VWAP", "Rolling_Mean_7", "Rolling_Std_7"], 
                           default=["SMA", "EMA", "RSI"])
target_feature = "Close"

if st.button("Predict"):
    with st.spinner("Fetching and Processing Data..."):
        data = fetch_data(start_date, end_date, stock_index)
        data = calculate_technical_indicators(data)

    with st.spinner("Preparing Data..."):
        X, y, scaler = prepare_data(data, features, target_feature)

    with st.spinner("Training Model..."):
        model, train_data, val_data, test_data = train_model(X, y)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = train_data, val_data, test_data

    with st.spinner("Evaluating Model..."):
        predictions, metrics = evaluate_model(model, X_test, y_test)

    st.subheader("Performance Metrics")
    st.write(metrics)

    st.subheader("Prediction Plot")
    plot = plot_predictions(data.index[-len(y_test):], y_test, predictions)
    st.pyplot(plot)
