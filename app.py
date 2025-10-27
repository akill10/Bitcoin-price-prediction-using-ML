import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Conv1D, Flatten, Dense, Dropout, MaxPooling1D
from datetime import datetime, timedelta

main_color = 'royalblue'
button_color = '#0074D9'
footer_color = '#31a354'  # Enhanced green
dev_color = '#181818'     # Deep black for dev footer
plt.style.use("seaborn-v0_8-poster")

st.set_page_config(page_title="Bitcoin Price Prediction", layout="wide")
st.title("💰 Bitcoin Price Prediction using Deep Networks")

# --- PAGE STATE MANAGEMENT ---
if 'show_prediction' not in st.session_state:
    st.session_state['show_prediction'] = False

# Show Home page if not predicting
if not st.session_state['show_prediction']:
    # ----- BITCOIN INFORMATION -----
    st.markdown(
        """
        <div style='font-size:1.05em; font-family:Segoe UI,Helvetica,Arial,sans-serif; color:#444; background:#fcf8e3; border-radius:7px; padding:10px 16px; border-left:6px solid #FFD700;'>
        <b>What is Bitcoin?</b> <br>
        <b>Bitcoin (BTC)</b> is the world's first decentralized cryptocurrency, launched in 2009 by an anonymous founder known as Satoshi Nakamoto.
        Bitcoin operates on a peer-to-peer blockchain network, has a fixed supply of 21 million coins, and is recognized for its volatility, security, and innovation in digital finance. It is traded 24/7, and prices are driven by global demand, regulations, market sentiment, technology adoption, and macroeconomic trends.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "<span style='color:darkorange;font-size:1.1em;font-family:Segoe UI,Helvetica,Arial,sans-serif;'><b>This app fetches live Bitcoin prices and predicts future trends using deep learning models (LSTM, CNN, Hybrid).</b></span>",
        unsafe_allow_html=True
    )
    st.sidebar.header("Settings")
    model_type = st.sidebar.selectbox(
        "Choose prediction model:",
        ["LSTM", "1D CNN", "LSTM+CNN Hybrid"])
    look_back = st.sidebar.slider("Number of days to train on:", 30, 200, 60)
    future_days = st.sidebar.slider("Days to predict ahead (up to 6 months):", 1, 180, 30)
    graph_type = st.sidebar.selectbox(
        "Prediction graph type:", [
            "Line", "Bar", "Scatter", "Histogram", "Step", "Pie", "Boxplot"
        ]
    )
    st.sidebar.markdown("### 🔎 Predict on Specific Date")
    date_input = st.sidebar.date_input(
        "Select a date for prediction",
        min_value=datetime.now().date() + timedelta(days=1),
        max_value=datetime.now().date() + timedelta(days=future_days),
        value=datetime.now().date() + timedelta(days=1)
    )

    predict_btn = st.sidebar.button(
        label="🪙",
        help="Click to run prediction for selected days",
        use_container_width=False
    )
    st.markdown(
        f"""
        <style>
            div.stButton > button:first-child {{
                background-color: {button_color};
                color: white;
                font-weight: bold;
                border-radius:8px;
                border:none;
                padding: 0.2em 0.7em;
                font-size: 1.5em;
                min-width:2.3em;
                min-height:2.3em;
            }}
        </style>
        """, unsafe_allow_html=True
    )

    if predict_btn:
        st.session_state['show_prediction'] = True
        st.session_state['model_type'] = model_type
        st.session_state['look_back'] = look_back
        st.session_state['future_days'] = future_days
        st.session_state['graph_type'] = graph_type
        st.session_state['date_input'] = date_input

# Show Prediction Page
if st.session_state['show_prediction']:
    model_type = st.session_state['model_type']
    look_back = st.session_state['look_back']
    future_days = st.session_state['future_days']
    graph_type = st.session_state['graph_type']
    date_input = st.session_state['date_input']

    st.header("📈 Bitcoin Historical Data (last 3 years)")
    data = yf.download('BTC-USD', start=(datetime.now()-timedelta(days=3*365)).strftime('%Y-%m-%d'), end=datetime.now().strftime('%Y-%m-%d'))
    st.success('Data successfully loaded!')
    st.line_chart(data['Close'])

    # Preprocessing
    st.header("🔄 Data Preprocessing")
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(np.array(data['Close']).reshape(-1,1))
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    def create_dataset(dataset, look_back=1):
        X, Y = [], []
        for i in range(len(dataset) - look_back - 1):
            X.append(dataset[i:(i+look_back), 0])
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)

    X_train, y_train = create_dataset(train_data, look_back)
    X_test, y_test = create_dataset(test_data, look_back)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    st.write(
        f"<div style='font-size:1.1em;font-family:Segoe UI,Helvetica,Arial,sans-serif;'>"
        f"Training data shape: <span style='color:orange'>{X_train.shape}</span>, "
        f"Testing data shape: <span style='color:green'>{X_test.shape}</span>"
        "</div>", 
        unsafe_allow_html=True
    )

    # Model selection
    st.header("🤖 Model Training")
    model = Sequential()
    if model_type == "LSTM":
        model.add(LSTM(24, input_shape=(look_back, 1)))
        model.add(Dropout(0.1))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(1))
        desc = "LSTM"
    elif model_type == "1D CNN":
        model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(look_back, 1)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(12, activation='relu'))
        model.add(Dense(1))
        desc = "1D CNN"
    else:  # LSTM+CNN Hybrid
        model.add(Conv1D(16, kernel_size=3, activation='relu', input_shape=(look_back, 1)))
        model.add(LSTM(16, return_sequences=False))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(1))
        desc = "LSTM+CNN Hybrid"

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=64, epochs=3, verbose=0)
    st.success(f"Model ({desc}) training complete.")

    # Predictions
    train_predict = model.predict(X_train, verbose=0)
    test_predict = model.predict(X_test, verbose=0)
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)

    train_range = range(look_back, len(train_predict) + look_back)
    test_range = range(len(train_predict) + (look_back * 2) + 1, len(scaled_data) - 1)

    # Plot Results
    st.header("📉 Actual vs Predicted Prices")
    fig1, ax1 = plt.subplots(figsize=(12,6))
    ax1.plot(data.index, scaler.inverse_transform(scaled_data), label='Actual Price', color='dimgray', linewidth=2)
    ax1.plot(data.index[train_range], train_predict, label='Train Prediction', color='royalblue', linestyle='--')
    ax1.plot(data.index[test_range], test_predict, label='Test Prediction', color='darkorange', linestyle='-.')
    ax1.set_xlabel('Date', fontsize=13)
    ax1.set_ylabel('Bitcoin Price', fontsize=13)
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.set_title(f"Actual vs Predicted Prices ({desc})", color=main_color, fontsize=15, fontweight='bold')
    ax1.legend(fontsize=12)
    plt.tight_layout()
    st.pyplot(fig1)

    # Future Forecasting
    st.header(f"🔮 {future_days}-Day Forecast")
    last_days = scaled_data[-look_back:]
    forecast_input = last_days.reshape(1, look_back, 1)
    future_pred = []
    for _ in range(future_days):
        next_pred = model.predict(forecast_input, verbose=0)[0][0]
        future_pred.append(next_pred)
        forecast_input = np.append(forecast_input[:, 1:, :], [[[next_pred]]], axis=1)
    future_pred = scaler.inverse_transform(np.array(future_pred).reshape(-1, 1)).flatten()
    future_dates = [datetime.now().date() + timedelta(days=i+1) for i in range(future_days)]
    forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_pred})
    st.dataframe(forecast_df)

    # Prediction for Specific Date
    if date_input in forecast_df["Date"].values:
        predicted_val = forecast_df.loc[forecast_df["Date"] == date_input, "Predicted Price"].values[0]
        st.markdown(
            f"<b style='color:{main_color};font-size:1.2em;font-family:Segoe UI,Helvetica,Arial,sans-serif;'>🪙 Predicted price for {date_input} is : {predicted_val:,.2f} USD ({desc} model)</b>",
            unsafe_allow_html=True
        )
    else:
        st.warning("Selected date is outside the prediction range.")

    # Enhanced User-Selectable Graph for Forecast
    st.subheader(f"Prediction Graph: {graph_type}")
    if graph_type == "Line":
        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(forecast_df["Date"], forecast_df["Predicted Price"], color=main_color, marker='o', linewidth=2)
        ax.set_title(f"Line Chart - Predicted Bitcoin Price ({desc})", color=main_color, fontname="Segoe UI", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    elif graph_type == "Bar":
        fig, ax = plt.subplots(figsize=(12,6))
        ax.bar(forecast_df["Date"], forecast_df["Predicted Price"], color=main_color, alpha=0.7)
        ax.set_title(f"Bar Chart - Predicted Bitcoin Price ({desc})", color=main_color, fontname="Segoe UI", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    elif graph_type == "Scatter":
        fig, ax = plt.subplots(figsize=(12,6))
        ax.scatter(forecast_df["Date"], forecast_df["Predicted Price"], color=main_color, s=90)
        ax.set_title(f"Scatter Plot - Predicted Bitcoin Price ({desc})", color=main_color, fontname="Segoe UI", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    elif graph_type == "Histogram":
        fig, ax = plt.subplots(figsize=(12,6))
        ax.hist(forecast_df["Predicted Price"], color=main_color, bins=min(10, future_days), alpha=0.85)
        ax.set_title(f"Histogram - Predicted Bitcoin Price ({desc})", color=main_color, fontname="Segoe UI", fontsize=14)
        ax.set_xlabel('Predicted Price')
        ax.set_ylabel('Frequency')
        ax.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
    elif graph_type == "Step":
        fig, ax = plt.subplots(figsize=(12,6))
        ax.step(forecast_df["Date"], forecast_df["Predicted Price"], color=main_color, where='mid', linewidth=2)
        ax.set_title(f"Step Chart - Predicted Bitcoin Price ({desc})", color=main_color, fontname="Segoe UI", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    elif graph_type == "Pie":
        fig, ax = plt.subplots(figsize=(12,6))
        labels = [d.strftime('%d-%b') for d in forecast_df["Date"]]
        ax.pie(forecast_df["Predicted Price"], labels=labels, autopct='%1.1f%%', colors=cm.get_cmap('tab10').colors)
        ax.set_title(f"Pie Chart - Predicted Bitcoin Price Distribution ({desc})", color=main_color, fontname="Segoe UI", fontsize=14)
        plt.tight_layout()
        st.pyplot(fig)
    elif graph_type == "Boxplot":
        fig, ax = plt.subplots(figsize=(12,6))
        ax.boxplot(forecast_df["Predicted Price"], patch_artist=True,
                   boxprops=dict(facecolor=main_color, color=main_color))
        ax.set_title(f"Boxplot - Predicted Bitcoin Price ({desc})", color=main_color, fontname="Segoe UI", fontsize=14)
        ax.set_xticklabels(['Predicted'])
        ax.set_xlabel("Prediction")
        ax.set_ylabel("Price")
        ax.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

    # Enhanced green thank you
    st.markdown(
        f"<div style='margin-top:32px; color:{footer_color}; font-weight:bold; font-size:1.18em; padding:10px 0 6px 0; letter-spacing:0.5px; font-family:Segoe UI,Helvetica,Arial,sans-serif;'>Thank you for using the Bitcoin Prediction App!</div>",
        unsafe_allow_html=True
    )
    # Developed by footer in black
    st.markdown(
        f"<div style='text-align:center; color:{dev_color}; font-weight:600; font-size:1.1em; font-family:Poppins,Segoe UI,Helvetica,Arial,sans-serif; letter-spacing:1.2px;'>Developed by AN Team</div>",
        unsafe_allow_html=True
    )
