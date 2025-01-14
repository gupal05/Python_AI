import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
from datetime import datetime
from tensorflow.keras.models import load_model
import os

# Streamlit 페이지 설정
st.set_page_config(page_title="Stock Price Prediction", layout="wide")

# 1. 데이터 로드 및 전처리
st.title("Stock Price Prediction App")

# 데이터 로드 - CSV 파일로 변경
file_path = "dataset/data.csv"  # CSV 파일 경로
data = pd.read_csv(file_path, usecols=['Date', 'Close'])
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# 스케일링
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 데이터셋 생성 함수
def create_dataset(dataset, look_back=60):
    X, y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:i + look_back, 0])
        y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(y)

look_back = 60
X, y = create_dataset(scaled_data, look_back)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# 데이터 분리
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 모델 캐싱 및 학습
@st.cache_resource
def train_model(X_train, y_train, X_test, y_test):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
    model.save('stock_price_model.h5')
    return model

# 모델이 존재하면 로드, 없으면 학습
if os.path.exists('stock_price_model.h5'):
    model = load_model('stock_price_model.h5')
    st.write("Loaded pre-trained model.")
else:
    model = train_model(X_train, y_train, X_test, y_test)
    st.write("Training completed and model saved.")

# 모델 평가
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 3. Streamlit UI 구성
st.subheader("Stock Price Prediction")

# 예측 성능 출력
st.write("### Model Performance Metrics:")
st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")
st.write(f"**R² Score:** {r2:.4f}")

# 날짜 입력받기
input_date = st.date_input(
    "Enter a date for prediction (up to 30 days from now):",
    min_value=datetime(2024, 1, 2),
    max_value=datetime(2025, 12, 31)
)

# 4. 예측 기능 추가
if st.button("Predict"):
    scaled_recent_data = scaled_data[-look_back:]
    X_predict = np.array([scaled_recent_data[:, 0]])
    X_predict = np.reshape(X_predict, (X_predict.shape[0], X_predict.shape[1], 1))

    # 예측 실행
    prediction = model.predict(X_predict)
    predicted_price = scaler.inverse_transform(prediction)

    # 결과 출력
    st.write(f"### Predicted price for {input_date}: ${predicted_price[0][0]:.2f}")

# 5. 시각화
st.write("### Actual vs Predicted Prices")
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_rescaled = scaler.inverse_transform(y_pred)

plt.figure(figsize=(14, 5))
plt.plot(y_test_rescaled, label="Actual Prices", color='blue')
plt.plot(y_pred_rescaled, label="Predicted Prices", color='red')
plt.title("Stock Price Prediction")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()

st.pyplot(plt)
