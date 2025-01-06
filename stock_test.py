import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import streamlit as st

# 페이지 설정
st.set_page_config(page_title="주식 예측", page_icon="📈", layout="wide")

# 데이터 로드
@st.cache_data
def load_data():
    # 데이터 경로는 적절히 수정해야 함
    data = pd.read_csv('dataset/data.csv')
    return data

# 데이터 전처리
@st.cache_data
def preprocess_data(data):
    # Date 컬럼을 'YYYYMMDD' 형식으로 변경 (datetime 형식으로 변환 후 형식 지정)
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce', utc=True)  # 'Date' 컬럼을 datetime 형식으로 변환, 시간대 처리
    data['Date'] = data['Date'].dt.strftime('%Y%m%d')  # 'YYYYMMDD' 형식으로 변환
    data = data.dropna()  # 결측값 제거
    X = data[['Date', 'Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits']]  # 독립 변수
    y = data['Close']  # 종속 변수
    return X, y

# 모델 학습
def train_model(X_train, y_train):
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

# 예측 함수: 날짜를 기반으로 Open, High, Low, Volume, Dividends, Stock Splits, Close 예측
def predict_all(date):
    # 데이터 로드 및 전처리
    data = load_data()
    X, y = preprocess_data(data)

    # 훈련 데이터와 테스트 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 모델 학습
    rf_model = train_model(X_train, y_train)

    # 각 변수별 예측
    date_input = np.array([[date, 0, 0, 0, 0, 0, 0]])  # 날짜만 입력 (다른 값은 0으로 초기화)
    
    # Open, High, Low, Volume, Dividends, Stock Splits 예측
    open_pred = rf_model.predict(date_input)   # 예측된 시작가
    high_pred = rf_model.predict(date_input)   # 예측된 최고가
    low_pred = rf_model.predict(date_input)    # 예측된 최저가
    volume_pred = rf_model.predict(date_input) # 예측된 거래량
    dividends_pred = rf_model.predict(date_input)  # 예측된 배당금
    stock_splits_pred = rf_model.predict(date_input)  # 예측된 주식 분할 여부

    # 위에서 예측한 값들을 기반으로 Close 예측
    all_features = np.array([[date, open_pred[0], high_pred[0], low_pred[0], volume_pred[0], dividends_pred[0], stock_splits_pred[0]]])
    
    # 종가 예측
    close_pred = rf_model.predict(all_features)  # 최종적으로 예측된 종가

    # 예측된 값들 반환
    return {
        'Open': open_pred[0],
        'High': high_pred[0],
        'Low': low_pred[0],
        'Volume': volume_pred[0],
        'Dividends': dividends_pred[0],
        'Stock Splits': stock_splits_pred[0],
        'Close': close_pred[0]
    }

# 제목
st.title("주식 예측 모델")
st.write("주식의 시작가, 최고가, 최저가, 거래량, 배당금, 주식 분할 여부를 예측하고, 이를 바탕으로 종료가(Close)를 예측합니다.")

# 날짜 입력 받기
date = st.text_input("예측할 날짜 (YYYYMMDD 형식)", "19800317")

# 예측 실행
if date:
    prediction = predict_all(date)
    st.subheader(f"{date}의 예측 결과:")
    st.write(f"Open: {prediction['Open']:.2f}")
    st.write(f"High: {prediction['High']:.2f}")
    st.write(f"Low: {prediction['Low']:.2f}")
    st.write(f"Volume: {prediction['Volume']:.2f}")
    st.write(f"Dividends: {prediction['Dividends']:.2f}")
    st.write(f"Stock Splits: {prediction['Stock Splits']:.2f}")
    st.write(f"Close: {prediction['Close']:.2f}")
