import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# 폰트지정
plt.rcParams['font.family'] = 'Malgun Gothic'

# 마이너스 부호 깨짐 지정
plt.rcParams['axes.unicode_minus'] = False

# 숫자가 지수표현식으로 나올 때 지정
pd.options.display.float_format = '{:.2f}'.format

# 페이지 설정
st.set_page_config(page_title="주식 예측", page_icon="📈", layout="wide")

# 제목
st.title("주식 예측 모델")
st.write("랜덤 포레스트 모델을 이용해 인텔 주식의 종가를 예측합니다.")

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

# 예측 및 시각화
def predict_and_visualize(rf_model, X_test, y_test, y_pred):
    # 예측 결과 시각화
    st.subheader("예측 결과 비교")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_test.index, y_test, color='blue', label='실제 값', linestyle='--', linewidth=2)
    ax.plot(y_test.index, y_pred, color='red', label='예측 값', linewidth=2)
    ax.set_title('인텔 주식 예측: 실제 값 vs 예측 값')
    ax.set_xlabel('날짜')
    ax.set_ylabel('주식 종가')
    ax.legend()
    st.pyplot(fig)  # figure 객체를 넘겨줌
    
    # 성능 평가 지표
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write(f"평균 제곱 오차 (MSE): {mse:.2f}")
    st.write(f"R² 점수: {r2:.2f}")

# 히트맵 (상관 관계 분석)
def plot_heatmap(data):
    st.subheader("상관 관계 히트맵")
    correlation_matrix = data.corr()
    plt.figure(figsize=(12, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    st.pyplot()

# 데이터 로드
st.sidebar.header("데이터 분석 옵션")
data = load_data()

# 데이터 전처리
X, y = preprocess_data(data)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 랜덤 포레스트 모델 학습
rf_model = train_model(X_train, y_train)

# 예측
y_pred = rf_model.predict(X_test)

# 예측 및 시각화
predict_and_visualize(rf_model, X_test, y_test, y_pred)

# 상관관계 히트맵
plot_heatmap(data)

# 종료 메시지
st.sidebar.write("주식 예측 및 시각화가 완료되었습니다.")
