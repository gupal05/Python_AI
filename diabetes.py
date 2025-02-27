import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# 폰트 지정
plt.rcParams['font.family'] = 'Malgun Gothic'

# 마이너스 부호 깨짐 지정
plt.rcParams['axes.unicode_minus'] = False

# 숫자가 지수표현식으로 나올 때 지정
pd.options.display.float_format = '{:.2f}'.format

# Streamlit 앱 UI
st.title('당뇨병 예측 시스템')  # 앱 제목
st.write('Glucose, BMI, Age 값을 입력하여 당뇨병 예측을 해보세요.')  # 설명 텍스트

# 데이터 로드 및 전처리 함수
def load_data(file_path='dataset/diabetes.csv'):
    # 주어진 경로에서 데이터를 읽어 DataFrame으로 반환
    data = pd.read_csv(file_path)
    return data

# 모델 학습 함수
def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)  # random_state를 고정하여 결과의 재현성 보장
    model.fit(X_train, y_train)  # 학습 데이터로 모델을 학습
    return model

# 모델 로드 함수
def load_trained_model(model_filename='diabetes_model.pkl'):
    # 저장된 모델을 불러옴
    return joblib.load(model_filename)

# 예측 함수
def predict_diabetes(model, glucose, bmi, age):
    # 입력된 특성(혈당, BMI, 나이)을 바탕으로 예측하는 함수
    input_data = np.array([[glucose, bmi, age]])  # 입력 데이터를 배열로 변환
    prediction = model.predict(input_data)[0]  # 모델을 사용하여 예측
    return prediction  # 예측 결과 반환

# 데이터 분석 및 시각화 함수
def visualize_data(data):
    # 데이터의 분포를 시각화하는 함수
    st.write("### 1. 데이터 분포")  # 제목 출력
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1행 3열의 서브플롯 생성
    
    # 혈당(Glucose) 분포
    axes[0].hist(data['Glucose'], bins=20, color='blue', edgecolor='black')  # 혈당 값의 히스토그램
    axes[0].set_title('Glucose 분포')  # 서브플롯 제목
    axes[0].set_xlabel('Glucose')  # x축 레이블
    axes[0].set_ylabel('빈도')  # y축 레이블
    
    # BMI 분포
    axes[1].hist(data['BMI'], bins=20, color='green', edgecolor='black')  # BMI 값의 히스토그램
    axes[1].set_title('BMI 분포')  # 서브플롯 제목
    axes[1].set_xlabel('BMI')  # x축 레이블
    axes[1].set_ylabel('빈도')  # y축 레이블
    
    # 나이(Age) 분포
    axes[2].hist(data['Age'], bins=20, color='red', edgecolor='black')  # 나이 값의 히스토그램
    axes[2].set_title('Age 분포')  # 서브플롯 제목
    axes[2].set_xlabel('Age')  # x축 레이블
    axes[2].set_ylabel('빈도')  # y축 레이블
    
    plt.tight_layout()  # 서브플롯 간의 간격 자동 조정
    st.pyplot(fig)  # Streamlit에서 시각화된 그래프 출력

# 상관관계 히트맵
def plot_correlation_matrix(data):
    corr = data.corr()  # 상관관계 계산
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax = ax)
    st.write("### 2. 상관관계 히트맵")
    st.pyplot(fig)

# 특성 중요도 시각화 함수
def plot_feature_importances(model, feature_names):
    feature_importances = model.feature_importances_  # 특성 중요도 추출
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)  # 중요도에 따라 정렬

    # 특성 중요도를 막대그래프로 시각화
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, ax=ax, palette='viridis')
    plt.title('특성 중요도 (Feature Importance)')
    st.write("### 3. 특성 중요도")
    st.pyplot(fig)

# Streamlit 앱 실행 함수
def run_streamlit_app():
    # 데이터 로드 및 전처리
    data = load_data()  # 데이터 로드
    selected_features = ['Glucose', 'BMI', 'Age']  # 사용할 특성 선택
    X = data[selected_features]  # 입력 변수
    y = data['Outcome']  # 출력 변수 (당뇨병 여부)

    # 학습 데이터와 테스트 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 데이터셋을 80:20으로 분리

    # 모델 학습
    model = train_model(X_train, y_train)  # 학습 데이터로 모델 학습
    
    # 예측 수행
    y_pred = model.predict(X_test)
    
    # 모델 평가
    # R2, MSE, MAE 계산
    r2 = r2_score(y_test, y_pred)  # R² (결정계수)
    mse = mean_squared_error(y_test, y_pred)  # MSE (평균 제곱 오차)
    mae = mean_absolute_error(y_test, y_pred)  # MAE (평균 절대 오차)

    # 평가 결과 출력
    st.write(f"R² (결정계수): {r2:.2f}")
    st.write(f"MSE (평균 제곱 오차): {mse:.2f}")
    st.write(f"MAE (평균 절대 오차): {mae:.2f}")

    # 정확도 출력 (분류 모델에서 정확도도 확인 가능)
    accuracy = model.score(X_test, y_test)
    st.write(f"Accuracy: {accuracy:.2f}")

    # 데이터 시각화
    visualize_data(data)  # 데이터의 분포 시각화
    plot_correlation_matrix(data)  # 특성 간 관계 히트맵 시각화

    # 특성 중요도 시각화
    plot_feature_importances(model, selected_features)  # 특성 중요도 시각화

    # 사용자 입력 받기
    st.write("### 4. 예측하기")  # 제목 출력
    glucose = st.slider('Glucose (혈당 수치)', min_value=0, max_value=200, value=100)  # 혈당 값 입력 받기
    bmi = st.slider('BMI (체질량 지수)', min_value=0.0, max_value=50.0, value=25.0, step=0.1)  # BMI 값 입력 받기
    age = st.slider('Age (나이)', min_value=0, max_value=100, value=30)  # 나이 값 입력 받기

    # 예측하기 버튼
    if st.button('예측하기'):  # 예측하기 버튼 클릭 시
        # 저장된 모델 로드
        model = load_trained_model()  # 모델 불러오기

        # 예측하기
        prediction = predict_diabetes(model, glucose, bmi, age)  # 예측 수행

        # 결과 출력
        if prediction == 1:
            st.write('예측 결과: 당뇨병 가능성이 높습니다.')  # 당뇨병 가능성이 높을 경우
        else:
            st.write('예측 결과: 당뇨병 가능성이 낮습니다.')  # 당뇨병 가능성이 낮을 경우


# 앱 실행
if __name__ == "__main__":  # 이 파일이 직접 실행될 때
    run_streamlit_app()  # Streamlit 앱 실행
