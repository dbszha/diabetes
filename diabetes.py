import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 데이터 불러오기
diabetes_df = pd.read_csv("data/diabetes.csv")

# 이상치 제거 (BMI, Glucose, BloodPressure, SkinThickness, Insulin)
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

for col in ["BMI", "Glucose", "BloodPressure", "SkinThickness", "Insulin"]:
    diabetes_df = remove_outliers(diabetes_df, col)

# 특성과 타겟 분리
X = diabetes_df[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
y = diabetes_df['Outcome']

# 데이터 정규화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 모델 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report_dict = classification_report(y_test, y_pred, output_dict=True)
classification_df = pd.DataFrame(report_dict).transpose()

# Streamlit UI
st.set_page_config(page_title="당뇨병 및 건강 지표 분석", layout="wide")

# 사이드바에서 메뉴 항목 선택
menu = st.sidebar.selectbox("Go to", ["Home", "데이터분석", "데이터시각화", "머신러닝보고서"])

def home():
    st.title("당뇨병 및 건강 지표 분석")
    
    # 간단한 질병 설명 추가
    st.markdown("""
    ### 당뇨병(디아베티스)란?
    당뇨병은 혈당 수치가 지속적으로 높은 상태인 만성 질환입니다. 
    당뇨병에는 1형과 2형이 있으며, 2형 당뇨병은 성인에게 가장 흔한 형태입니다.
    당뇨병은 체중 증가, 고혈압, 심장 질환 등 다양한 합병증을 초래할 수 있습니다.
    
    - **1형 당뇨병**: 주로 어린이나 청소년에게 발생하며, 췌장이 인슐린을 생산하지 못합니다.
    - **2형 당뇨병**: 성인에게 더 흔하며, 인슐린 저항이 발생하고, 혈당 조절이 어렵습니다.
    - **임신성 당뇨병**: 임신 중에 발생하며, 출산 후 대부분은 사라지지만, 이후 당뇨병 발병 위험이 높아질 수 있습니다.
    
    당뇨병은 증상이 초기에는 미미할 수 있으므로, 정기적인 혈당 검사를 통해 조기에 발견하고 관리하는 것이 중요합니다.
    """)

    # 관련 이미지 추가 (이미지는 URL 또는 로컬 파일 경로로 제공)
    st.image("diabetes.jpg", width=500)

    # 데이터 설명
    st.markdown("""
    **데이터 설명**
    - **Pregnancies**: 임신 횟수
    - **Glucose**: 포도당 수치
    - **BloodPressure**: 혈압
    - **SkinThickness**: 피부 두께
    - **Insulin**: 인슐린 수치
    - **BMI**: 체질량지수
    - **DiabetesPedigreeFunction**: 당뇨 유전 지수
    - **Age**: 나이
    - **Outcome**: 당뇨 여부 (0: 정상, 1: 당뇨)
    """)

def eda():
    st.title("데이터 시각화")
    chart_tabs = st.tabs(["Histogram", "Boxplot", "Heatmap"])
    
    with chart_tabs[0]:
        st.subheader("연령, 포도당, 혈압, BMI 분포")
        # Plotly histogram
        fig = px.histogram(diabetes_df, x="Age", nbins=20, title="Age Distribution")
        st.plotly_chart(fig)
        
        fig = px.histogram(diabetes_df, x="Glucose", nbins=20, title="Glucose Distribution")
        st.plotly_chart(fig)
        
        fig = px.histogram(diabetes_df, x="BloodPressure", nbins=20, title="Blood Pressure Distribution")
        st.plotly_chart(fig)
        
        fig = px.histogram(diabetes_df, x="BMI", nbins=20, title="BMI Distribution")
        st.plotly_chart(fig)

    with chart_tabs[1]:
        st.subheader("Boxplot for BMI and Blood Pressure")
        plot_option = st.selectbox("Select the variable to view", ["BMI", "BloodPressure"])
        
        if plot_option == "BMI":
            fig = px.box(diabetes_df, x="Outcome", y="BMI", title="BMI Distribution by Outcome")
            st.plotly_chart(fig)
        
        if plot_option == "BloodPressure":
            fig = px.box(diabetes_df, x="Outcome", y="BloodPressure", title="Blood Pressure Distribution by Outcome")
            st.plotly_chart(fig)

    with chart_tabs[2]:
        st.subheader("상관관계 히트맵")
        corr_matrix = diabetes_df.corr()
        fig = px.imshow(corr_matrix, title="Correlation Heatmap", color_continuous_scale="RdBu")
        st.plotly_chart(fig)

def model_performance():
    st.title("모델 성능 평가")
    st.write(f'### 모델 정확도: {accuracy:.2f}')
    st.subheader("Classification Report")
    st.dataframe(classification_df)

# 선택한 메뉴에 따라 화면에 다른 내용을 표시
if menu == "Home":
    home()
elif menu == "데이터분석":
    eda()
elif menu == "데이터시각화":
    eda()
elif menu == "머신러닝보고서":
    model_performance()
