import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# 페이지 설정
st.set_page_config(page_title="🌞 Sunspot Forecast", layout="wide")
st.title("🌞 Prophet Forecast with Preprocessed Sunspot Data")

# ----------------------------------
# [1] 데이터 불러오기
# ----------------------------------
df = pd.read_csv('./sunspots_for_prophet.csv')
df['ds'] = pd.to_datetime(df['ds'])

st.subheader("📄 데이터 미리보기")
st.dataframe(df.head())

# ----------------------------------
# [2] Prophet 모델 정의 및 학습
# ----------------------------------
model = Prophet(yearly_seasonality=False, changepoint_prior_scale=0.05, seasonality_mode='additive')
model.add_seasonality(name='sunspot_cycle', period=11, fourier_order=5)
model.fit(df)

# ----------------------------------
# [3] 예측 수행
# ----------------------------------
future = model.make_future_dataframe(periods=30, freq='YE')
forecast = model.predict(future)

# ----------------------------------
# [4] 기본 시각화
# ----------------------------------
st.subheader("📈 Prophet Forecast Plot")
fig1 = model.plot(forecast)
st.pyplot(fig1)

st.subheader("📊 Forecast Components")
fig2 = model.plot_components(forecast)
st.pyplot(fig2)

# ----------------------------------
# [5] 커스텀 시각화: 실제값 vs 예측값 + 신뢰구간
# ----------------------------------
st.subheader("📉 Custom Plot: Actual vs Predicted with Prediction Intervals")
fig3, ax = plt.subplots(figsize=(14, 6))

ax.plot(df["ds"], df["y"], label='Actual', color='blue', marker='o', linestyle='-')
ax.plot(forecast["ds"], forecast["yhat"], label='Predicted', color='red', linestyle='--')
ax.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], color='pink', alpha=0.3, label='Prediction Interval')

ax.set_xlabel("Year")
ax.set_ylabel("Sun Activity")
ax.legend(loc='upper left')
ax.grid(True)

st.pyplot(fig3)

# ----------------------------------
# [6] 잔차 분석 시각화
# ----------------------------------
st.subheader("📉 Residual Analysis (예측 오차 분석)")
merged = pd.merge(df, forecast[['ds', 'yhat']], on='ds', how='inner')
merged['residual'] = merged['y'] - merged['yhat']

fig4, ax2 = plt.subplots(figsize=(14, 4))
ax2.plot(merged["ds"], merged["residual"], label='Residual', color='purple', marker='o', linestyle='-')
ax2.axhline(0, color='black', linestyle='--')

ax2.set_xlabel("Year")
ax2.set_ylabel("Residual")
ax2.legend(loc='upper right')
ax2.grid(True)

st.pyplot(fig4)

# ----------------------------------
# [7] 잔차 통계 요약 출력
# ----------------------------------
st.subheader("📌 Residual Summary Statistics")
st.write(merged["residual"].describe())
