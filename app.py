import streamlit as st
# 애플리케이션 제목
st.title("전남대 AI")
# 페이지 제목
st.header("Linear Regression")
import numpy as np                   # numpy: 데이터 분석 및 관리 라이브러리  as: 또는 -> 대체한다
import tensorflow as tf              # tensorflow: 딥러닝 라이브러리
import matplotlib.pyplot as plt      # matplotlib: 시각화 라이브러리(그래프)
from tensorflow.keras import *
import io
from contextlib import redirect_stdout

num_epochs = 100  # 초기 epoche 값 
data_np = np.loadtxt('./data.csv', dtype = float, delimiter = ',')    # loadtxt : 데이터를 불러오는 기능(txt, csv 등등등)
st.text('data_csv file -----------------')
st.write(data_np)
st.write(data_np.shape)
x = data_np[: , 0:1]  # X Value
y = data_np[: , 1:2]  # Y Value

# 두 개의 열을 생성합니다.
col1, col2 = st.columns(2)
# 첫 번째 열에 x 값을 표시합니다.
col1.header("X Values")
col1.write(x)
# 두 번째 열에 y 값을 표시합니다.
col2.header("Y Values")
col2.write(y)

fig = plt.figure(figsize = (5,2))
plt.plot(x,y)
st.pyplot(plt)

fig = plt.figure(figsize = (5,2))
plt.plot(x,y)
plt.scatter(x,y)
st.pyplot(plt)

st.write(x.shape)
st.write(x.shape[1:])

x1_input = Input(shape=(x.shape[1:]), name = 'x1_input')    # 입력층
x1_Dense_1 = layers.Dense(50, name = 'Dense_1')(x1_input)    # Dense : y = wx + b # 은닉층
x1_Dense_2 = layers.Dense(50, name = 'Dense_2')(x1_Dense_1)  # 은닉층
final = layers.Dense(1, name = 'final')(x1_Dense_2)          # 출력층
model = Model(inputs=x1_input, outputs = final)
model.compile(optimizer = 'adam', loss = 'mse')      # optimizer: 최적화 수식,  loss: 오차 수식
model.summary()

# 캡처할 출력을 위한 문자열 버퍼를 생성합니다.
buf = io.StringIO()        # 표준 출력을 버퍼로 재지향합니다.
with redirect_stdout(buf):
    model.summary()

st.text(buf.getvalue())   # 버퍼의 내용을 가져와 Streamlit에 표시합니다.
model_train = model.fit(x, y, epochs=st.sidebar.slider("epochs 횟수", min_value=0, max_value=200, value=100, step=5))

fig = plt.figure(figsize = (15,5))
plt.plot(model_train.history['loss'])
st.text("Model Train")
st.pyplot(plt)

prediction = model.predict([x])
fig = plt.figure(figsize = (15,5))
plt.plot(y, label='Actual')
plt.plot(prediction, label='Prediction')
# 범례를 추가합니다.
plt.legend()
st.text("Model Prediction")
st.pyplot(plt)

# Slide BAR 상기 model.fit 할때 eoches 값을 계산하여 그래프에 반영.
sidebar_date = st.sidebar.date_input("작성 날짜")
sidebar_time = st.sidebar.time_input("작성 시간")
fig.canvas.manager.full_screen_toggle()
