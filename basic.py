import numpy as np                   # numpy: 데이터 분석 및 관리 라이브러리  as: 또는 -> 대체한다
import tensorflow as tf              # tensorflow: 딥러닝 라이브러리
import matplotlib.pyplot as plt      # matplotlib: 시각화 라이브러리(그래프)
import streamlit as st


data_np = np.loadtxt('./data.csv', dtype = float, delimiter = ',')    # loadtxt : 데이터를 불러오는 기능(txt, csv 등등등)
print(data_np)
print(data_np.shape)
x = data_np[: , 0:1]   # : -> 데이터 전체를 선택 
print(x)
y = data_np[: , 1:2]
print(y)
fig = plt.figure(figsize = (5,2))

plt.plot(x,y)

plt.show()

fig = plt.figure(figsize = (5,2))

plt.plot(x,y)
plt.scatter(x,y)

plt.show()
print(x.shape)
print(x.shape[1:])

from tensorflow.keras import *

x1_input = Input(shape=(x.shape[1:]), name = 'x1_input')    # 입력층

x1_Dense_1 = layers.Dense(50, name = 'Dense_1')(x1_input)    # Dense : y = wx + b # 은닉층
x1_Dense_2 = layers.Dense(50, name = 'Dense_2')(x1_Dense_1)  # 은닉층

final = layers.Dense(1, name = 'final')(x1_Dense_2)          # 출력층

model = Model(inputs=x1_input, outputs = final)
model.compile(optimizer = 'adam', loss = 'mse')      # optimizer: 최적화 수식,  loss: 오차 수식
model.summary()
model_train = model.fit(x, y, epochs = 300)

fig = plt.figure(figsize = (15,5))
plt.plot(model_train.history['loss'])
plt.show()

prediction = model.predict([x])
print(prediction)
fig = plt.figure(figsize = (15,5))
plt.plot(y)
plt.plot(prediction)
plt.show()
