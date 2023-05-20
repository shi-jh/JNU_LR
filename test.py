import micropip
import asyncio

async def install_libraries():
    await micropip.install('numpy') # NumPy 라이브러리 다운로드
    await micropip.install('pandas') # pandas 라이브러리 다운로드

async def main():
    await install_libraries()
    
    import numpy as np                   # numpy: 데이터 분석 및 관리 라이브러리  as: 또는 -> 대체한다
    import tensorflow as tf              # tensorflow: 딥러닝 라이브러리
    import matplotlib.pyplot as plt      # matplotlib: 시각화 라이브러리(그래프)
    import pandas as pd

# 데이터 불러오기
    data = pd.read_csv('data.csv')

# 입력 데이터와 출력 데이터로 나누기
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

# 입력 데이터와 출력 데이터의 평균 구하기
    X_mean = np.mean(X, axis=0)
    y_mean = np.mean(y)

# 입력 데이터와 출력 데이터의 편차 구하기
    X_dev = X - X_mean
    y_dev = y - y_mean

# 회귀 계수 구하기
    beta = np.dot(np.linalg.inv(np.dot(X_dev.T, X_dev)), np.dot(X_dev.T, y_dev))

# 절편 구하기
    intercept = y_mean - np.dot(X_mean, beta)

# 학습 결과 출력
    print('회귀계수: ', beta)
    print('절편: ', intercept)

#

loop = asyncio.get_event_loop()
loop.run_until_complete(main())


