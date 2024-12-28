import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('data.csv')

print(df)


# 2. 데이터 처리
# 필요한 데이터 전처리 작업을 수행합니다.
# 여기서는 간단하게 날짜별 데이터의 합계를 계산한다고 가정합니다.
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
monthly_data = df.resample('M').sum()  # 월별 합계 계산


# 3. 데이터 시각화
# Matplotlib을 사용하여 시각화합니다.
plt.figure(figsize=(10, 6))
plt.plot(monthly_data.index, monthly_data['value'], marker='o')
plt.title('Monthly Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.grid(True)
plt.show()