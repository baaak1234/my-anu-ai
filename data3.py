import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('employee_data.csv')

print(df.head())
print("=====")
print(df.info())
print("=====")
print(df.describe())


# 특정 열 선택
names = df['이름']
print(names.head())

# 특정 행 선택 (인덱스로) 0이 첫번째 행.
first_row = df.iloc[0]

# 조건을 이용한 필터링
older_than_30 = df[df['나이'] > 30]
print(older_than_30.head())

# 부서별로 그룹화하여 나이의 평균 계산
grouped_df = df.groupby('부서')['나이'].mean()
print(grouped_df)

# 결측치 확인
print(df.isnull().sum());

# 결측치 채우기 fillna 함수 이용(예: 나이의 결측치를 평균 나이로 채우기)
df['나이'].fillna(df['나이'].mean(), inplace=True)


# # 결측치가 있는 행 제거
# df_dropped = df.dropna()

# 새로운 열 추가
df['연령대'] = df['나이'].apply(lambda x: '30대' if 30 <= x < 40 else '30대 이하' if x < 30 else '40대 이상')
print(df.head())

# 수정된 데이터를 다시 CSV 파일로 저장
df.to_csv('/Users/Admin/Desktop/my_code/cursor/modified_employee_data.csv', index=False)
