import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. 데이터 로드 및 전처리
iris = load_iris()
X = iris.data
y = iris.target

# 이진 분류 문제를 위해 클래스 0과 1만 선택
binary_mask = y < 2  # 클래스 0과 1만 선택
X_binary = X[binary_mask]
y_binary = y[binary_mask]

# 2. 데이터 분할 (학습 데이터와 테스트 데이터)
X_train, X_test, y_train, y_test = train_test_split(X_binary, y_binary, test_size=0.2, random_state=42)

# 3. 로지스틱 회귀 모델 생성 및 학습
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. 모델 평가
y_pred = model.predict(X_test)

# 정확도 평가
accuracy = accuracy_score(y_test, y_pred)
print("정확도:", accuracy)

# 분류 보고서 및 혼동 행렬 출력
print("\n분류 보고서:")
print(classification_report(y_test, y_pred))

print("혼동 행렬:")
print(confusion_matrix(y_test, y_pred))

# 5. 새로운 데이터 예측
new_data = [[5.0, 3.4, 1.5, 0.2]]  # 꽃받침 길이/너비, 꽃잎 길이/너비
prediction = model.predict(new_data)
print("\n새 데이터의 클래스 예측:", prediction)

