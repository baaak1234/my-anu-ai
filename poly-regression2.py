import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. 데이터 준비
# 예제 데이터 생성 (X는 독립 변수, y는 종속 변수)
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # 0~10 범위의 값
y = 3 * X**2 + 2 * X + 1 + np.random.randn(100, 1) * 10  # 2차 방정식에 노이즈 추가

# 2. 다항 특징 생성
degree = 2  # 다항식 차수
poly_features = PolynomialFeatures(degree=degree, include_bias=False)
X_poly = poly_features.fit_transform(X)

# 3. 모델 학습
model = LinearRegression()
model.fit(X_poly, y)

# 4. 예측
X_new = np.linspace(0, 10, 100).reshape(100, 1)
X_new_poly = poly_features.transform(X_new)
y_pred = model.predict(X_new_poly)


# 5. 결과 시각화
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X_new, y_pred, color="red", label="Predicted Model")
plt.title(f"Polynomial Regression (degree={degree})")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

# 6. 모델 평가
mse = mean_squared_error(y, model.predict(X_poly))
print(f"Mean Squared Error: {mse:.2f}")
