import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# 1. 데이터 생성
np.random.seed(0)
data_1 = np.random.normal(loc=0.0, scale=1.0, size=(300, 2))
data_2 = np.random.normal(loc=5.0, scale=1.0, size=(300, 2))
data_3 = np.random.normal(loc=10.0, scale=1.0, size=(300, 2))
data = np.vstack((data_1, data_2, data_3))

# 2. 클러스터 개수를 결정하기 위한 BIC, AIC 계산
bic_scores = []
aic_scores = []
n_components_range = range(1, 10)  # 클러스터 개수 후보

for n_components in n_components_range:
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=0)
    gmm.fit(data)
    bic_scores.append(gmm.bic(data))  # BIC 계산
    aic_scores.append(gmm.aic(data))  # AIC 계산

# 3. 최적의 클러스터 개수 선택
optimal_bic_n = n_components_range[np.argmin(bic_scores)]
optimal_aic_n = n_components_range[np.argmin(aic_scores)]

print(f"최적의 클러스터 개수 (BIC 기준): {optimal_bic_n}")
print(f"최적의 클러스터 개수 (AIC 기준): {optimal_aic_n}")

# 4. BIC, AIC 시각화
plt.figure(figsize=(10, 6))
plt.plot(n_components_range, bic_scores, label='BIC', marker='o')
plt.plot(n_components_range, aic_scores, label='AIC', marker='s')
plt.axvline(optimal_bic_n, color='r', linestyle='--', label=f'Optimal BIC ({optimal_bic_n})')
plt.axvline(optimal_aic_n, color='g', linestyle='--', label=f'Optimal AIC ({optimal_aic_n})')

plt.title("BIC and AIC Scores for Gaussian Mixture Model")
plt.xlabel("Number of Components")
plt.ylabel("Score")
plt.legend()
plt.show()
