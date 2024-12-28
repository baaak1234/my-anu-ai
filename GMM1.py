import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# 1. 데이터 생성
np.random.seed(0)
# 두 개의 서로 다른 분포에서 데이터를 샘플링
data_1 = np.random.normal(loc=0.0, scale=1.0, size=(300, 2))
data_2 = np.random.normal(loc=5.0, scale=1.0, size=(300, 2))

# 데이터 병합
data = np.vstack((data_1, data_2))

# 2. GMM 모델 생성 및 훈련
gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0)
gmm.fit(data)

# 3. 데이터 클러스터링
labels = gmm.predict(data)

# 4. 결과 시각화
plt.figure(figsize=(8, 6))

# 클러스터별 데이터 시각화
for i in range(2):
    cluster_data = data[labels == i]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f"Cluster {i}")

# GMM의 각 가우시안 중심 위치
centers = gmm.means_
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Centers')

plt.title("Gaussian Mixture Model Clustering")
plt.legend()
plt.show()
