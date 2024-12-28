import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

# 1. 샘플 데이터 생성
# make_moons 데이터셋: 비선형 분포의 두 개의 반달 모양 클러스터
X, y = make_moons(n_samples=300, noise=0.1, random_state=42)

# 2. DBSCAN 모델 생성 및 학습
# epsilon=0.2, 최소 샘플 수=5
dbscan = DBSCAN(eps=0.2, min_samples=5)
clusters = dbscan.fit_predict(X)

# 3. 클러스터 결과 시각화
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='rainbow', s=30)
plt.title("DBSCAN Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.colorbar(label="Cluster ID")
plt.show()