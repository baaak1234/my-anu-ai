import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# Iris 데이터셋 로드
iris = load_iris()
data = iris.data
target = iris.target

# K-Means 클러스터링
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(data)
labels = kmeans.labels_

# 결과 비교 (실제 라벨 vs K-Means 라벨)
df = pd.DataFrame({'Actual Label': target, 'KMeans Label': labels})
print(df.head(10))

# PCA를 사용하여 2차원으로 축소 중요!
pca = PCA(n_components=2)
data_2d = pca.fit_transform(data)

# 시각화
plt.figure(figsize=(8, 5))
plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='viridis', s=50, label='Clusters')
centroids_2d = pca.transform(kmeans.cluster_centers_)
plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title("K-Means Clustering on Iris Dataset")
plt.legend()
plt.show()