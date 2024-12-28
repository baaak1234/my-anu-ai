import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 샘플 데이터 생성
np.random.seed(42)
data, labels = make_blobs(n_samples=500, centers=8, cluster_std=1.0, random_state=42)

# 데이터 시각화
plt.scatter(data[:, 0], data[:, 1], s=25, color='gray')
plt.title("Generated Data")
plt.show()


# K-means 알고리즘 적용
kmeans = KMeans(n_clusters=8, random_state=42)  # 클러스터 개수 4개로 설정
kmeans.fit(data)

# 클러스터 할당 및 중심 추출
cluster_labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# 결과 시각화
for i in range(8):
    cluster_points = data[cluster_labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i+1}")
    
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=400, label="Centroids")
plt.legend()
plt.title("K-means Clustering")
plt.show()