import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering

# 데이터 생성
X, _ = make_blobs(n_samples=100, centers=3, random_state=42, cluster_std=1.5)

# 덴드로그램 그리기
plt.figure(figsize=(10, 7))
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title("Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Euclidean Distance")
plt.show()

# 계층적 군집화 모델
hc = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)

# 결과 시각화
plt.scatter(X[:, 0], X[:, 1], c=y_hc, cmap='viridis', s=50)
plt.title("Hierarchical Clustering")
plt.show()
