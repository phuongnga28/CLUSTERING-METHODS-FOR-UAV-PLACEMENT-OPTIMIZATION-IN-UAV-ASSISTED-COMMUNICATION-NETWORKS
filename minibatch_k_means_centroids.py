import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
import time

def get_mini_batch_k_means_centroids(data, num_of_clusters):
    # Thực hiện phân cụm bằng MiniBatchKMeans
    start_time = time.time()
    data = np.array(data)
    mini_batch_k_means = MiniBatchKMeans(n_clusters=num_of_clusters, random_state=0, batch_size=1024, max_iter=1000, tol=0.0001)
    mini_batch_k_means.fit(data)
    idx = mini_batch_k_means.fit_predict(data)

    # Get cluster centroids
    centroids = mini_batch_k_means.cluster_centers_

    # Getting the Clusters Associated with each centroid
    mini_batch_k_means_clusters = []
    for i in range(num_of_clusters):
        cluster_i = data[idx == i]
        mini_batch_k_means_clusters.append(cluster_i)
    clustering_time = time.time() - start_time
    return centroids, mini_batch_k_means_clusters, clustering_time

def visualize_mini_batch_k_means(data, num_of_clusters, mini_batch_k_means_centroids, mini_batch_k_means_clusters):
    data = np.array(data)
    plt.figure(figsize=(10, 10))
    plt.xlim(0, 8)
    plt.ylim(0, 8)
    # Chia màu khác nhau cho từng cụm
    for i in range(num_of_clusters):
        plt.scatter(mini_batch_k_means_clusters[i][:, 0], mini_batch_k_means_clusters[i][:, 1], s=2)
    # plot tâm cụm bằng chữ x
    plt.scatter(mini_batch_k_means_centroids[:, 0], mini_batch_k_means_centroids[:, 1], s=50, c='black', marker='x', label='UAV')
    plt.legend(loc='upper right', fontsize=14)
    plt.show()