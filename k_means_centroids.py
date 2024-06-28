from sklearn.cluster import KMeans
import numpy as np
import warnings
import matplotlib.pyplot as plt
import time
def get_kmeans_centroids(data, num_of_clusters):
    start_time = time.time()
    warnings.filterwarnings("ignore", category=FutureWarning)
    kmeans = KMeans(n_clusters=num_of_clusters, random_state=0)
    kmeans.fit(data)
    idx = kmeans.fit_predict(data)

    # Get cluster centroids
    centroids = kmeans.cluster_centers_

    data = np.array(data)

    # Getting the Clusters Associated with each centroid
    k_means_clusters = []
    for i in range(num_of_clusters):
        cluster_i = data[idx == i]
        k_means_clusters.append(cluster_i)
    
    clustering_time = time.time() - start_time
    
    return centroids, k_means_clusters, clustering_time

def visualize_kmeans(data, num_of_clusters, k_means_centroids, k_means_clusters):
    data = np.array(data)
    plt.figure(figsize=(10, 10))
    plt.xlim(0, 8)
    plt.ylim(0, 8)
    # Chia màu khác nhau cho từng cụm
    for i in range(num_of_clusters):
        plt.scatter(k_means_clusters[i][:, 0], k_means_clusters[i][:, 1], s=2)
    # plot tâm cụm bằng chữ x
    plt.scatter(k_means_centroids[:, 0], k_means_centroids[:, 1], s=50, c='black', marker='x',label='UAV')
    plt.legend(loc='upper right', fontsize=14)
    plt.show()
