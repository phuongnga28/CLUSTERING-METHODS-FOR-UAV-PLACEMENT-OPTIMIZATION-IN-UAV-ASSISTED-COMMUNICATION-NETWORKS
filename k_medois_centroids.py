import numpy as np
import matplotlib.pyplot as plt
import time

def k_medoids(data, number_of_cluster, max_iter=100):
    # Khởi tạo các tâm cụm ngẫu nhiên
    start_time = time.time()
    data = np.array(data)
    n_samples = data.shape[0]
    init_medoids = np.random.choice(n_samples, number_of_cluster, replace=False)
    medoids = data[init_medoids]

    # Gán nhãn cho các điểm dữ liệu ban đầu
    labels = np.argmin(np.linalg.norm(data[:, np.newaxis] - medoids, axis=2), axis=1)

    for _ in range(max_iter):
        # Tìm các tâm mới
        new_medoids = np.empty_like(medoids)
        for i in range(number_of_cluster):
            cluster_points = data[labels == i]
            costs = np.sum(
                np.linalg.norm(cluster_points[:, np.newaxis] - cluster_points, axis=2),
                axis=1,
            )
            new_medoids[i] = cluster_points[np.argmin(costs)]

        # Tính toán lại các nhãn
        new_labels = np.argmin(
            np.linalg.norm(data[:, np.newaxis] - new_medoids, axis=2), axis=1
        )

        # Kiểm tra xem thuật toán đã hội tụ chưa
        if np.array_equal(labels, new_labels):
            break

        # Cập nhật tâm cụm và nhãn
        medoids = new_medoids
        labels = new_labels
        # Lấy các điêm của từng cum
        k_medoids_clusters = []
        for i in range(number_of_cluster):
            cluster_i = data[labels == i]
            k_medoids_clusters.append(cluster_i)
    clustering_time = time.time() - start_time
    return medoids, k_medoids_clusters, clustering_time
    
def visualize_k_medoids(data, number_of_cluster, k_medoids_centroids, k_medoids_clusters):
    data = np.array(data)
    plt.figure(figsize=(10, 10))
    plt.xlim(0, 8)
    plt.ylim(0, 8)
    # Chia màu khác nhau cho từng cụm
    for i in range(number_of_cluster):
        plt.scatter(k_medoids_clusters[i][:, 0], k_medoids_clusters[i][:, 1], s=2)
    # plot tâm cụm bằng chữ x
    plt.scatter(k_medoids_centroids[:, 0], k_medoids_centroids[:, 1], s=50, c='black', marker='x',label='UAV')
    plt.legend(loc='upper right', fontsize=14)
    plt.show()
