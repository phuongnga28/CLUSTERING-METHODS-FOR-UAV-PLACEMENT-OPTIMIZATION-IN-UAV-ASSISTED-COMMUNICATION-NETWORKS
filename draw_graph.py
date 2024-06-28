import matplotlib.pyplot as plt
import numpy as np
def plot_data_with_kmeans_centroids(cluster_data, kmeans_centroids):
    # Plotting
    plt.figure(figsize=(8, 6))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Define colors for clusters

    for i in range(len(cluster_data)):
        plt.scatter(cluster_data[i][:, 0], cluster_data[i][:, 1], c=colors[i % len(colors)], label=f'Cluster {i+1}')
        # Đánh số cho các cụm cluster
        plt.text(np.mean(cluster_data[i][:, 0]), np.mean(cluster_data[i][:, 1]), f'{i+1}', fontsize=12, color='black', weight='regular')
        plt.scatter(kmeans_centroids[:, 0], kmeans_centroids[:, 1], marker='x', s=200, c='black', label='Centroids')

    plt.title('Generated Data from Multiple Gaussian Distributions')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()

def plot_data_with_random_centroids(data, random_centroids, plane_size):
    plt.figure(figsize=(10, 10))
    plt.xlim(0, plane_size)
    plt.ylim(0, plane_size)
    plt.scatter(data[:, 0], data[:, 1], s=2, label="User's Location")
    plt.scatter(random_centroids[:, 0], random_centroids[:, 1], marker='x', s=200, c='black', label='Random UAV Position')
    plt.legend(loc='upper right', fontsize=14)
    plt.show()

def plot_density_with_range_of_random_centroids(range, data, centroids):
    plt.figure(figsize=(10, 10))
    plt.xlim(0, 8)
    plt.ylim(0, 8)
    plt.scatter(data[:, 0], data[:, 1], s=2, label="User's Location")
    # draw a red circle with radius = range around each centroid
    for i in range(len(centroids)):
        circle = plt.Circle((centroids[i][0], centroids[i][1]), range, color='r', fill=False)
        plt.gca().add_artist(circle)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, c='black', label='Centroids')
    plt.show()
    