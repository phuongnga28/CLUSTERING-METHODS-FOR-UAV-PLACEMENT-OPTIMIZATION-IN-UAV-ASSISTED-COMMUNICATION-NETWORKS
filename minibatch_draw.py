import numpy as np
import matplotlib.pyplot as plt
import time

def initialize_centroids(X, k):
    return X[np.random.choice(X.shape[0], k, replace=False)]

def nearest_centroid(C, x):
    distances = np.linalg.norm(C - x, axis=1)
    return np.argmin(distances)

def mini_batch_kmeans(X, k, batch_size, iterations):
    # Initialize centroids
    times = []
    start_time = time.time()
    C = initialize_centroids(X, k)
    init_time = time.time() - start_time
    times.append(f"Step 1: Initialize centroids {init_time:.6f}s")

    v = np.zeros(k)
    centroids_history = []
    radii_history = []

    cumulative_time = init_time

    for i in range(iterations):
        step_start_time = time.time()
        
        # Select a mini-batch of samples
        M = X[np.random.choice(X.shape[0], batch_size, replace=False)]
        
        # Cache the center nearest to each x in M
        d = {tuple(x): nearest_centroid(C, x) for x in M}
        
        for x in M:
            x = tuple(x)
            c = d[x]  # Get cached center for this x
            v[c] += 1  # Update per-center counts
            eta = 1 / v[c]  # Get per-center learning rate
            C[c] = (1 - eta) * C[c] + eta * np.array(x)  # Take gradient step
        
        # Store the current state of centroids and calculate radii
        centroids_history.append(C.copy())
        
        # Calculate radii for the current centroids
        radii = np.zeros(k)
        for j in range(k):
            cluster_points = np.array([x for x in X if nearest_centroid(C, x) == j])
            if len(cluster_points) > 0:
                distances = np.linalg.norm(cluster_points - C[j], axis=1)
                radii[j] = np.max(distances)
        radii_history.append(radii)

        step_time = time.time() - step_start_time
        cumulative_time += step_time
        times.append(f"Step {i + 2}: Iteration {i + 1} {cumulative_time:.6f}s")
    
    return C, centroids_history, radii_history, times

def plot_progress(X, centroids_history, radii_history, times):
    fig, ax = plt.subplots(figsize=(6, 6))  # Tạo subplot với kích thước cố định
    for i, (centroids, radii) in enumerate(zip(centroids_history, radii_history)):
        ax.scatter(X[:, 0], X[:, 1], c='gray', s=30, alpha=0.5, label='Data Points' if i == 0 else "")
        ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, marker='X', label='Centroids')
        for j in range(len(centroids)):
            circle = plt.Circle((centroids[j][0], centroids[j][1]), radii[j], color='red', fill=False, linestyle='--')
            ax.add_artist(circle)
        ax.set_title(times[i])
        ax.legend()
        plt.pause(10)  # Dừng một chút để có thể thấy rõ hơn
        if i < len(centroids_history) - 1:
            ax.clear()  # Xóa subplot trước khi vẽ tiếp
        
    plt.show()

# Example usage:
# New Data set X with more and diverse points
np.random.seed(42)
X1 = np.random.randn(100, 2) + np.array([5, 5])
X2 = np.random.randn(100, 2) + np.array([15, 15])
X3 = np.random.randn(100, 2) + np.array([25, 5])
X = np.vstack([X1, X2, X3])

# Number of clusters
k = 3

# Mini-batch size
batch_size = 20

# Number of iterations
iterations = 6

# Perform mini-batch KMeans
centroids, centroids_history, radii_history, times = mini_batch_kmeans(X, k, batch_size, iterations)

# Plot the progress
plot_progress(X, centroids_history, radii_history, times)

print("Final centroids:")
print(centroids)
