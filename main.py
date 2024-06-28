import numpy as np
from generate_uniform_data import generate_uniform_data, visualize_data
from optimize_pow_height_cluster import optimize_pow_height_cluster, f3
from k_means_centroids import get_kmeans_centroids, visualize_kmeans
from plot_range_with_density import plot_all_UAV_range_with_density
from k_medois_centroids import k_medoids, visualize_k_medoids
import csv
import os
from draw_graph import plot_data_with_random_centroids,plot_density_with_range_of_random_centroids
from minibatch_k_means_centroids import get_mini_batch_k_means_centroids, visualize_mini_batch_k_means
# Initialize the parameters
num_of_clusters = 50
data_points_per_cluster = 270
no_of_users = num_of_clusters * data_points_per_cluster


# Initialize the optimal parameters
optimal_data = np.zeros((num_of_clusters, 7))
k_medoids_data = np.zeros((num_of_clusters, 7))
minibatch_k_means_data = np.zeros((num_of_clusters, 7))

power_threshold = 1 #40 W = 46,02 dBm 
bw_uav = 200000  # 20KHz
alpha = 0.5
chan_capacity_thresh = 1000000  # 1Mbps
Noise_Power = 0.0002
height_threshold = 0.4  # Landmark 72 cao 329m
#check if uniform_data.npy exists then load data from file

if os.path.exists('uniform_data.npy'):
    data = np.load('uniform_data.npy')
else:
    data = generate_uniform_data(no_of_users, 8)


k_means_centroids, k_means_clusters, k_means_time = get_kmeans_centroids(data, num_of_clusters)
k_medoids_centroids, k_medoids_clusters, k_medoids_time = k_medoids(data, num_of_clusters)
mini_batch_k_means_centroids, mini_batch_k_means_clusters, mini_batch_k_means_time = get_mini_batch_k_means_centroids(data, num_of_clusters)

print("Kmeans time: ", k_means_time)
print("Kmedoids time: ", k_medoids_time)
print("Mini batch kmeans time: ", mini_batch_k_means_time)
print("-----------------------")

# Split data for plotting
cluster_data = np.array_split(data, num_of_clusters)


# Tạo ngẫu nhiên các tọa độ x và y trong khoảng [0, plane_size)
X_random = np.random.uniform(0, 8, num_of_clusters)
Y_random = np.random.uniform(0, 8, num_of_clusters)

# Kết hợp các tọa độ x và y lại thành một mảng 2D
random_centroids = np.column_stack((X_random, Y_random))


   
print("Generating data...")

# Calculate the distance from each data point to each centroid
for i in range(num_of_clusters):
    (
        optimal_data[i, 0],
        optimal_data[i, 1],
        optimal_data[i, 2],
        optimal_data[i, 3],
        optimal_data[i, 4],
        optimal_data[i, 5],
        optimal_data[i, 6],
    ) = optimize_pow_height_cluster(
        k_means_clusters[i],
        k_means_centroids[i],
        power_threshold,
        height_threshold,
        alpha,
        chan_capacity_thresh,
        bw_uav,
        Noise_Power,
    )

for i in range(num_of_clusters):
    (
        k_medoids_data[i, 0],
        k_medoids_data[i, 1],
        k_medoids_data[i, 2],
        k_medoids_data[i, 3],
        k_medoids_data[i, 4],
        k_medoids_data[i, 5],
        k_medoids_data[i, 6],
    ) = optimize_pow_height_cluster(
        k_medoids_clusters[i],
        k_medoids_centroids[i],
        power_threshold,
        height_threshold,
        alpha,
        chan_capacity_thresh,
        bw_uav,
        Noise_Power,
    )

for i in range(num_of_clusters):
    (
        minibatch_k_means_data[i, 0],
        minibatch_k_means_data[i, 1],
        minibatch_k_means_data[i, 2],
        minibatch_k_means_data[i, 3],
        minibatch_k_means_data[i, 4],
        minibatch_k_means_data[i, 5],
        minibatch_k_means_data[i, 6],
    ) = optimize_pow_height_cluster(
        mini_batch_k_means_clusters[i],
        mini_batch_k_means_centroids[i],
        power_threshold,
        height_threshold,
        alpha,
        chan_capacity_thresh,
        bw_uav,
        Noise_Power,
    )

print("MiniBatch Kmeans data: ")
print(minibatch_k_means_data)



# Sum the 4th column of optimal_data
total_users_served_by_kmeans = np.sum(optimal_data[:, 4])
total_users_served_by_kmedoids = np.sum(k_medoids_data[:, 4])
total_users_served_by_minibatch_kmeans = np.sum(minibatch_k_means_data[:, 4])

print(f"Total users served by kmean: {total_users_served_by_kmeans}")
print(f"Total users served by kmedoids: {total_users_served_by_kmedoids}")
print(f"Total users served by minibatch kmeans: {total_users_served_by_minibatch_kmeans}")
print("-----------------------")
# Calculate the percentage of users served
percent_users_served = total_users_served_by_kmeans / no_of_users * 100
percent_users_served_kmedoids = total_users_served_by_kmedoids / no_of_users * 100
percent_users_served_minibatch_kmeans = total_users_served_by_minibatch_kmeans / no_of_users * 100
print(f"Optimal users served by kmean: {percent_users_served}" + "%")
print(f"Optimal users served by kmedoids: {percent_users_served_kmedoids}" + "%")
print(f"Optimal users served by minibatch kmeans: {percent_users_served_minibatch_kmeans}" + "%")
print("-----------------------")

# Visualize data
visualize_data(data, 8)
# Initialize array to store whether each user is served
user_served = np.zeros(len(data))
plot_data_with_random_centroids(data, random_centroids, 8)

# Calculate random number of users served
for i in range(len(data)):
    # Loop through each random centroid
    for j in range(len(random_centroids)):
        r = np.sqrt(
            (data[i][0] - random_centroids[j][0]) ** 2
            + (data[i][1] - random_centroids[j][1]) ** 2
        )
        if (
            f3(
                optimal_data[j, 0],
                height_threshold,
                r,
                bw_uav,
                Noise_Power,
                chan_capacity_thresh,
            )
            >= 0
        ):
            user_served[i] = 1
            # print(f'r equal {r}, P equal {optimal_data[j, 0]}')
            # print('F3 equal', f3(optimal_data[j, 0], height_threshold, r, bw_uav, Noise_Power, chan_capacity_thresh))
            break
 
print("Visualizing data...")
# Visualize data

# Plot data with kmeans centroids

print("Kmeans centroids")
visualize_kmeans(data, num_of_clusters, k_means_centroids, k_means_clusters)
name = "KMeans"
plot_all_UAV_range_with_density(optimal_data, num_of_clusters, data, k_means_clusters, k_means_centroids,name)

# Plot data with Kmedoids centroids
print("Kmedoids centroids")
visualize_k_medoids(data, num_of_clusters, k_medoids_centroids, k_medoids_clusters)
name = "Kmedoids"
plot_all_UAV_range_with_density(k_medoids_data, num_of_clusters, data, k_medoids_clusters, k_medoids_centroids,name)

# Plot data with MiniBatch Kmeans centroids
print("MiniBatch Kmeans centroids")
visualize_mini_batch_k_means(data, num_of_clusters, mini_batch_k_means_centroids, mini_batch_k_means_clusters)
name = "MiniBatchKMeans"
plot_all_UAV_range_with_density(minibatch_k_means_data, num_of_clusters, data, mini_batch_k_means_clusters, mini_batch_k_means_centroids,name)

# Plot density with range of random centroids
plot_all_UAV_range_with_density(minibatch_k_means_data, num_of_clusters, data, mini_batch_k_means_clusters, random_centroids, name)
# Count the number of users served
total_random_user_served = int(np.sum(user_served))
percent_users_random_served = total_random_user_served / no_of_users * 100
print(f"Total users random served: {total_random_user_served}")
print(f"Percent users random served: {percent_users_random_served}" + "%")
print("-----------------------")

average_maximum_range_kmeans = np.mean(optimal_data[:, 2])
average_maximum_range_kmedoids = np.mean(k_medoids_data[:, 2])
average_maximum_range_minibatch_kmeans = np.mean(minibatch_k_means_data[:, 2])
print(f"Average maximum range of kmeans: {average_maximum_range_kmeans} km")
print(f"Average maximum range of kmedoids: {average_maximum_range_kmedoids} km")
print(f"Average maximum range of minibatch kmeans: {average_maximum_range_minibatch_kmeans} km")
print("-----------------------")
# Calculate the average power of optimal_data
average_power = np.mean(optimal_data[:, 0])
average_power_kmedoids = np.mean(k_medoids_data[:, 0])
average_power_minibatch_kmeans = np.mean(minibatch_k_means_data[:, 0])
print(f"Average power of kmeans optimal data: {average_power} W")
print(f"Average power of kmedoids optimal data: {average_power_kmedoids} W")
print(f"Average power of minibatch kmeans optimal data: {average_power_minibatch_kmeans} W")
print("-----------------------")
# Calculate the total channel capacity of optimal_data
total_channel_capacity = np.sum(optimal_data[:, 3])
total_channel_capacity = total_channel_capacity / 1000000
total_channel_capacity_kmedoids = np.sum(k_medoids_data[:, 3])
total_channel_capacity_kmedoids = total_channel_capacity_kmedoids / 1000000
total_channel_capacity_minibatch_kmeans = np.sum(minibatch_k_means_data[:, 3])
total_channel_capacity_minibatch_kmeans = total_channel_capacity_minibatch_kmeans / 1000000
print(f"Total channel capacity of kmeans optimal data: {total_channel_capacity} Mbit/s")
print(f"Total channel capacity of kmedoids optimal data: {total_channel_capacity_kmedoids} Mbit/s")
print(f"Total channel capacity of minibatch kmeans optimal data: {total_channel_capacity_minibatch_kmeans} Mbit/s")
print("-----------------------")
# Calculate the average user's channel capacity
average_channel_capacity = total_channel_capacity / total_users_served_by_kmeans
# convert to Mbps
average_channel_capacity = average_channel_capacity 
average_channel_capacity_kmedoids = total_channel_capacity_kmedoids / total_users_served_by_kmedoids
average_channel_capacity_kmedoids = average_channel_capacity_kmedoids
print(f"Average channel capacity each user of kmeans optimal data: {average_channel_capacity} Mbit/s")
print(f"Average channel capacity each user of kmedoids optimal data: {average_channel_capacity_kmedoids} Mbit/s")
print(f"Average channel capacity each user of minibatch kmeans optimal data: {total_channel_capacity_minibatch_kmeans / total_users_served_by_minibatch_kmeans} Mbit/s")
print("-----------------------")

# Calculate the average service time
average_service_time = np.mean(optimal_data[:, 6])
average_service_time_kmedoids = np.mean(k_medoids_data[:, 6])
average_service_time_minibatch_kmeans = np.mean(minibatch_k_means_data[:, 6])
print(f"Average service time of kmeans optimal data: {average_service_time} minutes")
print(f"Average service time of kmedoids optimal data: {average_service_time_kmedoids} minutes")
print(f"Average service time of minibatch kmeans optimal data: {average_service_time_minibatch_kmeans} minutes")
print("-----------------------")
