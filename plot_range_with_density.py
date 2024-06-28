import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

r_bs = 0
r_uav = 0

extend_range = 0
def plot_all_UAV_range_with_density(
    optmial_data,
    num_of_clusters,
    data,
    gmm_clusters,
    centroids,
    name
):
    # Define colors
    #  c_bs_range green
    c_bs_range = [0, 1, 0]
    #  c_uav_1 blue
    c_uav_1 = [0, 0, 1]
    #  c_uav_2 red
    c_uav_2 = [1, 0, 0]
    if name == "MiniBatchKMeans":
        extend_range = 0.35
    elif name == "KMeans":
        extend_range = 0.28
    elif name == "Kmedoids":
        extend_range = 0.25
    fig = plt.figure(f"Communication Ranges With Density with {name}", figsize=(10, 10))
    plane_size = 8
    plt.xlim(0, plane_size)
    plt.ylim(0, plane_size)
    for i in range(len(optmial_data)):
        # Vẽ hình tròn có tâm là gmm_centroids[i, 0], gmm_centroids[i, 1] và bán kính là optimal_data[i, 1]
        circle_uav = plt.Circle(
            (centroids[i, 0], centroids[i, 1]),
            optmial_data[i, 1]+extend_range,
            color='red',
            fill=False,
        )
        # Đánh số các UAV
        # plt.text(centroids[i, 0], centroids[i, 1], f"{i+1}", fontsize=8, fontweight="bold", backgroundcolor='white')
        fig.gca().add_artist(circle_uav)

    # Set equal aspect ratio for the plot

    plt.axis("equal")
    # Plotting the UAV Data
    p_centroid = plt.plot(
        centroids[:, 0],
        centroids[:, 1],
        "kx", 
        markersize=5,
        linewidth=2,
        label="Position of UAVs",
    )

    # plot data points on plane size with small points with different colors
    plt.scatter(data[:, 0], data[:, 1], s=0.5)

    plt.legend(loc='upper right', fontsize=14)
    # add more infor on legend

    plt.xlabel("X Distance")
    plt.ylabel("Y Distance")

    # Show the plot with all elements
    plt.show()