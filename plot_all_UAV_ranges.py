import numpy as np
import matplotlib.pyplot as plt
r_bs = 0
r_uav = 0
def plot_all_UAV_range(uav_1, uav_2, centroids, x_bs, y_bs, P_bs, h_bs, P_uav, bw_bs, bw_uav, height_threshold, capacity_thresh, var_n):
    # Define colors
    #  c_bs_range green
    c_bs_range = [0, 1, 0]
    #  c_uav_1 blue
    c_uav_1 = [0, 0, 1]
    #  c_uav_2 red
    c_uav_2 = [1, 0, 0]

    fig = plt.figure('Communication Ranges', figsize=(10, 10))

    
    # Find and plot the radius of coverage of the UAVs
    d_uav = np.linspace(0, 1000, 1000)
    capacity_uav = bw_uav * np.log(1 + P_uav / ((d_uav ** 2 + (height_threshold) ** 2) * var_n))
    r_uav = d_uav[np.argwhere(np.diff(np.sign(capacity_uav - capacity_thresh)) != 0)].squeeze()
    for i in range(len(centroids)):
        plt.plot(centroids[i, 0] + r_uav * np.cos(np.linspace(0, 2 * np.pi, 100)),
                 centroids[i, 1] + r_uav * np.sin(np.linspace(0, 2 * np.pi, 100)),
                 color='r')

    # Find and plot the radius of the coverage of the base station
    x = np.linspace(0, 1000, 1000)
    capacity_bs = bw_bs * np.log(1 + P_bs / ((x ** 2 + (h_bs) ** 2) * var_n))
    r_bs = x[np.argwhere(np.diff(np.sign(capacity_bs - capacity_thresh)) != 0)].squeeze()
    plt.plot(x_bs + r_bs * np.cos(np.linspace(0, 2 * np.pi, 100)),
             y_bs + r_bs * np.sin(np.linspace(0, 2 * np.pi, 100)),
             color='g')

    # Set equal aspect ratio for the plot
    plt.axis('equal')
    # Plotting the UAV Data
    p_centroid = plt.plot(centroids[:, 0], centroids[:, 1], 'kx', markersize=10, linewidth=3, label='Centroids')
    p_center = plt.plot(x_bs, y_bs, 'ks', markersize=10, linewidth=3, label='Base Station')
    # p_uav_1 = plt.plot(uav_1[:, 0], uav_1[:,1], 'bo', markersize=5, linewidth=2, label='Intersection UAV 1')
    # p_uav_2 = plt.plot(uav_2[:,0], uav_2[:,1], 'ro', markersize=5, linewidth=2, label='Intersection UAV 2')
    p_uav_1 = plt.scatter(uav_1[:, 0], uav_1[:, 1], s=70, c=[c_uav_1], marker='^', label='UAV Intersection 1')
    p_uav_2 = plt.scatter(uav_2[:, 0], uav_2[:, 1], s=40, c=[c_uav_2], marker='^', label='UAV Intersection 2')
    # Add legend and labels
    plt.legend(loc='upper left')
    plt.title('Communication Ranges')
    plt.xlabel('X Distance')
    plt.ylabel('Y Distance')

    # Show the plot with all elements
    plt.show()
