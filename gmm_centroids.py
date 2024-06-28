from sklearn.mixture import GaussianMixture
import numpy as np
import warnings
import matplotlib.pyplot as plt
def get_gmm_centroids(data, num_of_clusters):
    warnings.filterwarnings("ignore", category=FutureWarning)
    # Thực hiện phân cụm bằng GMM
    # Khởi tạo mô hình GMM sao cho GMM đạt hiệu suất tốt nhất
    
    gmm = GaussianMixture(n_components=num_of_clusters, init_params='k-means++', random_state=0)

    gmm.fit(data)
    idx = gmm.fit_predict(data)

    # Lấy trung tâm của các cụm
    centroids = gmm.means_
    data = np.array(data)
    # Chuyển đổi data thành mảng NumPy nếu chưa
    #data = np.array(data)

    # Lấy các điểm trong từng cụm
    gmm_clusters = []
    for i in range(num_of_clusters):
        cluster_i = data[idx == i]
        gmm_clusters.append(cluster_i)
    
    return centroids, gmm_clusters

def visualize_gmm(data, num_of_clusters, gmm_centroids, gmm_clusters):
    # Hiển thị dữ liệu và trung tâm cụm
    data = np.array(data)
    plt.figure(figsize=(10, 10))
    # Chia màu khác nhau cho từng cụm
    for i in range(num_of_clusters):
        plt.scatter(gmm_clusters[i][:, 0], gmm_clusters[i][:, 1], s=2)
    # plot tâm cụm bằng chữ x
    plt.scatter(gmm_centroids[:, 0], gmm_centroids[:, 1], s=50, c='black', marker='x')
    plt.show()


# import open3d as o3d
# import numpy as np

# def visualize_gmm_3d_with_height(data, num_of_clusters, gmm_centroids, gmm_clusters, height_threshold):
#     # Add height threshold to gmm_centroids
#     gmm_centroids_with_height = np.hstack((gmm_centroids, np.full((num_of_clusters, 1), height_threshold)))

#     # Add z=0 to gmm_clusters
#     for i in range(num_of_clusters):
#         gmm_clusters[i] = np.hstack((gmm_clusters[i], np.zeros((len(gmm_clusters[i]), 1))))

#     # Convert arrays to open3d point cloud format
#     gmm_centroids_cloud = o3d.geometry.PointCloud()
#     gmm_centroids_cloud.points = o3d.utility.Vector3dVector(gmm_centroids_with_height)
#     gmm_centroids_cloud.paint_uniform_color([1, 0, 0])  # Set the color of centroids to red

#     # Assign a unique color to each cluster
#     colors = plt.cm.tab10(np.arange(num_of_clusters))[:, :3]  # Generate unique RGB colors
#     gmm_clusters_cloud = []
#     for i in range(num_of_clusters):
#         cluster_cloud = o3d.geometry.PointCloud()
#         cluster_cloud.points = o3d.utility.Vector3dVector(gmm_clusters[i])
#         cluster_cloud.paint_uniform_color(colors[i])
#         gmm_clusters_cloud.append(cluster_cloud)

#     # Combine centroids and clusters into a single point cloud
#     all_clouds = [gmm_centroids_cloud] + gmm_clusters_cloud

#     # Create a visualization object
#     vis = o3d.visualization.Visualizer()
#     vis.create_window()
#     for cloud in all_clouds:
#         vis.add_geometry(cloud)

#     # Get the current view control
#     view_control = vis.get_view_control()
    
#     # Zoom out as far as possible
#     view_control.set_zoom(0.5)  # Adjust the zoom level as needed
    
#     # Set the size of the centroids to be larger
#     gmm_centroids_cloud.scale(1.5, center=gmm_centroids_cloud.get_center())  # Increase the scale factor as needed
    
#     # Run the visualization
#     vis.run()
#     vis.destroy_window()