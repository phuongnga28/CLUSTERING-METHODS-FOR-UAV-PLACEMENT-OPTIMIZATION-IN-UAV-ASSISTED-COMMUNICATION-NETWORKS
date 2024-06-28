import numpy as np
from matplotlib import pyplot as plt

def generate_uniform_data(num_points, plane_size):
    # Tạo các tọa độ x và y ngẫu nhiên từ phân phối đều trong khoảng [0, plane_size)
    x = np.random.uniform(0, plane_size, num_points)
    y = np.random.uniform(0, plane_size, num_points)

    # Kết hợp các tọa độ x và y lại thành một mảng 2D
    data = np.column_stack((x, y))
    return data

def visualize_data(data, plane_size):
    plt.figure(figsize=(10, 10))
    plt.xlim(0, plane_size)
    plt.ylim(0, plane_size)
    plt.scatter(data[:, 0], data[:, 1], s=2, label="User's Location")
    plt.legend(loc='upper right', fontsize=14)
    plt.show()
