from scipy.optimize import minimize
import numpy as np
from math import sqrt, log2
import math



def optimize_pow_height_cluster(
    clusters, centroid, p_thresh, h_thresh, alpha, channel_cap_thresh, bw_uav, var_n
):
    distance_remain_users_in_cluster = []
    distance_from_user_to_centroids = []
    # Calculate distance from users to centroid
    dist = np.sum((clusters - centroid) ** 2, axis=1)
    sorted_indices = np.argsort(dist)
    dist = dist[sorted_indices]

    # Hàm
    def objective(x):
        return alpha * x[0] + (1 - alpha) * x[1]

    # Constraints
    constraints = [
        {"type": "ineq", "fun": lambda x: p_thresh - x[0]},  # P <= P_threshold
        {"type": "ineq", "fun": lambda x: x[1] - h_thresh},  # H >= H_threshold
        {"type": "ineq", "fun": lambda x: x[0]},  # P >= 0
        {
            "type": "ineq",
            "fun": lambda x: x[0] / (((np.sqrt(dist[i]) + x[1] ** 2)**2) * var_n)
            - (2 ** (channel_cap_thresh / bw_uav) - 1),
        },  # BW*log (1 + P/(D^2 + H^2)) >= Channel_cap_thresh
    ]

    # Initialize variables
    pow_val = 0
    height_val = 0
    rad_val = 0
    users_served_val = 0
    total_users_val = len(clusters)
    service_time_val = 0
    # Loop through sorted indices
    for i in range(len(sorted_indices) - 1, -1, -1):
        # Minimize the objective function subject to constraints
        sol = minimize(objective, [p_thresh, h_thresh], constraints=constraints)

        # Check if constraints are satisfied
        if sol.success and all(
            constraint["fun"](sol.x) >= 0 for constraint in constraints
        ):
            pow_val = sol.x[0]
            height_val = sol.x[1]
            rad_val = np.sqrt(dist[i])
            users_served_val = i + 1
            # get all the distance of the remaining users in the cluster
            distance_remain_users_in_cluster = dist[:i+1]
            service_time_val = service_time(pow_val)
            break
    f3(pow_val, height_val, rad_val, bw_uav, var_n, channel_cap_thresh)
    channel_capacity = 0
    for r in distance_remain_users_in_cluster:
        channel_capacity += calculate_chanel_capacity(pow_val, height_val, r, bw_uav, var_n)    
    return pow_val, height_val, rad_val, channel_capacity, users_served_val, total_users_val, service_time_val


def f3(P, H, r, bw_uav, N, C_threshold):
    # print(f"pow: {P}, height: {H}, r: {r}, bw_uav: {bw_uav}, N: {N}, C_threshold: {C_threshold}")
    # print(P/((np.sqrt(r**2 + H**2)) * N))
    # print(1+ P/((np.sqrt(r**2 + H**2)) * N))
    # print(bw_uav * math.log2(1+ P/((np.sqrt(r**2 + H**2)) * N)) - C_threshold)
    return bw_uav * math.log2(1+ P/((np.sqrt(r**2 + H**2))**2 * N)) - C_threshold


def calculate_chanel_capacity(P, H, r, bw_uav, N):
    return bw_uav * math.log2(1+ P/((np.sqrt(r + H**2))**2 * N))

def service_time(P_module):
    # Điều kiện ban đầu
    time_minutes = 20.9  # Thời gian bay (phút)
    time_hours = time_minutes / 60  # Thời gian bay (giờ)
    V = 7.2  # Điện áp (V)

    # Tính công suất module phát (mAh)
    P_module_mAh = (P_module * time_hours)*1000 / V

    # Tính công suất tiêu thụ của UAV (mAh)
    P_UAV = 29000 / time_hours

    # Tính tổng công suất tiêu thụ
    P_total = P_UAV + P_module_mAh

    # Tính thời gian phục vụ (giờ)
    service_time_hours = 29000 / P_total

    return service_time_hours * 60  # Trả về thời gian phục vụ tính bằng phút
