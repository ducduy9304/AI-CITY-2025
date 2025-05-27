import numpy as np

def to_bev(u, v, H, x_origin, y_origin, scale):
    point = np.array([u, v, 1])
    ground_point = np.linalg.inv(H) @ point
    ground_point /= ground_point[2]
    x_map = int((ground_point[0] + x_origin) * scale)
    y_map = int((33.5 - ground_point[1]) * scale)
    return x_map, y_map