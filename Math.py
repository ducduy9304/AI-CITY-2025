import numpy as np

def rotation_matrix(yaw):

    R = np.array([
        [np.cos(yaw), np.sin(yaw), 0],
        [-np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])
    return R


# option to rotate and shift (for label info)
def create_corners(dimension, location=None, R=None):
    dx = dimension[0] / 2
    dy = dimension[1] / 2
    dz = dimension[2] / 2

    x_corners = [dx, -dx, -dx, dx, dx, -dx, -dx, dx] 
    y_corners = [-dy, -dy, dy, dy, -dy, -dy, dy, dy]
    z_corners = [dz, dz, dz, dz, -dz, -dz, -dz, -dz]

    corners = [x_corners, y_corners, z_corners]

    # rotate if R is passed in
    if R is not None:
        corners = np.dot(R, corners)

    # shift if location is passed in
    if location is not None:
        for i,loc in enumerate(location):
            corners[i,:] = corners[i,:] + loc

    final_corners = []
    for i in range(8):
        final_corners.append([corners[0][i], corners[1][i], corners[2][i]])


    return final_corners


# takes in a 3d point and projects it into 2d
def project_3d_word_to_pixel(points, E, K):
    points = np.array(points)
    points_homo = np.hstack([points, np.ones((points.shape[0], 1))])
    P = K @ E

    pixels = (P @ points_homo.T).T
    pixels_2d = pixels[:, :2] / pixels[:, 2:3]

    return pixels_2d

def project_pixel_to_3dworld(pixel, d, E, K):
    u = pixel[0]
    v = pixel[1]
    pixel = np.array([u, v, 1])
    point_cam = d * (np.linalg.inv(K) @ pixel)
    R = E[:, :3]
    t = E[:, -1]
    point_world = np.linalg.inv(R) @ (point_cam - t)

    return point_world

def to_bev(pixels, H, x_origin, y_origin, scale):
    point = np.array([pixels[0], pixels[1], 1])
    ground_point = np.linalg.inv(H) @ point
    ground_point /= ground_point[2]
    x_map = int((ground_point[0] + x_origin) * scale)
    y_map = int((y_origin - ground_point[1]) * scale) # dùng 33.5 nếu muốn show đúng trên bev map, còn dùng y_origin trong calib thì sẽ bị lệch
    return x_map, y_map
