import numpy as np
from Math import rotation_matrix, create_corners, project_3d_word_to_pixel
from enum import Enum
import cv2 as cv



class cv_colors(Enum):
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)

def plot_3d_box(img, E, K, yaw, dimension, center_location):

    # plot_3d_pts(img, [center], center, calib_file=calib_file, cam_to_img=cam_to_img)
    
    R_y = rotation_matrix(yaw)
    corners = create_corners(dimension, location = center_location, R = R_y)

    pixels_2d = project_3d_word_to_pixel(corners, E, K, R_y, center_location)
    pixels_2d = pixels_2d.astype(int)
    
    edges = [
        (0,4), (1,5), (2,6), (3,7),
        (0,1), (1,2), (2,3), (3,0),
        (4,5), (5,6), (6,7), (7,4)
    ]

    for i, j in edges:
        
        cv.line(img, tuple(pixels_2d[i]), tuple(pixels_2d[j]), cv_colors.GREEN.value, 1)


    front_indices = [0, 1, 2, 3, 4, 5, 6, 7]

    for idx, pt in enumerate(pixels_2d):
        cv.putText(img, str(idx), tuple(pt), cv.FONT_HERSHEY_SIMPLEX, 0.5, cv_colors.RED.value, 1)

    # === Vẽ mũi tên thể hiện hướng yaw (hướng Z local sau quay) ===
    # Chuyển center_location sang tọa độ ảnh
    center_3d = np.array(center_location).reshape(3, 1)
    forward_dir = R_y @ np.array([[0], [0], [1]])  # trục Z local sau quay
    arrow_tip_3d = center_3d + 1.0 * forward_dir   # scale 1.0 mét cho vector chỉ hướng

    pts_3d = np.hstack((center_3d, arrow_tip_3d))  # (3, 2)
    pts_cam = (E @ np.vstack((pts_3d, np.ones((1, 2)))))  # (3,2)
    pts_img = (K @ pts_cam[:3] / pts_cam[2])[:2].T  # (2,2)

    pt1 = tuple(pts_img[0].astype(int))
    pt2 = tuple(pts_img[1].astype(int))
    

    cv.line(img, tuple(pixels_2d[0]), tuple(pixels_2d[5]), cv_colors.BLUE.value, 1)
    cv.line(img, tuple(pixels_2d[1]), tuple(pixels_2d[4]), cv_colors.BLUE.value, 1)

