import numpy as np
from Math import rotation_matrix, create_corners, project_3d_word_to_pixel
from enum import Enum
import cv2 as cv
import math

class cv_colors(Enum):
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)

def plot_3d_box(img, E, K, yaw, dimension, center_location, yaw_gt = None):
    
    R_y = rotation_matrix(yaw)
    corners = create_corners(dimension, location = center_location, R = R_y)
    corners = np.array(corners)
    
    pixels_2d = project_3d_word_to_pixel(corners, E, K)
    pixels_2d = pixels_2d.astype(int)
    
    edges = [
        (0,4), (1,5), (2,6), (3,7),
        (0,1), (1,2), (2,3), (3,0),
        (4,5), (5,6), (6,7), (7,4)
    ]

    for i, j in edges:
        
        cv.line(img, tuple(pixels_2d[i]), tuple(pixels_2d[j]), cv_colors.GREEN.value, 1)



    for idx, pt in enumerate(pixels_2d):
        cv.putText(img, str(idx), tuple(pt), cv.FONT_HERSHEY_SIMPLEX, 0.5, cv_colors.RED.value, 1)

    bottom_corners = [4,5,6,7]
    bottom_pts = corners[bottom_corners, :]

    bottom_center_3d = np.mean(bottom_pts, axis = 0, keepdims=True).T
    forward_dir = R_y @ np.array([[0], [-1], [0]])
    arrow_tip_3d = bottom_center_3d + 1.0 * forward_dir
    pts_3d = np.hstack((bottom_center_3d, arrow_tip_3d))


    # center_3d = np.array(center_location).reshape(3, 1)
    # forward_dir = R_y @ np.array([[0], [0], [1]])  
    # arrow_tip_3d = center_3d + 1.0 * forward_dir   

    # pts_3d = np.hstack((center_3d, arrow_tip_3d))  # (3, 2)
    pts_cam = (E @ np.vstack((pts_3d, np.ones((1, 2)))))  # (3,2)
    pts_img = (K @ pts_cam[:3] / pts_cam[2])[:2].T  # (2,2)

    
    pt1 = tuple(pts_img[0].astype(int))
    pt2 = tuple(pts_img[1].astype(int))

    # print(f"Arrow from {pt1} to {pt2}, image shape: {img.shape}")

    if np.all(np.isfinite(pt1)) and np.all(np.isfinite(pt2)):
        cv.arrowedLine(img, pt1, pt2, (0, 255, 255), 2, tipLength=0.3)
        cv.putText(img, f'{math.degrees(yaw):.1f}deg', pt2, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    
    # --- Draw arrow for yaw_gt (if provided) ---
    if yaw_gt is not None:
        R_gt = rotation_matrix(yaw_gt)
        forward_dir_gt = R_gt @ np.array([[0], [-1], [0]])
        arrow_tip_gt_3d = bottom_center_3d + 0.5 * forward_dir_gt
        pts_gt_3d = np.hstack((bottom_center_3d, arrow_tip_gt_3d))

        pts_gt_cam = (E @ np.vstack((pts_gt_3d, np.ones((1, 2)))))
        pts_gt_img = (K @ pts_gt_cam[:3] / pts_gt_cam[2])[:2].T

        pt1_gt = tuple(pts_gt_img[0].astype(int))
        pt2_gt = tuple(pts_gt_img[1].astype(int))

        if np.all(np.isfinite(pt1_gt)) and np.all(np.isfinite(pt2_gt)):
            cv.arrowedLine(img, pt1_gt, pt2_gt, (0, 0, 255), 2, tipLength=0.3)
            cv.putText(img, f'{math.degrees(yaw_gt):.1f}deg', pt2_gt, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


    cv.line(img, tuple(pixels_2d[0]), tuple(pixels_2d[5]), cv_colors.BLUE.value, 1)
    cv.line(img, tuple(pixels_2d[1]), tuple(pixels_2d[4]), cv_colors.BLUE.value, 1)

