import cv2 as cv
import numpy as np
from load_tracking_calib_1cam import load_groundtruth_and_calibration
from project_bev import to_bev

# === Load ảnh gốc và map ===
# img = cv.imread('frame0_camera4.jpg')
# img = cv.resize(img, (1280, 720))

map_file = 'map.png'
bev = cv.imread(map_file)


# === Load calibration ===
gt_file = 'ground_truth.json'
calib_file = 'calibration.json'
video_file = 'Camera_0006.mp4'
calib_camera_id = 'Camera_06'
gt_camera_id = 'Camera_0006'

# === Lấy H, R, t, K ===
H, extrinsic, intrinsic, scale, x_origin, y_origin, tracks = load_groundtruth_and_calibration(
    gt_file=gt_file, 
    calib_file=calib_file, 
    calib_camera_id=calib_camera_id,
    gt_camera_id=gt_camera_id, 
    target_id=4
)

# === Extrinsic R, t ===

# === Hàm chiếu world point sang BEV ===
def project_3d_world_to_pixel(points, E, K):
    points = np.array(points).reshape(1, 3) 
    points_homo = np.hstack([points, np.ones((points.shape[0], 1))])
    P = K @ E

    pixels = (P @ points_homo.T).T
    pixels_2d = pixels[:, :2] / pixels[:, 2:3]

    return pixels_2d

# === Các điểm world cần vẽ ===

p1_world = np.array([-0.27, -21.5, 0])
p2_world = np.array([-0.27, -21.5, 0])

# === Chiếu sang BEV ===
# Chiếu từng điểm
pixel1 = project_3d_world_to_pixel(p1_world, extrinsic, intrinsic)

pixel2 = project_3d_world_to_pixel(p2_world, extrinsic, intrinsic)
u1, v1 = pixel1[0]
u2, v2 = pixel2[0]
# Sang BEV
p1_bev = to_bev(u1, v1, H, x_origin, y_origin, scale)
p2_bev = to_bev(u2, v2, H, x_origin, y_origin, scale)


# === Vẽ điểm và mũi tên ===
cv.circle(bev, p1_bev, 6, (0, 255, 255), -1)
cv.circle(bev, p2_bev, 6, (0, 0, 255), -1)
cv.arrowedLine(bev, p1_bev, p2_bev, (255, 255, 0), 2)

# === Hiển thị label tọa độ ===
label1 = f"({p1_world[0]},{p1_world[1]},{p1_world[2]})"
label2 = f"({p2_world[0]},{p2_world[1]},{p2_world[2]})"
cv.putText(bev, label1, (p1_bev[0] + 5, p1_bev[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
cv.putText(bev, label2, (p2_bev[0] + 5, p2_bev[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

# === Hiển thị kết quả ===
bev = cv.resize(bev, (1280, 720))
cv.imshow("BEV: World Y axis test", bev)
cv.waitKey(0)
