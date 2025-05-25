import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import json



path_cam0 = 'Camera_0001.mp4'
cap_cam0= cv.VideoCapture(path_cam0)
cap_cam0.set(cv.CAP_PROP_POS_FRAMES, 0)
success_cam0, frame_cam0 = cap_cam0.read()
cap_cam0.release()
frame_cam0_rgb = cv.cvtColor(frame_cam0, cv.COLOR_BGR2RGB)





map_img = cv.imread('map.png')
map_img_rgb = cv.cvtColor(map_img, cv.COLOR_BGR2RGB)


with open("calibration.json", 'r') as f:
    cameras = json.load(f)

for sensor in cameras['sensors']:
    if sensor['type'] == 'camera' and sensor['id'] == 'Camera_01':
        K_1 = sensor['intrinsicMatrix']
        E_1 = sensor['extrinsicMatrix']
        H_1 = sensor['homography']
        scale_1 = sensor['scaleFactor']
        origin_x1 = sensor['translationToGlobalCoordinates']['x']
        origin_y1 = sensor['translationToGlobalCoordinates']['y']

        break



E_1 = np.array(E_1)
K_1 = np.array(K_1)
H_1 = np.array(H_1)




points_cam = [
    [129, 891],
    [539, 839],
    [643, 976],
    [132, 1041]
]

projected_pts_map = []



for (u, v) in points_cam:
    point_img = np.array([u, v, 1])
    
    point_ground = np.linalg.inv(H_1) @ point_img
    point_ground = point_ground / point_ground[2]
    
    x_map = int((point_ground[0] + origin_x1) * scale_1)
    y_map = int((origin_y1 - point_ground[1]) * scale_1)
    
    projected_pts_map.append((x_map, y_map))

print(projected_pts_map)

# === show matplotlib ===
# === Map ===
fig, axs = plt.subplots(2, 1, figsize=(14, 8))
axs[0].imshow(frame_cam0_rgb)
axs[0].scatter([pt[0] for pt in points_cam], [pt[1] for pt in points_cam], color='red', label='Camera Points')

axs[1].imshow(map_img_rgb)
for pt in projected_pts_map:
    axs[1].scatter([pt[0]], [pt[1]], color='blue')
polygon = np.array(projected_pts_map + [projected_pts_map[0]])
axs[1].plot(polygon[:, 0], polygon[:, 1], color='cyan', linestyle='--', label='Projected Region')

axs[0].set_title("Camera View")
axs[1].set_title("BEV Map Projection")
axs[0].legend()
axs[1].legend()
plt.tight_layout()
plt.show()