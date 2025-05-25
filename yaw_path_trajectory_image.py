import cv2 as cv 
import json
import numpy as np
import matplotlib.pyplot as plt
import math
from project_bev import to_bev
from load_tracking_calib_1cam import load_groundtruth_and_calibration


# --- Load calibration ---
with open("calibration.json", 'r') as f:
    cameras = json.load(f)

for sensor in cameras['sensors']:
    if sensor['type'] == 'camera' and sensor['id'] == 'Camera':
        K = sensor['intrinsicMatrix']
        E = sensor['extrinsicMatrix']
        H = np.array(sensor['homography'])
        scale = sensor['scaleFactor']
        x_origin = sensor['translationToGlobalCoordinates']['x']
        y_origin = sensor['translationToGlobalCoordinates']['y']

        break

# --- Init ---
cap = cv.VideoCapture('Camera_0000.mp4')
with open('ground_truth.json', 'r') as f:
    data = json.load(f)

trajectory = []
points_to_project = {}

bev_map = cv.imread('map.png')
bev_map_original = bev_map.copy()
trajectory_bev = []



# --- Loop video ---
while True:
    frame_idx = int(cap.get(cv.CAP_PROP_POS_FRAMES))
    success, frame = cap.read()
    
    if not success:
        break

    frame_key = str(frame_idx)
    if frame_key in data:
        frame_data = data[frame_key]
        for obj in frame_data:
            if obj['object type'] == "Person" and obj['object id'] == 4:
                bbox_dict = obj.get('2d bounding box visible', {})
                if 'Camera_0004' in bbox_dict:
                    x1, y1, x2, y2 = bbox_dict['Camera_0000']
                    print(f"Width: {x2 - x1}, Height: {y2 - y1}")
                    cx = int((x1 + x2) / 2)
                    cy = int(y2)

                    trajectory.append((frame_idx, cx, cy))

                    # Vẽ bbox + tâm
                    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

                    # Tính yaw nếu có frame trước
                    if len(trajectory) >= 2:
                        _, cx_prev, cy_prev = trajectory[-2]
                        dx = cx - cx_prev
                        dy = cy - cy_prev
                        yaw_rad = math.atan2(dx, -dy)

                        # Vẽ mũi tên yaw
                        arrow_len = 50
                        x_arrow = int(cx + arrow_len * math.sin(yaw_rad)) 
                        y_arrow = int(cy - arrow_len * math.cos(yaw_rad))
                        cv.arrowedLine(frame, (cx, cy), (x_arrow, y_arrow), (0, 255, 255), 2)
                        cv.putText(frame, f"Yaw: {yaw_rad:.2f}", (cx + 5, cy - 10),
                                cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                    # # Ghi lại để chiếu sang BEV
                    # if frame_idx in target_frames:
                    #     points_to_project[frame_idx] = (cx, cy)
                    #     print(f"Projected point at frame {frame_idx}: ({cx}, {cy})")

    # Vẽ đường đi theo ảnh gốc
    for i in range(1, len(trajectory)):
        _, cx1, cy1 = trajectory[i - 1]
        _, cx2, cy2 = trajectory[i]
        cv.line(frame, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)

    frame = cv.resize(frame, (1200, 700))
    cv.imshow("Track Person ID=42 with Yaw", frame)
    if cv.waitKey(30) & 0xFF == ord('q'):
        break

    # if frame_idx > max(target_frames):
    #     break

cap.release()
cv.destroyAllWindows()

# --- Project to BEV ---
map = cv.imread('map.png')
map_rgb = cv.cvtColor(map, cv.COLOR_BGR2RGB)
projected_points = []


# for frame_id in target_frames:
#     if frame_id in points_to_project:
#         cx, cy = points_to_project[frame_id]
#         x_map, y_map = project_to_bev(cx, cy, H, x_origin, y_origin, scale)
#         projected_points.append((x_map, y_map))
#         print("BEV point", frame_id, ":", (x_map, y_map))

# --- Plot BEV ---
plt.figure(figsize=(10, 6))
plt.imshow(map_rgb)

for pt in projected_points:
    plt.scatter(pt[0], pt[1], color='red', s=6, label='Projected point')

plt.plot([pt[0] for pt in projected_points], [pt[1] for pt in projected_points],
         linestyle='-', color='blue', label='Trajectory line')

if len(projected_points) >= 2:
    x1, y1 = projected_points[0]
    x2, y2 = projected_points[1]
    dx = x2 - x1
    dy = y2 - y1
    yaw_rad = math.atan2(dx, -dy)
    print(f"Yaw angle (rad): {yaw_rad:.2f}")

    arrow_len = 100
    x_arrow = x1 + arrow_len * math.sin(yaw_rad)
    y_arrow = y1 - arrow_len * math.cos(yaw_rad)

    plt.arrow(x1, y1, x_arrow - x1, y_arrow - y1,
              head_width=15, head_length=20, fc='orange', ec='orange')
    plt.text(x1 + 10, y1 - 10, f"Yaw: {yaw_rad:.1f}", color='orange', fontsize=12)

plt.title("Projected trajectory on BEV map")
plt.legend()
plt.axis('off')
plt.show()
