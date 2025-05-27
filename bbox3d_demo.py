import json
import numpy as np
import cv2 as cv
from Plotting import plot_3d_box  # Ensure this file is correct

# === Load calibration data ===
with open("calibration.json", 'r') as f:
    cameras = json.load(f)

for sensor in cameras['sensors']:
    if sensor['type'] == 'camera' and sensor['id'] == 'Camera_04':
        K = np.array(sensor['intrinsicMatrix'])
        E = np.array(sensor['extrinsicMatrix'])  # 3x4 matrix
        H = sensor['homography']
        break

# === Load ground truth ===
with open("ground_truth.json", "r") as f:
    data = json.load(f)

# === Open video ===
path = 'Camera_0004.mp4'
cap = cv.VideoCapture(path)

# === Iterate through frames ===
while True:
    frame_idx = int(cap.get(cv.CAP_PROP_POS_FRAMES))
    success, frame = cap.read()
    if not success:
        break

    frame_key = str(frame_idx)
    if frame_key in data: 
        frame_data = data[frame_key]
        for obj in frame_data:
            if obj['object type'] == "Person" and obj['object id'] == 42:
                location = np.array(obj['3d location'])
                dimension = np.array(obj['3d bounding box scale'])
                yaw = obj['3d bounding box rotation'][2]
                bbox_dict = obj['2d bounding box visible']

                # Check if Camera_0004 has bbox
                if 'Camera_0004' not in bbox_dict:
                    continue
                bbox_2d = bbox_dict['Camera_0004']

                print("Frame:", frame_idx, "2D BBox:", bbox_2d)

                frame_rgb = cv.cvtColor(frame.copy(), cv.COLOR_BGR2RGB)

                # === Draw 3D bounding box ===
                plot_3d_box(frame_rgb, E, K, yaw, dimension, location)

                frame = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)
                break  # Only show 1 object each frame

    frame = cv.resize(frame, (1200, 700))
    cv.imshow("3D BBox Tracking", frame)

    # Press 'q' to quit
    if cv.waitKey(0) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

# === Optionally: Re-plot on specific frame 500 ===
# cap = cv.VideoCapture(path)
# cap.set(cv.CAP_PROP_POS_FRAMES, 500)
# success, frame = cap.read()
# cap.release()

# frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

# Use the last known object (you can adjust ID or find again if needed)
# plot_3d_box(frame, E, K, yaw, dimension, location)
# frame = cv.resize(frame, (1200, 700))
# cv.imshow("3D BBox Frame 500", frame)
# cv.waitKey(0)
# cv.destroyAllWindows()
