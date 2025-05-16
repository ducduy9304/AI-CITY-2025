import json
import numpy as np
import cv2 as cv
from Plotting import plot_3d_box


with open("calibration.json", 'r') as f:
    cameras = json.load(f)


for sensor in cameras['sensors']:
    if sensor['type'] == 'camera' and sensor['id'] == 'Camera_04':
        K = sensor['intrinsicMatrix']
        E = sensor['extrinsicMatrix']
        H = sensor['homography']
        break


K = np.array(K)

# extrinsic
E = np.array(E)


with open("ground_truth.json", "r") as f:
    data = json.load(f)

# frame_data = data["500"]
# for obj in frame_data:
#     if obj['object type'] == "Person" and obj['object id'] == 59:
#         location = obj['3d location']
#         dimension = obj['3d bounding box scale']
#         orientation = obj['3d bounding box rotation']




# ##############################################

# center_location = np.array(location)
# dimension = np.array(dimension)
# yaw = orientation[2]


# R_y = rotation_matrix(yaw)
# corners = create_corners(dimension, location = center_location, R = R_y)
# corners_pixel = project_3d_word_to_pixel(corners, E_1, K_1, R_y, center_location)
# print("Corners in pixel coordinates:", corners_pixel)

path = 'Camera_0004.mp4'
cap = cv.VideoCapture(path)

while True:
    frame_idx = int(cap.get(cv.CAP_PROP_POS_FRAMES))
    success, frame = cap.read()
    if not success:
        break

    frame_key = str(frame_idx)
    if frame_key in data:
        frame_data = data[frame_key]
        for obj in frame_data:
            if obj['object type'] == "Person" and obj['object id'] == 59:
                location = np.array(obj['3d location'])
                dimension = np.array(obj['3d bounding box scale'])
                yaw = obj['3d bounding box rotation'][2]

                frame_rgb = cv.cvtColor(frame.copy(), cv.COLOR_BGR2RGB)
                plot_3d_box(frame_rgb, E, K, yaw, dimension, location)
                frame = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)
                break

    # Resize and show the frame
    frame = cv.resize(frame, (1200, 700))
    cv.imshow("3D BBox Tracking", frame)
    
    # Press 'q' to quit early
    if cv.waitKey(15) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
# cap.set(cv.CAP_PROP_POS_FRAMES, 500)
# success, frame = cap.read()
# cap.release()
# frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

# plot_3d_box(frame, E_1, K_1, yaw, dimension, center_location)
# frame = cv.resize(frame, (1200, 700))
# cv.imshow("3D BBox", frame)
# cv.waitKey(0)
# cv.destroyAllWindows()

