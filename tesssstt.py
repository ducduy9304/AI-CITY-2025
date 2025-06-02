from Math import project_pixel_to_3dworld
import cv2
import h5py
import json
import numpy as np
from load_tracking_calib_1cam import load_groundtruth_and_calibration
from scipy.spatial.transform import Rotation as R
from Plotting import plot_3d_box
import math




def compute_h(h_pixel, depth_point, depth_frame, E, K):
    u = depth_point[0]
    v = depth_point[1]
    R_matrix    = E[:, :3]
    rvec, _     = cv2.Rodrigues(R_matrix)
    rotation    = R.from_rotvec(rvec.flatten())
    _, pitch_cam, _ = rotation.as_euler('zyx', degrees=False)
    fy = K[1, 1]
    d  = depth_frame[v, u] / 1000.0
    h_estimate = (d * h_pixel) / fy
    # Convert to real-world height using camera pitch (in radians!)
    h_final = h_estimate / np.sin(np.deg2rad(90) - pitch_cam) 

    return h_final


def compute_wl(pts:dict, depth_frame:np.ndarray, E:np.ndarray, K:np.ndarray, 
                  thresh_shoulder: float = 0.1, thresh_ankle: float = 0.3,
                  l_const: float = 0.5, w_const: float = 0.8):
    
    def to_word(coord):
        x_pix, y_pix = coord
        d = depth_frame[int(y_pix), int(x_pix)] / 1000
        if d <= 0:
            return None
        return project_pixel_to_3dworld(coord, d, E, K)
    
    # shoulder
    l_sh_px = pts.get('L_Shoulder')
    r_sh_px = pts.get('R_Shoulder')
    if (l_sh_px is not None) and (r_sh_px is not None):
        world_l_shoulder = to_word(l_sh_px)
        world_r_shoulder = to_word(r_sh_px)
    else:
        world_l_shoulder = world_r_shoulder = None
    
    
    # ankle
    l_ankle_px = pts.get('L_Ankle')
    r_ankle_px = pts.get('R_Ankle')
    if (l_ankle_px is not None) and (r_ankle_px is not None):
        world_l_ankle = to_word(l_ankle_px)
        world_r_ankle = to_word(r_ankle_px)
    else:
        world_l_ankle = world_r_ankle = None

    # distance
    w_final = w_const
    l_final = l_const

    shoulder_dist = 0.0
    if (world_l_shoulder is not None) and (world_r_shoulder is not None):
        shoulder_dist = np.linalg.norm(world_l_shoulder - world_r_shoulder)
    
    ankle_dist = 0.0
    if (world_l_ankle is not None) and (world_r_ankle is not None):
        ankle_dist = np.linalg.norm(world_l_ankle - world_r_ankle)
    
    is_shoulder_open = shoulder_dist >= thresh_shoulder
    is_ankle_open = ankle_dist >= thresh_ankle

    if is_shoulder_open:
        w_final = shoulder_dist
        if is_ankle_open:
            l_final = min(ankle_dist, 0.8)
        else:
            l_final = l_const
    else:
        w_final = w_const
        if is_ankle_open:
            l_final = min(ankle_dist, 0.8)
        else:
            l_final = l_const
    
    return w_final, l_final



def compute_yaw(pts: dict, depth_frame: np.ndarray, E: np.ndarray, K: np.ndarray, center_location: np.ndarray):
    """
    Compute yaw from shoulder first, 
    if dont have shoulder or person is rotating then use all the point in face to calculate the yaw
    """
    # if we have shouler points
    ls = pts.get('L_Shoulder')
    rs = pts.get('R_Shoulder')
    if (ls is not None) and (rs is not None):
        x_ls, y_ls = ls
        x_rs, y_rs = rs
        # depth pixel at shoulder (metter)
        d_ls = depth_frame[int(y_ls), int(x_ls)] / 1000
        d_rs = depth_frame[int(y_rs), int(x_rs)] / 1000

        if (d_ls > 0) and (d_rs > 0):
            X_ls, Y_ls, Z_ls = project_pixel_to_3dworld(ls, d_ls, E, K)
            X_rs, Y_rs, Z_rs = project_pixel_to_3dworld(rs, d_rs, E, K)
            dx_sh = X_ls - X_rs
            dy_sh = Y_ls - Y_rs
            
            # if distance between 2 shoulder is to min
            if math.hypot(dx_sh, dy_sh) >= 0.001:
                yaw_rad = math.atan2(-dx_sh, dy_sh)
                return yaw_rad
            
    # if shoulder points is wrong
    face_world_pts = []
    for key in ['Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear']:
        coord = pts.get(key)
        if coord is None:
            continue
        x_pix, y_pix = coord
        d = depth_frame[int(y_pix), int(x_pix)] / 1000
        Xw, Yw, Zw = project_pixel_to_3dworld(coord, d, E, K)
        face_world_pts.append((Xw, Yw, Zw))

    if len(face_world_pts) > 0:
        face_world = np.array(face_world_pts) # (N,3)
        X_face = float(np.mean(face_world[:, 0]))
        Y_face = float(np.mean(face_world[:, 1]))
        # vector from center location to centroid face
        dx_f = X_face - center_location[0]
        dy_f = Y_face - center_location[1]
        # if distance centroid is small, consider as 0 degree
        if math.hypot(dx_f, dy_f) >= 0.0001:
            yaw_rad = math.atan2(-dx_f, dy_f)
            return yaw_rad
    
    # if not shoulder either face then fallback yaw = 0
    return 0.0






video_file       = 'Camera_0001.mp4'
depth_file       = 'Camera_0001.h5'
pose_file        = 'tracklet (3).json'
gt_file          = 'ground_truth.json'
calib_file       = 'calibration.json'
calib_cam_id     = 'Camera_01'
gt_cam_id        = 'Camera_0001'
target_id        = 28
track_id_to_plot = 1

# thresholds and constants
thresh_shoulder_min = 0.1  
thresh_ankle_move   = 0.3
l_const             = 0.5
w_const             = 0.8

# Load ground-truth bounding boxes and calibration
homography, extrinsic, intrinsic, scaleFactor, x_origin, y_origin, tracks, shape = \
    load_groundtruth_and_calibration(
        gt_file           = gt_file,
        calib_file        = calib_file,
        calib_camera_id   = calib_cam_id,
        gt_camera_id      = gt_cam_id,
        target_id         = target_id
    )

# Load pose (keypoints) data
with open(pose_file, 'r') as f:
    pose_data = json.load(f)

# Open depth HDF5
h5f = h5py.File(depth_file, 'r')
keys = list(h5f.keys())

cap = cv2.VideoCapture(video_file)

# Initialize last_valid_pts as an empty dict; we'll populate keys as soon as we see valid coordinates.
last_valid_pts = {
        'Nose':         None,
        'L_Eye':        None,
        'R_Eye':        None,
        'L_Ear':        None,
        'R_Ear':        None,
        'L_Shoulder':   None,
        'R_Shoulder':   None,
        'L_Hip':        None,
        'R_Hip':        None,
        'L_Ankle':      None,
        'R_Ankle':      None
}

while True:
    frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    success, frame_bgr = cap.read()
    if not success:
        break

    # Extract current keypoints for this frame (if any)
    pts = {
        'Nose':         None,
        'L_Eye':        None,
        'R_Eye':        None,
        'L_Ear':        None,
        'R_Ear':        None,
        'L_Shoulder':   None,
        'R_Shoulder':   None,
        'L_Hip':        None,
        'R_Hip':        None,
        'L_Ankle':      None,
        'R_Ankle':      None
    }

    # Find the person entry with this frame and track_id
    for person in pose_data:
        if person['frame'] == frame_idx + 1 and person['track_id'] == track_id_to_plot:
            kp = {p['name']: p['coordinates'] for p in person['keypoints']}
            pts = {
                'Nose':       kp.get('nose'),
                'L_Eye':      kp.get('left_eye'),
                'R_Eye':      kp.get('right_eye'),
                'L_Ear':      kp.get('left_ear'),
                'R_Ear':      kp.get('right_ear'),
                'L_Shoulder': kp.get('left_shoulder'),
                'R_Shoulder': kp.get('right_shoulder'),
                'L_Hip':      kp.get('left_hip'),
                'R_Hip':      kp.get('right_hip'),
                'L_Ankle':    kp.get('left_ankle'),
                'R_Ankle':    kp.get('right_ankle'),
            }
            break

    #  For each of the four keypoints, if current frame has a valid (x,y), update last_valid_pts; 
    #    if missing, fill in from last_valid_pts
    for key in ['L_Shoulder', 'R_Shoulder','L_Hip', 'R_Hip', 'L_Ankle', 'R_Ankle']:
        if pts[key] is not None:
            # Update last_valid_pts for this key if it is a valid (x,y) pair
            x, y = pts[key]
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                last_valid_pts[key] = pts[key]
                # print(f"Frame {frame_idx}: {key} present at ({x:.1f}, {y:.1f}); updating last_valid_pts.")
            else:
                # If coordinate format is malformed, treat as missing
                print(f"Frame {frame_idx}: {key} coordinates malformed, treating as missing.")
                pts[key] = None
        else:
            # Current frame missing this keypoint → fill from last_valid_pts if available
            if last_valid_pts[key] is not None:
                pts[key] = last_valid_pts[key]
                print(f"Frame {frame_idx}: {key} missing → using previous value ({last_valid_pts[key][0]:.1f}, {last_valid_pts[key][1]:.1f}).")
            else:
                print(f"Frame {frame_idx}: {key} missing and no previous value available.")

    # At this point, pts[<key>] is guaranteed either to be None (if never seen) or the last known (x,y).
    #  can log final set of keypoints for this frame:
    for key in ['L_Shoulder', 'R_Shoulder', 'L_Hip', 'R_Hip', 'L_Ankle', 'R_Ankle']:
        if pts[key] is None:
            print(f"Frame {frame_idx}: keypoint {key} remains None.")
        else:
            x, y = pts[key]
            # print(f"Frame {frame_idx}: final keypoint {key} = ({x:.1f}, {y:.1f}).")

    #  Retrieve bounding box, yaw, etc. from calibration/tracks
    cx, cy, x1, y1, w, h, yaw = tracks[frame_idx][0]

    #  Depth-based height estimation
    depth_frame = h5f[keys[frame_idx]][()]

    # COMPUTE HEIGHT !!!!!!!
    u = int(x1 + w / 2)
    v = int(y1 + h / 2)
    depth_point = [u, v]
    h_final = compute_h(h_pixel=h, depth_point=depth_point, depth_frame=depth_frame, E=extrinsic, K=intrinsic)

    w_final, l_final = compute_wl(pts=pts, depth_frame=depth_frame, E=extrinsic, K=intrinsic,
                                  thresh_shoulder=thresh_shoulder_min, thresh_ankle=thresh_ankle_move,
                                  l_const=l_const, w_const=w_const)
    



    # # Calculate 3D positions of keypoints (using last_valid_pts if filled)
    # # already ensured pts[key] is never None if it had been seen previously.
    # lshoulder = pts['L_Shoulder']
    # rshoulder = pts['R_Shoulder']
    # lhip      = pts['L_Hip']
    # rhip      = pts['R_Hip']
    # lankle    = pts['L_Ankle']
    # rankle    = pts['R_Ankle']



    # # Depth lookups (in meters)
    # d_lshoulder = depth_frame[int(lshoulder[1]), int(lshoulder[0])] / 1000.0
    # d_rshoulder = depth_frame[int(rshoulder[1]), int(rshoulder[0])] / 1000.0
    # d_lhip      = depth_frame[int(lhip[1]), int(lhip[0])] / 1000.0
    # d_rhip      = depth_frame[int(rhip[1]), int(rhip[0])] / 1000.0
    # d_lankle    = depth_frame[int(lankle[1]),    int(lankle[0])]    / 1000.0
    # d_rankle    = depth_frame[int(rankle[1]),    int(rankle[0])]    / 1000.0

    # # Project to 3D world coordinates
    # lshoulder_world = project_pixel_to_3dworld(lshoulder, d_lshoulder, extrinsic, intrinsic)
    # rshoulder_world = project_pixel_to_3dworld(rshoulder, d_rshoulder, extrinsic, intrinsic)
    # lhip_world = project_pixel_to_3dworld(lhip, d_lhip, extrinsic, intrinsic)
    # rhip_world = project_pixel_to_3dworld(rhip, d_rhip, extrinsic, intrinsic)
    # lankle_world    = project_pixel_to_3dworld(lankle,    d_lankle,    extrinsic, intrinsic)
    # rankle_world    = project_pixel_to_3dworld(rankle,    d_rankle,    extrinsic, intrinsic)

    # # Compute distances for width and length
    # shoulder_dist = np.linalg.norm(lshoulder_world - rshoulder_world) + 0.1
    # hip_dist    = np.linalg.norm(lhip_world    - rhip_world) + 0.1
    # ankle_dist    = np.linalg.norm(lankle_world    - rankle_world) + 0.1

    # # BOOLEN
    # ankle_available = (lankle is not None) and (rankle is not None)
    # hip_available = (lhip is not None) and (rhip is not None)
    # is_shoulder_open = shoulder_dist >= thresh_shoulder_min
    # is_ankle_open = ankle_dist >= thresh_ankle_move
    
    
    # if is_shoulder_open:
    #     w_final = shoulder_dist
    #     if is_ankle_open:
    #         l_final = min(ankle_dist, 0.8)
    #         print("stading or walking")
    #     else:
    #         l_final = l_const
    #         print('rotating')
    
    # else:
    #     # shoulder not open
    #     w_final = w_const
    #     if is_ankle_open:
    #         l_final = min(ankle_dist, 0.8)
    #         print("rotating or walking")
    #     else:
    #         l_final = l_const


    # check if ankle is available
    # if ankle_available:
    #     bottom_center = (lankle_world + rankle_world) * 0.5
    # elif hip_available:
    #     avg_hip = (lhip_world + rhip_world) * 0.5
    #     bottom_center = avg_hip - np.array([0, 0, h_final/2])
    #     print('Using hip for center location')
    



    # Compute 3D box center: bottom center + half height in Z
    # bottom_center   = (lankle_world + rankle_world) * 0.5
    avg_hip = (lhip_world + rhip_world) * 0.5
    bottom_center = avg_hip - np.array([0, 0, h_final/2 - 0.12])
    center_location = bottom_center + np.array([0, 0, h_final / 2 - 0.15])

    # COMPUTE YAW !!!!!!!
    yaw_rad = compute_yaw(pts = pts, depth_frame=depth_frame, E = extrinsic, K = intrinsic, center_location=center_location)
    yaw_deg = math.degrees(yaw_rad)

    x_center_pix = int(x1 + w/2)
    y_center_pix = int(y1 + h/2)
    arrow_len = 30
    x_arrow = int(x_center_pix - arrow_len * math.sin(yaw_rad))
    y_arrow = int(y_center_pix + arrow_len * math.cos(yaw_rad))

    cv2.arrowedLine(
        frame_bgr,
        (x_center_pix, y_center_pix),
        (x_arrow, y_arrow),
        (0, 255, 0),
        2,
        tipLength=0.2
    )
    cv2.putText(
        frame_bgr,
        f"{yaw_deg:.1f}°",
        (x_center_pix - 30, y_center_pix - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
        cv2.LINE_AA
    )


    
    dimension       = np.array([min(w_final + 0.05, 0.8), l_final, min(h_final, 1.9)])
    print(f"W: {dimension[0]}, L: {dimension[1]}, H: {dimension[2]}")

    # Draw a 3D bounding box on the image
    plot_3d_box(frame_bgr, extrinsic, intrinsic, yaw, dimension, center_location)

    # Show and wait for a keypress
    frame_bgr = cv2.resize(frame_bgr, (1280, 720))
    cv2.imshow("3d bbox", frame_bgr)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cap.release()
h5f.close()
cv2.destroyAllWindows()
