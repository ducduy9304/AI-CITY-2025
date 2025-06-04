from Math import project_pixel_to_3dworld
import cv2
import h5py
import json
import numpy as np
from load_tracking_calib_1cam import load_groundtruth_and_calibration
from scipy.spatial.transform import Rotation as R
from Plotting import plot_3d_box
import math
from Math import to_bev



def compute_h(h_pixel, depth_point, depth_frame, E, K):
    u = depth_point[0]
    v = depth_point[1]
    R_matrix = E[:, :3]
    rvec, _ = cv2.Rodrigues(R_matrix)
    rotation = R.from_rotvec(rvec.flatten())
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



def compute_yaw(pts, depth_frame, E, K, bbox_center):
    """
    Compute yaw from shoulder first, 
    if dont have shoulder or person is rotating then use all the point in face to calculate the yaw
    """
    def to_word(coord):
        x_pix, y_pix = coord
        d = depth_frame[int(y_pix), int(x_pix)] / 1000
        if d <= 0:
            return None
        return project_pixel_to_3dworld(coord, d, E, K)
    
    ls = pts.get('L_Shoulder')
    rs = pts.get('R_Shoulder')

    if (ls is not None) and (rs is not None):
        world_l_shoulder = to_word(ls)
        world_r_shoulder = to_word(rs)

        dx_sh = world_l_shoulder[0] - world_r_shoulder[0]
        dy_sh = world_l_shoulder[1] - world_r_shoulder[1]
        if math.hypot(dx_sh, dy_sh) >= 1e-3:
            # vx = -dy_sh
            # vy = dx_sh
            yaw = math.atan2(-dy_sh, dx_sh)


            return yaw

    # fallback: centroid all keypoint in face
    face_keys = ['Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear']
    bev_face_pts = []
    for k in face_keys:
        kp = pts.get(k)
        if kp is not None:
            world_face = to_word(kp)
            x_face = world_face[0]
            y_face = world_face[1]
            bev_face_pts.append((x_face, y_face))
    
    if bev_face_pts:
        arr = np.array(bev_face_pts, dtype=float)
        x_face = float(arr[:, 0].mean())
        y_face = float(arr[:, 1].mean())

        # project center bbox 2d into bev
        uc, vc = bbox_center
        x_box, y_box = to_word(bbox_center)


        dx_f = x_face - x_box
        dy_f = y_face - y_box
        if math.hypot(dx_f, dy_f) >= 1e-3:
            yaw = math.atan2(-dy_f, dx_f)
            return yaw
    
    # if not shoulder either face then fallback yaw = 0
    return 0.0


# load file
video_file       = 'Camera_0001.mp4'
depth_file       = 'Camera_0001.h5'
pose_file        = 'tracklet (3).json'
gt_file          = 'ground_truth.json'
calib_file       = 'calibration.json'
calib_cam_id     = 'Camera_01'
gt_cam_id        = 'Camera_0001'
map_file         = 'map.png'
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

# load bev map
bev_map = cv2.imread(map_file)

cap = cv2.VideoCapture(video_file)

# Initialize last_valid_pts as an empty dict
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

    # Extract current keypoints for this frame 
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
                # print(f"Frame {frame_idx}: {key} coordinates malformed, treating as missing.")
                pts[key] = None
        else:
            # Current frame missing this keypoint → fill from last_valid_pts if available
            if last_valid_pts[key] is not None:
                pts[key] = last_valid_pts[key]
            #     print(f"Frame {frame_idx}: {key} missing → using previous value ({last_valid_pts[key][0]:.1f}, {last_valid_pts[key][1]:.1f}).")
            # else:
            #     print(f"Frame {frame_idx}: {key} missing and no previous value available.")


    #  Retrieve bounding box, yaw, etc. from calibration/tracks
    _, _, x1, y1, w, h, yaw_gt = tracks[frame_idx][0]
    print('yaw grountruth', math.degrees(yaw_gt))


    #  Depth-based height estimation
    depth_frame = h5f[keys[frame_idx]][()]

    # COMPUTE HEIGHT !!!!!!!
    u_center = int(x1 + w / 2)
    v_center = int(y1 + h / 2)
    center_bbox2d = (u_center, v_center)
    h_final = compute_h(h_pixel=h, depth_point=center_bbox2d, depth_frame=depth_frame, E=extrinsic, K=intrinsic)

    w_final, l_final = compute_wl(pts=pts, depth_frame=depth_frame, E=extrinsic, K=intrinsic,
                                  thresh_shoulder=thresh_shoulder_min, thresh_ankle=thresh_ankle_move,
                                  l_const=l_const, w_const=w_const)

    d_center = depth_frame[v_center, u_center] / 1000
    center_location = project_pixel_to_3dworld(pixel=center_bbox2d, d=d_center, E=extrinsic, K=intrinsic)
    

    # COMPUTE YAW !!!!!!!
    yaw_rad = compute_yaw(pts=pts, depth_frame=depth_frame, E=extrinsic, K=intrinsic, bbox_center=center_bbox2d)
    yaw_deg = math.degrees(yaw_rad)

    
    dimension = np.array([min(w_final + 0.1, 0.8), l_final + 0.2, min(h_final, 1.9)])
    # print(f"W: {dimension[0]}, L: {dimension[1]}, H: {dimension[2]}")
                  
    # Draw a 3D bounding box on the image
    plot_3d_box(frame_bgr, extrinsic, intrinsic, yaw_gt, dimension, center_location, yaw_gt)

    # Show and wait for a keypress
    frame_bgr = cv2.resize(frame_bgr, (1600, 900))
    cv2.imshow("3d bbox", frame_bgr)



    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cap.release()
h5f.close()
cv2.destroyAllWindows()
