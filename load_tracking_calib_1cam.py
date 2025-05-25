import numpy as np
import cv2 as cv
import json
from collections import defaultdict

def load_groundtruth_and_calibration(gt_file, calib_file, calib_camera_id = 'Camera', gt_camera_id="Camera_0000", target_id = 0):
    with open(calib_file, 'r') as f:
        cameras = json.load(f)
    
    for sensor in cameras['sensors']:
        if sensor['type'] == 'camera' and sensor['id'] == calib_camera_id:
            H = np.array(sensor['homography'])
            extrinsic = np.array(sensor['extrinsicMatrix'])
            intrinsic = np.array(sensor['intrinsicMatrix'])
            scale = sensor['scaleFactor']
            x_origin = sensor['translationToGlobalCoordinates']['x']
            y_origin = sensor['translationToGlobalCoordinates']['y']
            break
    else:
        raise ValueError(f"Camera {calib_camera_id} not found in calibration file.")
    
    # load ground truth data
    with open(gt_file, 'r') as f:
        data = json.load(f)
    
    tracks = defaultdict(list)
    

    for frame_key, objs in data.items():
        frame_idx = int(frame_key)
        for obj in objs:
            if obj.get('object type') == "Person" and obj.get('object id') == target_id:
                bbox_dict = obj.get('2d bounding box visible', {})
                if gt_camera_id in bbox_dict:
                    x1, y1, x2, y2 = bbox_dict[gt_camera_id]
                    cx = int((x1 + x2) / 2)
                    cy = int(y2)
                    w = x2 - x1
                    h = y2 - y1
                    
                
                    # get yaw
                    rotation = obj.get('3d bounding box rotation', None)
                    if rotation is not None and len(rotation) == 3:
                        yaw = rotation[2]
                
                    tracks[frame_idx].append((cx, cy, x1, y1, w, h, yaw))
                    
                    
        
    return H, extrinsic, intrinsic, scale, x_origin, y_origin, tracks