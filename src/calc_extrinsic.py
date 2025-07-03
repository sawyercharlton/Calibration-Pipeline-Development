import json
import os
import time
import cv2
import yaml
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import logging


def setup_logger(log_path='calibration.log'):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')
    console.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file_handler)

    return logger


ORDER_VALID = (
    'cam2/cam1',
    'cam1/cam0',
)

STEREO_CALIBRATION_CRITERIA = (
    cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS,
    1000, 1e-6)


def _get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-c', '--config',
        help='Path to the config file',
        type=str,
        default='configs/calc_extrinsic.yaml',
    )

    args = parser.parse_args()

    return args


def _load_configs(path):
    with open(path, 'r') as yaml_file:
        configs = yaml.safe_load(yaml_file)

    return configs


def get_obj_points(configs):
    cols = configs['board']['cols']
    rows = configs['board']['rows']
    square_size = configs['board']['square_size']

    obj_points = np.zeros((cols * rows, 3), np.float32)
    obj_points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size

    return obj_points


def load_image_points(images_info, images):
    if not images_info:
        print("'images_info' not found.")

    if len(images_info.keys()) == 0:
        print("No images in images_info. Please run detect_board first.")

    img_points = []
    for key in tqdm(images):
        ret, corners = images_info[key]['findboardcorners_rgb']
        if not ret:
            continue

        img_points.append(corners)

    img_points = np.array(img_points, dtype=np.float32)

    return img_points


def find_matching_images(images_info, cam_1, cam_2):
    search_depth = min(len(images_info[cam_1]), len(images_info[cam_2]))

    matching_pairs = []
    for frame_idx in range(search_depth):
        if images_info[cam_1][frame_idx][0] \
                and images_info[cam_2][frame_idx][0]:
            matching_pairs.append(frame_idx)

    return matching_pairs


def calc_extrinsics(cam_1, cam_2, obj_points, intrinsics, extrinsics, configs):
    mtx_1 = np.array(intrinsics[cam_1]['mtx'], np.float32)
    dist_1 = np.array(intrinsics[cam_1]['dist'], np.float32)
    mtx_2 = np.array(intrinsics[cam_2]['mtx'], np.float32)
    dist_2 = np.array(intrinsics[cam_2]['dist'], np.float32)

    with open(configs['chessboards'], 'rb') as handler:
        images_info = pickle.load(handler)

    drop_first_n = configs.get('drop_first_n_pairs')
    max_pairs = configs.get('max_pairs')

    matching_pairs = find_matching_images(images_info, cam_1, cam_2)
    # Drop first N pairs
    if len(matching_pairs) > drop_first_n:
        matching_pairs = matching_pairs[drop_first_n:]
    else:
        matching_pairs = []

    if len(matching_pairs) > max_pairs:
        step = len(matching_pairs) // max_pairs
        matching_pairs = matching_pairs[::step][:max_pairs]

    # Technical Debt
    if len(matching_pairs) == 0:
        extrinsics[cam_1] = {
            'left_cam': cam_1,
            'right_cam': cam_2,
            'mtx_l': mtx_1.tolist(),
            'dist_l': dist_1.tolist(),
            'mtx_r': mtx_2.tolist(),
            'dist_r': dist_2.tolist(),
            'rotation': None,
            'transition': None,
        }
        return

    img_points_1 = np.array(
        [images_info[cam_1][frame_idx][1] for frame_idx in matching_pairs],
        np.float32)
    img_points_2 = np.array(
        [images_info[cam_2][frame_idx][1] for frame_idx in matching_pairs],
        np.float32)

    start_time = time.time()

    _, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
        np.tile(obj_points, (len(img_points_1), 1, 1)),
        img_points_1, img_points_2,
        mtx_1, dist_1, mtx_2, dist_2,
        (configs['image']['width'], configs['image']['height']),
        criteria=STEREO_CALIBRATION_CRITERIA, flags=cv2.CALIB_FIX_INTRINSIC)

    end_time = time.time()
    elapsed = end_time - start_time
    logger.info(
        f"Stereo calibration between '{cam_1}' and '{cam_2}' completed in {elapsed:.2f} seconds using "
        f"{len(matching_pairs)} matching image pairs.")

    extrinsics[cam_1] = {
        'left_cam': cam_1,
        'right_cam': cam_2,
        'mtx_l': mtx_1.tolist(),
        'dist_l': dist_1.tolist(),
        'mtx_r': mtx_2.tolist(),
        'dist_r': dist_2.tolist(),
        'rotation': R.tolist(),
        'transition': T.tolist(),
    }


def calc_reprojection_error(cam_1, cam_2, obj_points, extrinsics, configs):
    logger.info(f"[INFO] Starting reprojection error calculation between '{cam_1}' and '{cam_2}'...")

    with open(configs['chessboards'], 'rb') as handler:
        images_info = pickle.load(handler)

    matching_pairs = find_matching_images(images_info, cam_1, cam_2)

    # Technical Debt
    if len(matching_pairs) == 0:
        return

    img_points_1 = np.array(
        [images_info[cam_1][frame_idx][1] for frame_idx in matching_pairs],
        np.float32)
    img_points_2 = np.array(
        [images_info[cam_2][frame_idx][1] for frame_idx in matching_pairs],
        np.float32)

    mtx_1 = np.array(extrinsics[cam_1]['mtx_l'], dtype=np.float32)
    dist_1 = np.array(extrinsics[cam_1]['dist_l'], dtype=np.float32)
    mtx_2 = np.array(extrinsics[cam_1]['mtx_r'], dtype=np.float32)
    dist_2 = np.array(extrinsics[cam_1]['dist_r'], dtype=np.float32)
    R = np.array(extrinsics[cam_1]['rotation'], dtype=np.float32)
    T = np.array(extrinsics[cam_1]['transition'], dtype=np.float32)

    total_error = 0

    desc = f"Reprojection error [{cam_1} vs {cam_2}, {len(matching_pairs)} pairs]"
    for i in tqdm(range(len(img_points_1)), desc=desc, leave=False):
        _, rvec_l, tvec_l = cv2.solvePnP(obj_points, img_points_1[i], mtx_1, dist_1)
        rvec_r, tvec_r = cv2.composeRT(rvec_l, tvec_l, cv2.Rodrigues(R)[0], T)[:2]

        imgpoints1_projected, _ = cv2.projectPoints(obj_points, rvec_l, tvec_l, mtx_1, dist_1)
        imgpoints2_projected, _ = cv2.projectPoints(obj_points, rvec_r, tvec_r, mtx_2, dist_2)

        error1 = cv2.norm(img_points_1[i], imgpoints1_projected, cv2.NORM_L2) / len(imgpoints1_projected)
        error2 = cv2.norm(img_points_2[i], imgpoints2_projected, cv2.NORM_L2) / len(imgpoints2_projected)
        total_error += (error1 + error2) / 2

    average_error = total_error / len(img_points_1)
    logger.info(f"[INFO] Average reprojection error: {average_error:.4f} pixels\n")

def calc_extrinsic(configs):
    obj_points = get_obj_points(configs)

    with open(configs['intrinsics'], 'rb') as handler:
        intrinsics = pickle.load(handler)

    extrinsics = {}

    for cam_pair in ORDER_VALID:
        cam_1, cam_2 = cam_pair.split('/')

        calc_extrinsics(cam_1, cam_2, obj_points, intrinsics, extrinsics, configs)
        calc_reprojection_error(cam_1, cam_2, obj_points, extrinsics, configs)

    _store_artifacts(extrinsics, configs)


def _store_artifacts(artifact, configs):
    pkl_path = configs['output_dir']

    # Save as pickle (binary)
    with open(pkl_path, 'wb') as handle:
        pickle.dump(artifact, handle)

    # Derive JSON path by replacing .pkl with .json
    json_path = os.path.splitext(pkl_path)[0] + ".json"

    # Save as JSON (human-readable)
    with open(json_path, 'w') as handle:
        json.dump(artifact, handle, indent=2)



if __name__ == "__main__":
    args = _get_arguments()
    configs = _load_configs(args.config)

    logger = setup_logger(configs.get('log_pth'))
    logger.info(f"Config loaded: {configs}")

    calc_extrinsic(configs)
