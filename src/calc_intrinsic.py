import cv2
import yaml
import pickle
import json
import argparse
import numpy as np
import time
import platform
import psutil
import os
import logging


def setup_logger(log_path='intrinsic_calibration.log'):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')
    console.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file_handler)

    return logger

def _get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-c', '--config',
        help='Path to the config file',
        type=str,
        default='configs/calc_intrinsic.yaml',
    )

    args = parser.parse_args()

    return args


def _load_configs(path):
    with open(path, 'r') as yaml_file:
        configs = yaml.safe_load(yaml_file)

    return configs


def load_image_points(configs):
    with open(configs['chessboards'], 'rb') as handler:
        images_info = pickle.load(handler)

    if len(images_info.keys()) == 0:
        print("No images in images_info. Please run detect_chessboard first.")

    cameras = {}
    for camera in images_info.keys():
        cameras[camera] = {'img_points': []}

        for instance in images_info[camera]:
            ret, corners = instance
            if not ret:
                continue

            cameras[camera]['img_points'].append(corners)

        img_points = cameras[camera]['img_points']

        # Drop the first N chessboards
        drop_n = configs.get('drop_first_n_chessboard', 0)
        img_points = img_points[drop_n:]

        max_num = min(len(img_points), configs['max_chessboards'])

        # Uniformly spaced sampling
        step = len(img_points) / max_num
        sampled_points = [img_points[int(i * step)] for i in range(max_num)]

        cameras[camera]['img_points'] = sampled_points

    return cameras


def calculate_intrinsics(cameras_info, configs):
    cols = configs['board']['cols']
    rows = configs['board']['rows']
    square_size = configs['board']['square_size']

    obj_points = np.zeros((cols * rows, 3), np.float32)
    obj_points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size

    intrinsics = {}
    for key in cameras_info.keys():
        img_points = np.array(cameras_info[key]['img_points'],
                              dtype=np.float32)

        width = configs['image']['width']
        height = configs['image']['height']

        logger.info(f"Starting calibration for camera '{key}' with {len(img_points)} image samples...")
        start_time = time.time()
        ret, mtx, dist, rvecs, tvecs = \
            cv2.calibrateCamera(
                np.tile(obj_points, (len(img_points), 1, 1)),
                img_points,
                (width, height), None, None, flags=None)

        end_time = time.time()
        elapsed = end_time - start_time
        logger.info(f"Calibration for camera '{key}' completed in {elapsed:.2f} seconds.")

        mtx = mtx.tolist()
        dist = dist.tolist()
        # rvecs = [item.tolist() for item in rvecs]
        # tvecs = [item.tolist() for item in tvecs]

        intrinsics[key] = {
            "ret": ret,
            "mtx": mtx,
            "dist": dist,
            # "rvecs": rvecs,
            # "tvecs": tvecs,
        }

    # Hardware info
    cpu_name = platform.processor() or platform.uname().processor
    cpu_count = psutil.cpu_count(logical=True)
    mem = psutil.virtual_memory()
    logger.info(f"CPU: {cpu_name}")
    logger.info(f"Cores: {cpu_count}")
    logger.info(f"RAM: {mem.total / 1e9:.2f} GB (Available: {mem.available / 1e9:.2f} GB)")

    _store_artifacts(intrinsics, configs)


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


def calc_intrinsic(configs):
    cameras_info = load_image_points(configs)
    calculate_intrinsics(cameras_info, configs)


if __name__ == "__main__":
    args = _get_arguments()
    configs = _load_configs(args.config)

    logger = setup_logger(configs.get('log_path'))
    logger.info(f"Config loaded: {configs}")

    calc_intrinsic(configs)
