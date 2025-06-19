import os
import cv2
import yaml
import glob
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from threading import Thread


criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)


def _get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-c', '--config',
        help='Path to the config file',
        type=str,
        required=False,
        default='configs/detect_chessboard.yml',
    )

    args = parser.parse_args()

    return args


def _load_configs(path):
    with open(path, 'r') as yaml_file:
        configs = yaml.safe_load(yaml_file)

    return configs


def extract_chessboardcorners(path_imgs, images_info, camera_name, configs):
    offset = configs['images'][camera_name]['offset']

    paths_imgs = sorted(glob.glob(f"{path_imgs}/*"))
    paths_imgs = paths_imgs[offset:]

    if not images_info.__contains__(camera_name):
        images_info[camera_name] = []

    success_count = 0
    display_count = 0
    max_display = 10

    bar = tqdm(paths_imgs, dynamic_ncols=True, leave=False)
    bar.set_description(camera_name)
    for frame_index, path_img in enumerate(bar):
        image = cv2.imread(path_img)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(
            gray, (configs['board']['cols'], configs['board']['rows']),
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

        if ret:
            corners = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), criteria)
            corners = corners.reshape(configs['board']['rows'], configs['board']['cols'], 2)
            if corners[0, 0, 1] > corners[-1, -1, 1]:
                corners = corners[::-1, ::-1]
            corners = corners.reshape(-1, 1, 2).astype(np.float32)
            corners = corners.tolist()
        else:
            corners = []

        images_info[camera_name].append([ret, corners])

        if ret:
            success_count += 1

            # Save image (independent of display)
            if configs.get('save', False):
                _save(image.copy(), corners, configs['save_dir'], camera_name, frame_index)

            # Display image (independent of save)
            if configs.get('display', False) and display_count < max_display:
                if not _display(image, corners):
                    break
                display_count += 1

    print(f"\nFound {success_count} chessboards from {len(paths_imgs)} images for {camera_name}\n")


def _display(image, corners, save_dir, camera_name, frame_index):
    # Draw and annotate corners
    for idx_point, point in enumerate(corners):
        x = int(point[0][0])
        y = int(point[0][1])

        cv2.circle(image, (x, y), 5, (123, 105, 34), thickness=-1, lineType=8)
        cv2.putText(image, str(idx_point), (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=2)

    # Save annotated image to save_dir/camera_name/frame_index.jpg
    # camera_folder = os.path.join(save_dir, camera_name)
    # os.makedirs(camera_folder, exist_ok=True)
    # save_path = os.path.join(camera_folder, f"{frame_index}.jpg")
    # cv2.imwrite(save_path, image)

    # Display image
    cv2.imshow("Undistorted Image", image)
    key = cv2.waitKey(0)
    if key == ord('q'):
        return False

    return True

def _save(image, corners, save_dir, camera_name, frame_index):
    for idx_point, point in enumerate(corners):
        x = int(point[0][0])
        y = int(point[0][1])
        cv2.circle(image, (x, y), 5, (123, 105, 34), thickness=-1, lineType=8)
        cv2.putText(image, str(idx_point), (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=2)

    camera_folder = os.path.join(save_dir, camera_name)
    os.makedirs(camera_folder, exist_ok=True)
    save_path = os.path.join(camera_folder, f"{frame_index}.jpg")
    cv2.imwrite(save_path, image)


def calculate_total_success_dets(images_info):
    total_success_counter = 0
    for key in images_info.keys():
        for item in images_info[key]:
            if item[0]:
                total_success_counter += 1
    print(f"Grand num of found chessboards: {total_success_counter}")


def _store_artifacts(images_info, configs):
    with open(configs['output_dir'], 'wb') as handle:
        pickle.dump(images_info, handle)


def detect_chessboards(configs):
    images_info = {}

    processes = []
    for camera in configs['images']:
        process = Thread(
            target=extract_chessboardcorners,
            args=(configs['images'][camera]['path'], images_info,
                  camera, configs))
        process.start()
        processes.append(process)

        if not configs['parallel']:
            process.join()

    for process in processes:
        process.join()

    _store_artifacts(images_info, configs)

    calculate_total_success_dets(images_info)


if __name__ == "__main__":
    args = _get_arguments()
    configs = _load_configs(args.config)

    print(f"Config loaded: {configs}")

    detect_chessboards(configs)
