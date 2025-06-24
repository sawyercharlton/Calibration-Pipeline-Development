import os
import cv2
import yaml
import argparse


def _get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-c', '--config',
        help='Path to the config file',
        type=str,
        default='configs/vis_chessboard.yaml',
    )

    args = parser.parse_args()

    return args


def _load_configs(path):
    with open(path, 'r') as yaml_file:
        configs = yaml.safe_load(yaml_file)

    return configs


def create_combined_video(config):
    cam_left_dir = config['cam_left_dir']
    cam_right_dir = config['cam_right_dir']
    output_path = config['output_path']
    fps = config.get('fps', 30)

    cam4_files = set(os.listdir(cam_left_dir))
    cam5_files = set(os.listdir(cam_right_dir))
    matched_files = sorted(cam4_files & cam5_files)

    if not matched_files:
        print("No matching files found.")
        return

    img1 = cv2.imread(os.path.join(cam_left_dir, matched_files[0]))
    img2 = cv2.imread(os.path.join(cam_right_dir, matched_files[0]))

    if img1 is None or img2 is None:
        raise ValueError("Initial images could not be read.")

    if img1.shape[0] != img2.shape[0]:
        height = min(img1.shape[0], img2.shape[0])
        img1 = cv2.resize(img1, (int(img1.shape[1] * height / img1.shape[0]), height))
        img2 = cv2.resize(img2, (int(img2.shape[1] * height / img2.shape[0]), height))

    combined_frame = cv2.hconcat([img1, img2])
    frame_height, frame_width = combined_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    for filename in matched_files:
        img1 = cv2.imread(os.path.join(cam_left_dir, filename))
        img2 = cv2.imread(os.path.join(cam_right_dir, filename))

        if img1 is None or img2 is None:
            print(f"Skipping {filename} due to read error.")
            continue

        if img1.shape[0] != img2.shape[0]:
            height = min(img1.shape[0], img2.shape[0])
            img1 = cv2.resize(img1, (int(img1.shape[1] * height / img1.shape[0]), height))
            img2 = cv2.resize(img2, (int(img2.shape[1] * height / img2.shape[0]), height))

        combined = cv2.hconcat([img1, img2])
        out.write(combined)

    out.release()
    print(f"Video saved to {output_path}")

if __name__ == "__main__":
    args = _get_arguments()
    configs = _load_configs(args.config)

    print(f"Config loaded: {configs}\n")
    create_combined_video(configs)
