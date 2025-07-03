import os

import cv2
import yaml
import glob
import torch
import pickle
import argparse
import numpy as np
from tqdm import tqdm
# from mmpose.apis import MMPoseInferencer
from transformers import ViTModel, ViTImageProcessor
import mediapipe as mp


def _get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-c', '--config',
        help='Path to the config file',
        type=str,
        default='configs/skeleton_detection.yaml',
    )

    args = parser.parse_args()

    return args


def _load_configs(path):
    with open(path, 'r') as yaml_file:
        configs = yaml.safe_load(yaml_file)

    return configs


def _bbox_area(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def filter_sort(people_keypoints, configs):
    people_keypoints = [person
                        for person in people_keypoints
                        if person['bbox_score'] > configs['detect_threshold'] and \
                            person['bbox'][0][3] - person['bbox'][0][1] > configs['detect_min_height'] and \
                            person['bbox'][0][0] >= configs['detect_min_x'] and \
                            person['bbox'][0][2] <= configs['detect_max_x']]

    people_keypoints = sorted(
        people_keypoints,
        key=lambda x: _bbox_area(x['bbox'][0]),
        reverse=True)

    return people_keypoints


class MediaPipePoseWrapper:
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(static_image_mode=True)

    def __call__(self, image):
        img_rgb = image.copy()
        result = self.pose.process(img_rgb)
        if not result.pose_landmarks:
            return [{
                "predictions": []  # Empty list if no detection
            }]

        keypoints = []
        scores = []
        for lm in result.pose_landmarks.landmark:
            keypoints.append([lm.x * img_rgb.shape[1], lm.y * img_rgb.shape[0]])
            scores.append(lm.visibility)

        dummy_bbox = [[0, 0, img_rgb.shape[1], img_rgb.shape[0]]]  # (x1, y1, x2, y2)

        return [{
            "predictions": [{
                "keypoints": keypoints,
                "keypoint_scores": scores,
                "bbox": dummy_bbox,
                "bbox_score": 1.0
            }]
        }]


def _get_skeleton(image, inferencer, configs):
    result_generator = inferencer(image)
    
    detected_keypoints = []
    detected_confidences = []
    for result in result_generator:
        # print("Raw Model Output:", result)

        # predictions = filter_sort(
        #     result['predictions'][0], configs)
        if not result['predictions']:
            predictions = []
            # print("❌ No predictions found in model output.")

        else:
            predictions = filter_sort(result['predictions'], configs)
            # print("Filtered predictions:", predictions)  # ADD THIS

        if len(predictions) > 0:
            keypoints = predictions[0]['keypoints']
            confidences = (np.array(predictions[0]['keypoint_scores']) \
                            / configs['confidence_coeff']).tolist()
            # print(f"✅ Skeleton detected with {len(keypoints)} keypoints")

        else:
            keypoints = []
            confidences = []
            # print("❌ No skeleton detected")

        detected_keypoints.append(keypoints)
        detected_confidences.append(confidences)

    return detected_keypoints, detected_confidences


def extract_poses(dir, camera, model_2d, intrinsics, configs):
    mtx = np.array(intrinsics[camera]['mtx'], np.float32)
    dist = np.array(intrinsics[camera]['dist'], np.float32)

    poses = []
    poses_confidence = []

    offset = configs['images'][camera]['offset']
    exp_length = configs['experiment_length']
    paths_imgs = sorted(glob.glob(f"{dir}/*"))
    paths_imgs = paths_imgs[offset:offset + exp_length]

    visualizations = []
    detected_frames = []

    bar = tqdm(paths_imgs)
    bar.set_description(camera)
    for frame_idx, path_img in enumerate(bar):
        image = cv2.imread(path_img)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # img_rgb = cv2.undistort(img_rgb.copy(), mtx, dist, None, None)

        people_keypoints, confidences = _get_skeleton(img_rgb, model_2d, configs)

        has_keypoints = any(p for p in people_keypoints if p)
        if has_keypoints:
            detected_frames.append((frame_idx, img_rgb.copy(), people_keypoints, confidences))

        if configs['visualize']:
            # Still show in real-time
            visualize_keypoints(img_rgb, people_keypoints, confidences)

        poses.append(people_keypoints)
        poses_confidence.append(confidences)

    # Save evenly sampled detected frames
    save_dir = configs.get('save_dir')
    save_max = configs.get('max_saved_images_per_camera', 5)
    if save_dir:
        save_sampled_frames(detected_frames, save_dir=os.path.join(save_dir, camera), save_max=save_max)

    return poses, poses_confidence


# img_rgb = cv2.undistort(img_rgb.copy(), mtx, dist, None, None)
def save_sampled_frames(detected_frames, save_dir, save_max):
    if not detected_frames:
        return

    os.makedirs(save_dir, exist_ok=True)

    total = len(detected_frames)
    num_to_save = min(save_max, total)
    step = total / num_to_save

    for i in range(num_to_save):
        idx = round(i * step)
        if idx >= total:
            idx = total - 1

        frame_idx, image, keypoints, confidences = detected_frames[idx]
        for person in keypoints:
            for point in person:
                if point:
                    cv2.circle(image, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)

        save_path = os.path.join(save_dir, f"frame_{frame_idx:04d}.png")
        cv2.imwrite(save_path, image)


def visualize_keypoints(image, keypoints, confidences):
    image_vis = image.copy()

    for person in keypoints:
        for point in person:
            if point:
                cv2.circle(image_vis, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)

    cv2.imshow("Detected", cv2.resize(image_vis, (1280, 720)))
    cv2.waitKey(1)



def _store_artifacts(artifact, output):
    with open(output, 'wb') as handle:
        pickle.dump(artifact, handle)


def calc_2d_skeleton(cameras, model_2d, configs):
    with open(configs['intrinsics'], 'rb') as handler:
        intrinsics = pickle.load(handler)

    keypoints = {}
    for _, camera in enumerate(cameras):
        dir = configs['images'][camera]['path']

        pose, pose_confidence = extract_poses(
            dir, camera, model_2d, intrinsics, configs)
        
        keypoints[camera] = {
            'pose': pose,
            'pose_confidence': pose_confidence,
        }

    return keypoints


if __name__ == "__main__":
    args = _get_arguments()
    configs = _load_configs(args.config)
    print(f"Config loaded: {configs}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    processor = ViTImageProcessor.from_pretrained('google/vit-large-patch32-384')
    model = ViTModel.from_pretrained('google/vit-large-patch32-384')
    model.to(device)

    model_2d = MediaPipePoseWrapper()

    cameras = configs["images"].keys()
        
    keypoints = calc_2d_skeleton(cameras, model_2d, configs)
    # Count how many images had skeletons detected
    for cam, data in keypoints.items():
        num_detected = sum(len(p) > 0 and len(p[0]) > 0 for p in data['pose'])
        total = len(data['pose'])
        print(f"{cam}: Detected skeletons in {num_detected} out of {total} frames")

    _store_artifacts(
        keypoints,
        configs['output'])
