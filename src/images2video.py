import os
import cv2
import yaml
from glob import glob


def load_config(config_path='configs/images2video.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_video_from_images(image_dir):
    image_dir = os.path.abspath(image_dir)
    folder_name = os.path.basename(image_dir.rstrip('/'))
    output_video_path = os.path.join(image_dir, f'{folder_name}.mp4')

    image_paths = sorted(glob(os.path.join(image_dir, '*.*')), key=lambda x: x.lower())
    image_paths = [p for p in image_paths if any(p.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg'])]

    if not image_paths:
        print(f"No images found in {image_dir}")
        return

    # Read the first image to get frame size
    frame = cv2.imread(image_paths[0])
    height, width, _ = frame.shape
    fps = 30  # You can adjust FPS as needed

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for image_path in image_paths:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Skipping unreadable image {image_path}")
            continue
        out.write(img)

    out.release()
    print(f"Video written to: {output_video_path}")

if __name__ == '__main__':
    config = load_config()
    image_dir = config.get('images2video')
    if image_dir:
        create_video_from_images(image_dir)
    else:
        print("No 'images2video' path found in config.yaml.")
