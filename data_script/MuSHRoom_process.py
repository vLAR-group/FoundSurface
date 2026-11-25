import cv2
import numpy as np
from argparse import ArgumentParser
import os
from tqdm import tqdm
import json
from PIL import Image

depth_scale = 1000.0
parser = ArgumentParser()
parser.add_argument("--dataset_path", type=str, default="/path/to/MuSHRoom/room_datasets")
parser.add_argument("--output_path", type=str, default="/path/to/MuSHRoom/MuSHRoom_nerf")
args = parser.parse_args()

scene_lists = os.listdir(args.dataset_path)
scene_lists.sort()
for scene in tqdm(scene_lists):
    scene_path = os.path.join(args.dataset_path, scene, "kinect/long_capture")

    pose_file = os.path.join(scene_path, "transformations_colmap.json")

    with open(pose_file, 'r') as f:
        pose_data = json.load(f)
    fx, fy, cx, cy = pose_data["fl_x"], pose_data["fl_y"], pose_data["cx"], pose_data["cy"]
    k1, k2, p1, p2 = pose_data["k1"], pose_data["k2"], pose_data["p1"], pose_data["p2"]
    out_rgb_dir = os.path.join(args.output_path, scene, 'input')
    out_depth_dir = os.path.join(args.output_path, scene, 'depth')
    frames = pose_data["frames"]

    old_K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    distortion = np.array([k1, k2, p1, p2], dtype=np.float32)
    width, height = pose_data["w"], pose_data["h"]

    K, _ = cv2.getOptimalNewCameraMatrix(
        old_K, distortion, (width, height), 1, (width, height), True
    )
    map1, map2 = cv2.initUndistortRectifyMap(
        old_K, distortion, np.eye(3), K, (width, height), cv2.CV_32FC1
    )

    fx, fy, cx, cy = K[0, 0].item(), K[1, 1].item(), K[0, 2].item(), K[1, 2].item()
    fovx = 2 * np.arctan(cx / fx)
    fovy = 2 * np.arctan(cy / fy)
    os.makedirs(out_rgb_dir, exist_ok=True)
    os.makedirs(out_depth_dir, exist_ok=True)
    trans_info = {'train': {"depth_scale": depth_scale, 'cx': cx, 'cy': cy, 'fx': fx, 'fy': fy, "camera_angle_x":fovx, "camera_angle_y":fovy, 'frames': []},
        'test': { "depth_scale": depth_scale, 'cx': cx, 'cy': cy, 'fx': fx, 'fy': fy, "camera_angle_x":fovx, "camera_angle_y":fovy, 'frames': []},
        'all': {"depth_scale": depth_scale, 'cx': cx, 'cy': cy, 'fx': fx, 'fy': fy, "camera_angle_x":fovx, "camera_angle_y":fovy, 'frames': []},}

    for frame_idx, frame in enumerate(frames):
        img_path = os.path.join(scene_path, frame["file_path"])
        img = np.array(Image.open(img_path).convert("RGB"))
        img_name = os.path.basename(img_path)

        depth_path = os.path.join(scene_path, frame["depth_file_path"])
        depth_name = os.path.basename(depth_path)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / depth_scale
        depth[depth > 300.0] = 0.0

        undistorted_image = cv2.remap(
        img,
        map1,
        map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
        )

        undistorted_depth = cv2.remap(
            depth,
            map1,
            map2,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        pose = frame["transform_matrix"]


        undistorted_image = Image.fromarray(undistorted_image)
        undistorted_image.save(os.path.join(out_rgb_dir, img_name))
        undistorted_depth = (undistorted_depth * depth_scale).astype(np.uint16)
        cv2.imwrite(os.path.join(out_depth_dir, depth_name), undistorted_depth)


        info = {'file_path': f'input/{img_name}', 'depth_file_path': f'depth/{depth_name}',
                'transform_matrix': pose}
        trans_info['all']['frames'].append(info)
        if (frame_idx + 1) % 10 == 0:
            trans_info['test']['frames'].append(info)
        else:
            trans_info['train']['frames'].append(info)

    for split in ['train', 'test', 'all']:
        savedir = os.path.join(args.output_path, scene)
        with open(os.path.join(savedir, f'transforms_{split}.json'), 'w', encoding="utf-8") as fp:
            json.dump(trans_info[split], fp, indent=4)
