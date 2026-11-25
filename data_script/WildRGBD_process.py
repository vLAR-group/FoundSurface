
import os
from argparse import ArgumentParser
from tqdm import tqdm
import json
import numpy as np
import cv2
from utils import crop_image_depthmap, rescale_image_depthmap
import matplotlib.pyplot as plt
import PIL.Image


def imread_cv2(path, options=cv2.IMREAD_COLOR):
    """ Open an image or a depthmap with opencv-python.
    """
    if path.endswith(('.exr', 'EXR')):
        options = cv2.IMREAD_ANYDEPTH
    img = cv2.imread(path, options)
    if img is None:
        raise IOError(f'Could not load image={path} with {options=}')
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


parser = ArgumentParser()
parser.add_argument("--dataset_path", type=str, default="/path/to/wildrgbd-main")
parser.add_argument("--output_path", type=str, default="/path/to/wildrgbd/wildrgbd_nerf")
parser.add_argument("--img_size", type=int, default=512,
                    help=("lower dimension will be >= img_size * 3/4, and max dimension will be >= img_size"))
args = parser.parse_args()

img_size = args.img_size
category_lists = [entry.name for entry in os.scandir(args.dataset_path) if entry.is_dir()]
category_lists = ["TV"]
depth_scale = 1000.0
for  category in tqdm(category_lists):
    scene_lists = [entry.name for entry in os.scandir(os.path.join(args.dataset_path, category, "scenes")) if entry.is_dir()]
    scene_lists.sort()
    scene_lists = ['scene_000']
    for scene in tqdm(scene_lists):
        scene_path = os.path.join(args.dataset_path, category, "scenes", scene)
        img_dir = os.path.join(scene_path, "rgb")
        depth_dir = os.path.join(scene_path, "depth")
        mask_dir = os.path.join(scene_path, "masks")
        pose_dir = os.path.join(scene_path, "cam_poses.txt")
        intrinsic_dir = os.path.join(scene_path, "metadata")
        out_rgb_dir = os.path.join(args.output_path, category + "_" + scene, 'input')
        out_depth_dir = os.path.join(args.output_path, category + "_" + scene, 'depth')
        out_mask_dir = os.path.join(args.output_path, category + "_" + scene, 'mask')
        os.makedirs(out_rgb_dir, exist_ok=True)
        os.makedirs(out_depth_dir, exist_ok=True)
        os.makedirs(out_mask_dir, exist_ok=True)


        with open(intrinsic_dir, 'r') as f:
            metadata = json.load(f)

        K = np.array(metadata["K"]).reshape(3, 3).T
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        w, h = metadata["w"], metadata["h"]

        camera_intrinsics = np.array(
            [[fx, 0, cx],
             [0, fy, cy],
             [0, 0, 1]]
        )
        camera_to_world_content = np.genfromtxt(pose_dir)
        camera_to_world = camera_to_world_content[:, 1:].reshape(-1, 4, 4)

        fovx = 2 * np.arctan(cx / fx)
        trans_info = {'train': {'fx': fx, 'fy': fy, 'cx':cx, 'cy':cy, "camera_angle_x":fovx, "depth_scale": depth_scale, 'frames': []},
        'test': {'fx': fx, 'fy': fy, 'cx':cx, 'cy':cy, "camera_angle_x":fovx, "depth_scale": depth_scale, 'frames': []},
        'all': { 'fx': fx, 'fy': fy, 'cx':cx, 'cy':cy, "camera_angle_x":fovx, "depth_scale": depth_scale, 'frames': []},}

        rgb_img_lists = os.listdir(img_dir)
        rgb_img_lists.sort()

        for frame_id, rgb_img in enumerate(rgb_img_lists):
            depth_path = os.path.join(depth_dir, f'{frame_id:0>5d}.png')
            masks_path = os.path.join(mask_dir, f'{frame_id:0>5d}.png')
            rgb_path = os.path.join(img_dir, f'{frame_id:0>5d}.png')

            input_rgb_image = PIL.Image.open(rgb_path).convert('RGB')
            input_mask = plt.imread(masks_path)
            input_depthmap = imread_cv2(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float64)
            depth_mask = np.stack((input_depthmap, input_mask), axis=-1)
            H, W = input_depthmap.shape

            min_margin_x = min(cx, W - cx)
            min_margin_y = min(cy, H - cy)

            l, t = int(cx - min_margin_x), int(cy - min_margin_y)
            r, b = int(cx + min_margin_x), int(cy + min_margin_y)
            crop_bbox = (l, t, r, b)
            input_rgb_image, depth_mask, input_camera_intrinsics = crop_image_depthmap(
                input_rgb_image, depth_mask, camera_intrinsics, crop_bbox)


            scale_final = ((img_size * 3 // 4) / min(H, W)) + 1e-8
            output_resolution = np.floor(np.array([W, H]) * scale_final).astype(int)
            if max(output_resolution) < img_size:
                scale_final = (img_size / max(H, W)) + 1e-8
                output_resolution = np.floor(np.array([W, H]) * scale_final).astype(int)

            input_rgb_image, depth_mask, input_camera_intrinsics = rescale_image_depthmap(
                input_rgb_image, depth_mask, input_camera_intrinsics, output_resolution)
            input_depthmap = depth_mask[:, :, 0]
            input_mask = depth_mask[:, :, 1]

            camera_pose = camera_to_world[frame_id]


            img_save_path = os.path.join(out_rgb_dir, f"{frame_id:0>5d}.png")
            depth_save_path = os.path.join(out_depth_dir, f"{frame_id:0>5d}.png")
            mask_save_path = os.path.join(out_mask_dir, f"{frame_id:0>5d}.png")
            input_rgb_image.save(img_save_path)
            cv2.imwrite(depth_save_path, input_depthmap.astype(np.uint16))
            cv2.imwrite(mask_save_path, (input_mask * 255).astype(np.uint8))

            camera_pose[:3, 1:3] *= -1
            info = {'file_path': f"input/{frame_id:0>5d}.png", 'depth_file_path': f"depth/{frame_id:0>5d}.png",
                     'transform_matrix': camera_pose.tolist(), 'mask_file_path': f"mask/{frame_id:0>5d}.png"}

            trans_info['all']['frames'].append(info)
            if (int(frame_id) + 1) % 10 == 0:
                trans_info['test']['frames'].append(info)
            else:
                trans_info['train']['frames'].append(info)

        for split in ['train', 'test', 'all']:
            savedir = os.path.join(args.output_path, category + "_" + scene)
            with open(os.path.join(savedir, f'transforms_{split}.json'), 'w', encoding="utf-8") as fp:
                json.dump(trans_info[split], fp, indent=4)
