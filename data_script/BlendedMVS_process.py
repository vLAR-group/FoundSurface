import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm
import json
import re
from argparse import ArgumentParser
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

def load_cam(file, interval_scale=1):
    """ read camera txt file """
    cam = np.zeros((2, 4, 4))
    words = file.read().split()
    for i in range(0, 4):
        for j in range(0, 4):
            extrinsic_index = 4 * i + j + 1
            cam[0][i][j] = words[extrinsic_index]

    for i in range(0, 3):
        for j in range(0, 3):
            intrinsic_index = 3 * i + j + 18
            cam[1][i][j] = words[intrinsic_index]

    if len(words) == 29:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = 777
        cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * cam[1][3][2]
    elif len(words) == 30:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = words[29]
        cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * cam[1][3][2]
    elif len(words) == 31:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = words[29]
        cam[1][3][3] = words[30]
    else:
        cam[1][3][0] = 0
        cam[1][3][1] = 0
        cam[1][3][2] = 0
        cam[1][3][3] = 0

    return cam


def load_pfm(file):
    color = None
    width = None
    height = None
    scale = None
    data_type = None
    header = file.readline().decode('UTF-8').rstrip()

    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')
    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('UTF-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')
    scale = float((file.readline()).decode('UTF-8').rstrip())
    if scale < 0:
        data_type = '<f'
    else:
        data_type = '>f'
    data_string = file.read()
    data = np.frombuffer(data_string, data_type)
    shape = (height, width, 3) if color else (height, width)
    data = np.reshape(data, shape)
    data = cv2.flip(data, 0)
    return data


parser = ArgumentParser()
parser.add_argument("--dataset_path", type=str, default="/path/to/BlendedMVS/BlendedMVS")
parser.add_argument("--output_path", type=str, default="/path/to/BlendedMVS/BlendedMVS_nerf")

args = parser.parse_args()
data_dir = args.dataset_path

scene_list = os.listdir(data_dir)
scene_list.sort()
out_dir = args.output_path
for scene in tqdm(scene_list):
    scene_dir = os.path.join(data_dir, scene)
    depth_image_path = os.path.join(scene_dir, "rendered_depth_maps")
    rgb_image_path = os.path.join(scene_dir, "blended_images")
    cams_path = os.path.join(scene_dir, "cams")
    depth_image_list = os.listdir(depth_image_path)
    depth_image_list.sort()
    trans_info = {'train': {"depth_scale": 1.0, 'frames': []},
        'test': {"depth_scale": 1.0, 'frames': []},
        'all': { "depth_scale": 1.0, 'frames': []},}
    for idx, depth_image_name in enumerate(depth_image_list):
        depth_path = os.path.join(depth_image_path, depth_image_name)
        depth_name = depth_image_name.replace(".pfm", ".exr")
        with open(depth_path, 'rb') as file:
            depth = load_pfm(file)
        cam_path = depth_path.replace("rendered_depth_maps", "cams").replace(".pfm", "_cam.txt")
        with open(cam_path, 'r') as file:
            cam = load_cam(file)
        rgb_path = depth_path.replace("rendered_depth_maps", "blended_images").replace(".pfm", ".jpg")


        out_rgb_dir = os.path.join(out_dir, scene, 'input')
        out_depth_dir = os.path.join(out_dir, scene, 'depth')
        os.makedirs(out_rgb_dir, exist_ok=True)
        os.makedirs(out_depth_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_depth_dir, depth_name), depth)
        rgb_name = os.path.basename(rgb_path)
        shutil.copy2(rgb_path, os.path.join(out_rgb_dir, rgb_name))
        extrinsics = cam[0]
        pose = np.linalg.inv(extrinsics)

        fx, fy, cx, cy = cam[1][0][0], cam[1][1][1], cam[1][0][2], cam[1][1][2]
        fovx = 2 * np.arctan(cx / fx)


        pose[:3, 1:3] *= -1

        info = {'file_path': f'input/{rgb_name}', 'transform_matrix': pose.tolist(), "depth_file_path": f'depth/{depth_name}',
                'fx': fx, 'fy': fy, 'cx':cx, 'cy':cy, "camera_angle_x":fovx}

        trans_info['all']['frames'].append(info)
        if (int(idx) + 1) % 10 == 0:
            trans_info['test']['frames'].append(info)
        else:
            trans_info['train']['frames'].append(info)

    for split in ['train', 'test', 'all']:
        savedir = os.path.join(out_dir, scene)
        with open(os.path.join(savedir, f'transforms_{split}.json'), 'w', encoding="utf-8") as fp:
            json.dump(trans_info[split], fp, indent=4)
