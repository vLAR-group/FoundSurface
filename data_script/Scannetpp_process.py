import os
from argparse import ArgumentParser
from tqdm import tqdm
import json
import numpy as np
import pyrender
import trimesh
from PIL import Image
from utils import load_sfm, undistort_images, colmap_to_opencv_intrinsics, rescale_image_depthmap, opencv_to_colmap_intrinsics

os.environ['PYOPENGL_PLATFORM'] = 'egl'

OPENGL_TO_OPENCV = np.float32([[1, 0, 0, 0],
                               [0, -1, 0, 0],
                               [0, 0, -1, 0],
                               [0, 0, 0, 1]])


parser = ArgumentParser()
parser.add_argument("--dataset_path", type=str, default="/path/to/Scannetpp/val_data")
parser.add_argument("--output_path", type=str, default="/path/to/Scannetpp/Scannetpp_nerf")
args = parser.parse_args()

znear = 0.05
zfar = 20.0
scene_lists = os.listdir(args.dataset_path)
scene_lists.sort()
scene_lists = scene_lists[:1]
renderer = pyrender.OffscreenRenderer(0, 0)
depth_scale = 1000.0
for scene in tqdm(scene_lists):
    scene_path = os.path.join(args.dataset_path, scene)
    rgb_path = os.path.join(scene_path, "iphone/rgb")
    scan_path = os.path.join(scene_path, "scans")
    sfm_dir_iphone = os.path.join(scene_path, 'iphone/colmap')
    mask_dir_iphone = os.path.join(scene_path, 'iphone/rgb_masks')

    out_rgb_dir = os.path.join(args.output_path, scene, 'input')
    out_depth_dir = os.path.join(args.output_path, scene, 'depth')
    os.makedirs(out_rgb_dir, exist_ok=True)
    os.makedirs(out_depth_dir, exist_ok=True)

    ply_path = os.path.join(scan_path, 'mesh_aligned_0.05.ply')

    with open(ply_path, 'rb') as f:
        mesh_kwargs = trimesh.exchange.ply.load_ply(f)
    mesh_scene = trimesh.Trimesh(**mesh_kwargs)
    mesh = pyrender.Mesh.from_trimesh(mesh_scene, smooth=False)
    pyrender_scene = pyrender.Scene()
    pyrender_scene.add(mesh)

    img_idx_iphone, img_infos_iphone = load_sfm(
        sfm_dir_iphone, cam_type='iphone')

    trans_info = {'train': {"depth_scale": depth_scale, 'frames': []},
    'test': { "depth_scale": depth_scale, 'frames': []},
    'all': {"depth_scale": depth_scale, 'frames': []},}


    for rgb_name, idx in img_idx_iphone.items():
        img_info = img_infos_iphone[idx]
        rgb_img_path = os.path.join(rgb_path, img_info['path'])
        rgb = np.array(Image.open(rgb_img_path))
        mask = np.array(Image.open(os.path.join(mask_dir_iphone, img_info['path'][:-3] + 'png')))

        _, _, K, rgb, mask = undistort_images(img_info['intrinsics'], rgb, mask)

        intrinsics = colmap_to_opencv_intrinsics(K)
        image, mask, intrinsics = rescale_image_depthmap(
        rgb, mask, intrinsics, (960, 720))

        W, H = image.size
        intrinsics = opencv_to_colmap_intrinsics(intrinsics)

        renderer.viewport_width, renderer.viewport_height = W, H
        fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
        camera = pyrender.camera.IntrinsicsCamera(fx, fy, cx, cy, znear=znear, zfar=zfar)
        camera_node = pyrender_scene.add(camera, pose=img_info['cam_to_world'] @ OPENGL_TO_OPENCV)

        depth = renderer.render(pyrender_scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
        pyrender_scene.remove_node(camera_node)


        depth_outpath = os.path.join(out_depth_dir, img_info['path'][:-3] + 'png')
        depth = (depth * 1000).astype('uint16')
        depth_mask = (mask < 255)
        depth[depth_mask] = 0
        Image.fromarray(depth).save(depth_outpath)

        rgb_outpath = os.path.join(out_rgb_dir, img_info['path'][:-3] + 'jpg')
        image.save(rgb_outpath)

        fovx = 2 * np.arctan(cx / fx)
        fovy = 2 * np.arctan(cy / fy)
        pose = img_info['cam_to_world'].copy()
        pose[:3, 1:3] *= -1
        info = {'file_path': f"input/{img_info['path'][:-3]}"+ 'jpg', 'depth_file_path': f"depth/{img_info['path'][:-3]}"+ 'png', 'cx': cx, 'cy': cy, 'fx': fx,
                'fy': fy, 'transform_matrix': pose.tolist(), 'camera_angle_x': fovx, 'camera_angle_y': fovy}

        trans_info['all']['frames'].append(info)
        if (int(idx) + 1) % 10 == 0:
            trans_info['test']['frames'].append(info)
        else:
            trans_info['train']['frames'].append(info)

    for split in ['train', 'test', 'all']:
        savedir = os.path.join(args.output_path, scene)
        with open(os.path.join(savedir, f'transforms_{split}.json'), 'w', encoding="utf-8") as fp:
            json.dump(trans_info[split], fp, indent=4)
