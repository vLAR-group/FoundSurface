import cv2
import numpy as np
from argparse import ArgumentParser
import os
from tqdm import tqdm
import json
from PIL import Image
import decimal
import quaternion
import math
from scipy import interpolate

def value_to_decimal(value, decimal_places):
    decimal.getcontext().rounding = decimal.ROUND_HALF_UP
    return decimal.Decimal(str(float(value))).quantize(decimal.Decimal('1e-{}'.format(decimal_places)))

def get_up_vectors(pose_device_to_world):
    return np.matmul(pose_device_to_world, np.array([[0.0], [-1.0], [0.0], [0.0]]))


def get_right_vectors(pose_device_to_world):
    return np.matmul(pose_device_to_world, np.array([[1.0], [0.0], [0.0], [0.0]]))


def read_traj(traj_path):
    quaternions = []
    poses = []
    timestamps = []
    poses_p_to_w = []
    with open(traj_path) as f:
        traj_lines = f.readlines()
        for line in traj_lines:
            tokens = line.split()
            assert len(tokens) == 7
            traj_timestamp = float(tokens[0])

            timestamps_decimal_value = value_to_decimal(traj_timestamp, 3)
            timestamps.append(float(timestamps_decimal_value))

            angle_axis = [float(tokens[1]), float(tokens[2]), float(tokens[3])]
            r_w_to_p, _ = cv2.Rodrigues(np.asarray(angle_axis))
            t_w_to_p = np.asarray([float(tokens[4]), float(tokens[5]), float(tokens[6])])

            pose_w_to_p = np.eye(4)
            pose_w_to_p[:3, :3] = r_w_to_p
            pose_w_to_p[:3, 3] = t_w_to_p

            pose_p_to_w = np.linalg.inv(pose_w_to_p)

            r_p_to_w_as_quat = quaternion.from_rotation_matrix(pose_p_to_w[:3, :3])
            t_p_to_w = pose_p_to_w[:3, 3]
            poses_p_to_w.append(pose_p_to_w)
            poses.append(t_p_to_w)
            quaternions.append(r_p_to_w_as_quat)
    return timestamps, poses, quaternions, poses_p_to_w


def find_scene_orientation(poses_cam_to_world):
    if len(poses_cam_to_world) > 0:
        up_vector = sum(get_up_vectors(p) for p in poses_cam_to_world) / len(poses_cam_to_world)
        right_vector = sum(get_right_vectors(p) for p in poses_cam_to_world) / len(poses_cam_to_world)
        up_world = np.array([[0.0], [0.0], [1.0], [0.0]])
    else:
        up_vector = np.array([[0.0], [-1.0], [0.0], [0.0]])
        right_vector = np.array([[1.0], [0.0], [0.0], [0.0]])
        up_world = np.array([[0.0], [0.0], [1.0], [0.0]])

    device_up_to_world_up_angle = np.arccos(np.clip(np.dot(np.transpose(up_world),
                                                           up_vector), -1.0, 1.0)).item() * 180.0 / np.pi
    device_right_to_world_up_angle = np.arccos(np.clip(np.dot(np.transpose(up_world),
                                                              right_vector), -1.0, 1.0)).item() * 180.0 / np.pi

    up_closest_to_90 = abs(device_up_to_world_up_angle - 90.0) < abs(device_right_to_world_up_angle - 90.0)
    if up_closest_to_90:
        assert abs(device_up_to_world_up_angle - 90.0) < 45.0
        if device_right_to_world_up_angle > 90.0:
            sky_direction_scene = 'LEFT'
            cam_to_rotated_q = quaternion.from_rotation_vector([0.0, 0.0, math.pi / 2.0])
        else:
            sky_direction_scene = 'RIGHT'
            cam_to_rotated_q = quaternion.from_rotation_vector([0.0, 0.0, -math.pi / 2.0])
    else:
        assert abs(device_right_to_world_up_angle - 90.0) < 45.0
        if device_up_to_world_up_angle > 90.0:
            sky_direction_scene = 'DOWN'
            cam_to_rotated_q = quaternion.from_rotation_vector([0.0, 0.0, math.pi])
        else:
            sky_direction_scene = 'UP'
            cam_to_rotated_q = quaternion.quaternion(1, 0, 0, 0)
    cam_to_rotated = np.eye(4)
    cam_to_rotated[:3, :3] = quaternion.as_rotation_matrix(cam_to_rotated_q)
    rotated_to_cam = np.linalg.inv(cam_to_rotated)
    return sky_direction_scene, rotated_to_cam

depth_scale = 1000.0
parser = ArgumentParser()
parser.add_argument("--dataset_path", type=str, default="/path/to/ARKitScenes/download/ar_raw_all/raw/Training")
parser.add_argument("--selected_path", type=str, default="/path/to/ARKitScenes/arkitscenes_pairs/Training")
parser.add_argument("--output_path", type=str, default="/path/to/ARKitScenes/ARKitScenes_nerf")
args = parser.parse_args()

output_path = args.output_path
scene_lists = os.listdir(args.selected_path)
scene_lists.sort()
scene_lists = scene_lists[:1]
for scene in tqdm(scene_lists):
    scene_dir = os.path.join(args.dataset_path, scene)
    if not os.path.exists(scene_dir):
        print(f"Scene directory not found for scene {scene}, skipping.")
        continue
    depth_dir = os.path.join(scene_dir, 'lowres_depth')
    rgb_dir = os.path.join(scene_dir, 'vga_wide')
    intrinsics_dir = os.path.join(scene_dir, 'vga_wide_intrinsics')
    traj_path = os.path.join(scene_dir, 'lowres_wide.traj')
    img_files = sorted(os.listdir(rgb_dir))
    img_files = [c for idx, c in enumerate(img_files) if idx % 5 == 0]

    timestamps, poses, quaternions, poses_cam_to_world = read_traj(traj_path)
    poses = np.array(poses)
    quaternions = np.array(quaternions, dtype=np.quaternion)
    quaternions = quaternion.unflip_rotors(quaternions)

    sky_direction_scene, rotated_to_cam = find_scene_orientation(poses_cam_to_world)
    out_rgb_dir = os.path.join(args.output_path, scene, 'input')
    out_depth_dir = os.path.join(args.output_path, scene, 'depth')
    os.makedirs(out_rgb_dir, exist_ok=True)
    os.makedirs(out_depth_dir, exist_ok=True)

    all_images = sorted(
        [
            (basename, basename.split(".png")[0].split("_")[1])
            for basename in img_files
        ],
        key=lambda x: float(x[1]),
    )
    selected_images = []
    timestamps_selected = []
    timestamps = np.array(timestamps)
    timestamp_min = timestamps.min()
    timestamp_max = timestamps.max()
    for basename, frame_id in all_images:
        frame_id = float(frame_id)
        if frame_id < timestamp_min or frame_id > timestamp_max:
            continue
        selected_images.append((basename, frame_id))
        timestamps_selected.append(frame_id)

    timestamps_selected = np.array(timestamps_selected)
    spline = interpolate.interp1d(timestamps, poses, kind="linear", axis=0)
    interpolated_rotations = quaternion.squad(
        quaternions, timestamps, timestamps_selected
    )
    interpolated_positions = spline(timestamps_selected)


    trans_info = {'train': {"depth_scale": depth_scale, 'frames': []},
        'test': { "depth_scale": depth_scale,  'frames': []},
        'all': {"depth_scale": depth_scale, 'frames': []},}
    skip = 0
    need_search =  0
    skip_depth = 0
    need_search_depth = 0
    for i, (basename, frame_id) in enumerate(selected_images):
        vga_wide_path = os.path.join(rgb_dir, scene + f"_{frame_id:.3f}.png")


        depth_path = os.path.join(depth_dir, scene + f"_{frame_id:.3f}.png")
        search_interval = int(0.001 / 0.001)
        for timestamp in range(-search_interval, search_interval + 1):
            if os.path.exists(depth_path):
                break
            need_search_depth += 1
            depth_path = os.path.join(
                depth_dir,
                scene + f"_{float(frame_id) + timestamp * 0.001:.3f}.png"
            )
        if not os.path.exists(depth_path):
            skip_depth += 1
            continue


        intrinsic_fn = os.path.join(intrinsics_dir, f"{scene}_{frame_id:.3f}.pincam")
        search_interval = int(0.1 / 0.001)
        for timestamp in range(-search_interval, search_interval + 1):
            if os.path.exists(intrinsic_fn):
                break
            need_search += 1
            intrinsic_fn = os.path.join(
                intrinsics_dir,
                f"{scene}_{float(frame_id) + timestamp * 0.001:.3f}.pincam",
            )
        if not os.path.exists(intrinsic_fn):
            skip += 1
            continue

        w, h, fx, fy, cx, cy = np.loadtxt(intrinsic_fn)

        if sky_direction_scene == "RIGHT" or sky_direction_scene == "LEFT":
            w, h = h, w
            fx, fy = fy, fx
            cx, cy = cy, cx

        pose = np.eye(4)
        pose[:3, :3] = quaternion.as_rotation_matrix(interpolated_rotations[i])
        pose[:3, 3] = interpolated_positions[i]
        pose = pose @ rotated_to_cam

        img = Image.open(vga_wide_path)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if sky_direction_scene == "RIGHT":
            try:
                img = img.transpose(Image.Transpose.ROTATE_90)
            except Exception:
                img = img.transpose(Image.ROTATE_90)
            depth = cv2.rotate(depth, cv2.ROTATE_90_COUNTERCLOCKWISE)

        elif sky_direction_scene == "LEFT":
            try:
                img = img.transpose(Image.Transpose.ROTATE_270)
            except Exception:
                img = img.transpose(Image.ROTATE_270)
            depth = cv2.rotate(depth, cv2.ROTATE_90_CLOCKWISE)

        elif sky_direction_scene == "DOWN":
            try:
                img = img.transpose(Image.Transpose.ROTATE_180)
            except Exception:
                img = img.transpose(Image.ROTATE_180)
            depth = cv2.rotate(depth, cv2.ROTATE_180)


        W, H = img.size
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST_EXACT)
        cv2.imwrite(os.path.join(out_depth_dir, f"{basename}"), depth)
        img.save(os.path.join(out_rgb_dir, f"{basename}"))

        fx, fy, cx, cy = fx , fy , cx , cy
        fovx, fovy = 2 * np.arctan(w / (2 * fx)), 2 * np.arctan(h / (2 * fy))
        pose[:3, 1:3] *= -1
        info = {'file_path': f'input/{basename}', 'depth_file_path': f'depth/{basename}', "fx": fx, "fy": fy, "cx": cx, "cy": cy, 'transform_matrix': pose.tolist(),
                'camera_angle_x': fovx, 'camera_angle_y': fovy}
        trans_info['all']['frames'].append(info)
        if (i + 1) % 10 == 0:
            trans_info['test']['frames'].append(info)
        else:
            trans_info['train']['frames'].append(info)

    for split in ['train', 'test', 'all']:
        savedir = os.path.join(args.output_path, scene)
        with open(os.path.join(savedir, f'transforms_{split}.json'), 'w', encoding="utf-8") as fp:
            json.dump(trans_info[split], fp, indent=4)

    with open(os.path.join(args.output_path, "skip_info.txt"), 'a') as f:
        f.write(f"{scene}: skipped {skip} frames, needed to search for intrinsics for {need_search} frames.\n")
        f.write(f"Skipped frames: {skip}, Searched frames: {need_search}\n")
        f.write(f"Skipped depth frames: {skip_depth}, Searched depth frames: {need_search_depth}\n")
        f.write("\n")
