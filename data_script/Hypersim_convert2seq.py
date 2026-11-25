import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import json
from argparse import ArgumentParser
import cv2
from tqdm import tqdm
import torch
import networkx as nx
import shutil
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def convert_dpt2pt(depth, K, pose):
    H, W = depth.shape[0], depth.shape[1]
    x, y = np.meshgrid(np.arange(W, dtype=np.float32),
                            np.arange(H, dtype=np.float32))
    cam_pt = np.stack([(x - K[0, 2]) * depth / K[0, 0], (y - K[1, 2]) * depth / K[1, 1], K[2, 2] * np.ones_like(x) * depth, np.ones_like(x)], -1)
    pt = (pose[None, None, :, :]@cam_pt[:, :, :, None]).squeeze()[:, :, :3]
    return pt

def reconstruct_sequence_greedy(
                                overlap_matrix: np.ndarray,
                                max_neighbors: int = 5):
    """
    Reconstruct an image sequence greedily.

    Args:
        overlap_matrix: pairwise overlap ratios.
        max_neighbors: maximum number of neighbours per node.

    Returns:
        The reconstructed image sequence.
    """

    img_names = list(range(overlap_matrix.shape[0]))
    n_images = len(img_names)

    if n_images < 2:
        return img_names

    G = nx.Graph()
    for i in range(n_images):
        G.add_node(i, name=img_names[i])

    for i in range(n_images):
        overlaps = overlap_matrix[i]
        neighbor_indices = np.argsort(overlaps)[::-1][1:max_neighbors+1]

        for j in neighbor_indices:
            if overlaps[j] > 0.3:
                weight = 1.0 / (overlaps[j] + 1e-6)
                G.add_edge(i, j, weight=weight, overlap=overlaps[j])

    if not nx.is_connected(G):
        print("warning: graph is disconnected; increase max_neighbors or lower overlap_threshold")
        mst = nx.minimum_spanning_tree(G)
        for edge in mst.edges():
            if not G.has_edge(edge[0], edge[1]):
                G.add_edge(edge[0], edge[1], weight=1.0, overlap=0.1)

    avg_overlap = np.sum(overlap_matrix, axis=1)
    start_idx = np.argmax(avg_overlap)

    sequence_indices = [start_idx]
    visited = set([start_idx])

    while len(visited) < n_images:
        current = sequence_indices[-1]

        neighbors = list(G.neighbors(current))
        unvisited_neighbors = [n for n in neighbors if n not in visited]

        if not unvisited_neighbors:
            unvisited = [i for i in range(n_images) if i not in visited]
            if not unvisited:
                break

            best_overlap = -1
            best_node = -1
            for node in unvisited:
                if overlap_matrix[current, node] > best_overlap:
                    best_overlap = overlap_matrix[current, node]
                    best_node = node

            if best_node != -1:
                sequence_indices.append(best_node)
                visited.add(best_node)
            continue

        best_neighbor = -1
        best_overlap = -1
        for neighbor in unvisited_neighbors:
            overlap = G[current][neighbor].get('overlap', 0)
            if overlap > best_overlap:
                best_overlap = overlap
                best_neighbor = neighbor

        if best_neighbor != -1:
            sequence_indices.append(best_neighbor)
            visited.add(best_neighbor)
        else:
            break

    sequence = [img_names[idx] for idx in sequence_indices]
    return sequence

parser = ArgumentParser()
parser.add_argument("--dataset_path", type=str, default="/path/to/Datasets/Hypersim/Hypersim_nerf_new")
parser.add_argument("--output_path", type=str, default="/path/to/Datasets/Hypersim/Hypersim_all_Seq_nerf")
args = parser.parse_args()

MAX_DEPTH = 10000.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scene_lists = os.listdir(args.dataset_path)
scene_lists.sort()
for scene in tqdm(scene_lists):
    print(f"Processing scene: {scene}")
    scene_path = os.path.join(args.dataset_path, scene)
    out_scene_path = os.path.join(args.output_path, scene)

    with open(os.path.join(scene_path, "transforms_all.json"), 'r') as f:
        data = json.load(f)
    depth_scale = data["depth_scale"]
    if "fx" in data:
        fx, fy, cx, cy = data['fx'], data['fy'], data['cx'], data['cy']
    frames = data["frames"]
    per_view_pts_list, w2c_list, K_list = [], [], []
    depth_lsit = []
    for frame in frames:
        img_path = os.path.join(scene_path, frame["file_path"])
        depth_path = os.path.join(scene_path, frame["depth_file_path"])
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / depth_scale
        depth = cv2.resize(depth, (depth.shape[1]//2, depth.shape[0]//2), interpolation=cv2.INTER_NEAREST)
        if "fx" not in data:
            fx, fy, cx, cy = frame["fx"], frame["fy"], frame["cx"], frame["cy"]
        fx, fy, cx, cy = fx/2, fy/2, cx/2, cy/2
        pose = np.array(frame["transform_matrix"], dtype=np.float32)
        pose[:3, 1:3] *= -1
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        view_pts = convert_dpt2pt(depth, K, pose)
        per_view_pts_list.append(view_pts)
        extrinsics = np.linalg.inv(pose)
        w2c_list.append(extrinsics)
        K_list.append(K)
        depth_lsit.append(depth)
    depth_list = np.stack(depth_lsit, axis=0)
    w2c_list = np.stack(w2c_list, axis=0)
    K_list = np.stack(K_list, axis=0)
    per_view_pts_list = np.stack(per_view_pts_list, axis=0)
    per_view_pts_list = torch.from_numpy(per_view_pts_list).float().to(device)
    w2c_list = torch.from_numpy(w2c_list).float().to(device)
    K_list = torch.from_numpy(K_list).float().to(device)
    depth_lsit = torch.from_numpy(depth_list).float().to(device)
    overlap_ratio = torch.zeros((len(frames), len(frames)), dtype=torch.float32, device=device)
    for i in range(len(frames)):
        K = K_list[i]
        view_pts = per_view_pts_list[i]
        target_H, target_W = view_pts.shape[0], view_pts.shape[1]
        view_pts = view_pts.reshape(-1, 3)
        view_pts_homo = torch.cat([view_pts, torch.ones_like(view_pts[:, :1])], dim=-1)
        points_per_cam = view_pts_homo @ w2c_list.transpose(-1, -2)[...,:3]
        z_coords = points_per_cam[..., 2]
        z_mask = z_coords > 0.1

        uv_coords = torch.zeros_like(points_per_cam[..., :3])
        num_frames = points_per_cam.shape[0]
        step_size = 500
        for start in range(0, num_frames, step_size):
            end = min(start + step_size, num_frames)
            batch_points = points_per_cam[start:end, :, :3]
            batch_result = (K @ batch_points.transpose(-1, -2)).transpose(-1, -2)
            uv_coords[start:end] = batch_result

        uv_coords = uv_coords / (points_per_cam[..., 2:] + 1e-8)

        uv_coords = torch.round(uv_coords).long()
        frustum_mask = (uv_coords[..., 0] >= 0) & (uv_coords[..., 0] < target_W) & (uv_coords[..., 1] >= 0) & (uv_coords[..., 1] < target_H)

        frustum_mask = frustum_mask & z_mask
        uv_coords[..., 0][~frustum_mask], uv_coords[..., 1][~frustum_mask], z_coords[~frustum_mask] = 0, 0, MAX_DEPTH

        pixel_flatten = uv_coords[..., 1] * target_W + uv_coords[..., 0]
        per_view_min_depth = torch.full((len(frames), target_H * target_W), MAX_DEPTH, device=device)
        per_view_min_depth = per_view_min_depth.scatter_reduce_(-1, pixel_flatten.reshape(len(frames), -1), z_coords.reshape(len(frames), -1), reduce="amin")
        per_view_min_depth = per_view_min_depth.reshape(len(frames), target_H * target_W)
        error = torch.abs(per_view_min_depth - depth_lsit.reshape(len(frames), target_H * target_W)) < 0.025
        valid_mask = (depth_lsit > 0).reshape(len(frames), target_H * target_W )
        overlap_ratio[i] = (error * valid_mask).sum(-1) / (valid_mask.sum(-1) + 1e-8)

    overlap_ratio = (overlap_ratio + overlap_ratio.T)/2

    if overlap_ratio.max()<0.05:
        print(f"Warning: Low overlap ratio in scene {scene}, max overlap: {overlap_ratio.max().item():.4f}")
        continue
    seq_idx = reconstruct_sequence_greedy(overlap_ratio.cpu().numpy())
    trans_info = {'train': {"depth_scale": depth_scale, 'frames': []},
            'test': {"depth_scale": depth_scale, 'frames': []},
            'all': { "depth_scale": depth_scale, 'frames': []},}
    for idx, frame_idx in enumerate(seq_idx):
        frame = frames[frame_idx]
        img_path = os.path.join(scene_path, frame["file_path"])
        depth_path = os.path.join(scene_path, frame["depth_file_path"])
        out_rgb_dir = os.path.join(out_scene_path, 'input')
        out_depth_dir = os.path.join(out_scene_path, 'depth')
        os.makedirs(out_rgb_dir, exist_ok=True)
        os.makedirs(out_depth_dir, exist_ok=True)
        rgb_ext = os.path.splitext(frame["file_path"])[-1]
        depth_ext = os.path.splitext(frame["depth_file_path"])[-1]
        shutil.copy(img_path, os.path.join(out_rgb_dir, f"{idx:06d}{rgb_ext}"))
        shutil.copy(depth_path, os.path.join(out_depth_dir, f"{idx:06d}{depth_ext}"))
        fx, fy, cx, cy = data["fx"], data["fy"], data["cx"], data["cy"]

        pose = np.array(frame["transform_matrix"], dtype=np.float32)
        fovx = 2 * np.arctan(cx / fx)
        info = {'file_path': f'input/{idx:06d}{rgb_ext}', 'transform_matrix': pose.tolist(), "depth_file_path": f'depth/{idx:06d}{depth_ext}',
                'fx': fx, 'fy': fy, 'cx':cx, 'cy':cy, "camera_angle_x":fovx}

        trans_info['all']['frames'].append(info)
        if (int(idx) + 1) % 10 == 0:
            trans_info['test']['frames'].append(info)
        else:
            trans_info['train']['frames'].append(info)

    for split in ['train', 'test', 'all']:
        savedir = os.path.join(out_scene_path)
        with open(os.path.join(savedir, f'transforms_{split}.json'), 'w', encoding="utf-8") as fp:
            json.dump(trans_info[split], fp, indent=4)
