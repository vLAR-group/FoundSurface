"""Benchmark per-view inference latency, FPS and GPU memory."""
import torch
from vggt.models.vggt_backbone_pointmap_light import VGGT
import numpy as np
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import closed_form_inverse_se3
from utils.rays import compute_rays, get_posed_input, cam_coord2sph, cam_sph2coord, get_surface_normal
from utils.general_utils import torch_unique, get_patch_index
from utils.graphics_utils import create_pt3d
from utils.vis_utils import to_normalmap
from argparse import ArgumentParser
from raydf.RaySurfDNet import CondNet, RenderDPTNet, SceneAggNetMLP
from dataset import build_dataset
import os
from rayletdf.RaySurfDNet import RaySurfDNetMLPVGGTReduceMeanFuse
from rayletdf.spconv_unet import SpUNetBase
import math
from tqdm import tqdm
import roma
from torch.nn.utils.rnn import pad_sequence
import imageio
import einops
import torch_knnquery
import time
MAX_DEPTH = 20.0
MIN_DEPTH = 0.01
parser = ArgumentParser(description="Testing script parameters")
parser.add_argument('--test_dataset', required=True, type=str, help="testing set")
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--num_workers", default=8, type=int)
parser.add_argument("--iter_step", default=10000, type=int)
parser.add_argument("--method_name", type=str, default="foundsurface")
parser.add_argument("--split", type=str, default="test")
parser.add_argument("--model_path", required=True, type=str, help="stage-1 checkpoint")
parser.add_argument("--raylet_path", required=True, type=str, help="stage-2 raylet checkpoint")
parser.add_argument("--ray_encode_type", required=False, type=str, default="default_plucker")
parser.add_argument("--neighbor_num", default=2, type=int)
parser.add_argument("--dec_embed_dim", default=512, type=int)
parser.add_argument("--knn_num", default=5, type=int)
parser.add_argument("--topk", default=1, type=int)
parser.add_argument("--voxel_size", default=0.025, type=float)
parser.add_argument("--surface_patch_size", default=10, type=int)
parser.add_argument("--normal", action='store_true', help="Whether to derive normal maps from RayletDF")
parser.add_argument("--save_normal", action='store_true', help="Whether to save normal maps from RayletDF")
parser.add_argument("--ref2target_overlap_thresh", default=0.3, type=float)
args = parser.parse_args()

device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

model = VGGT()
vggt_ckpt = torch.load("./vggt/vggt_model.pt")
filter_ckpt = {k:v for k, v in vggt_ckpt.items() if "aggregator" in k or "camera_head" in k or "point_head" in k}
model.load_state_dict(filter_ckpt)
model.to(device)

test_dataloader = build_dataset(args.test_dataset, args.batch_size, args.num_workers, test=True)
rec_view_num = test_dataloader.dataset.rec_views
print(f"Number of reconstruction views: {rec_view_num}")

scene_rec_network = CondNet(dec_embed_dim=args.dec_embed_dim).to(device)
render_net = RenderDPTNet(dec_embed_dim=args.dec_embed_dim, dec_num_heads=4, dec_depth=4, mlp_ratio=4, rope_freq=100).to(device)
aggregate_net = SceneAggNetMLP(dec_embed_dim=2048).to(device)

ckpt = torch.load(args.model_path, weights_only=False)
scene_rec_network.load_state_dict(ckpt["rec_net"])
render_net.load_state_dict(ckpt["render_net"])
aggregate_net.load_state_dict(ckpt['aggregate_net'])

scene_rec_network.eval()
render_net.eval()
aggregate_net.eval()

raylet_ckpt = torch.load(args.raylet_path, weights_only=False)
raylet_render_net = RaySurfDNetMLPVGGTReduceMeanFuse(map_layer_dim=518+7*args.knn_num, W=512, out_dim=1, geo_token_dim=args.dec_embed_dim).to(device)
local_aggregate_net = SceneAggNetMLP(dec_embed_dim=args.dec_embed_dim).to(device)
point_transformer  = SpUNetBase(args.dec_embed_dim, args.dec_embed_dim, channels=(256,256,256,256), layers=(2,2,2,2),).to(device)

raylet_render_net.load_state_dict(raylet_ckpt['raylet_render_net'])
local_aggregate_net.load_state_dict(raylet_ckpt['local_aggregate_net'])
point_transformer.load_state_dict(raylet_ckpt['point_transformer'])
raylet_render_net.eval()
local_aggregate_net.eval()
point_transformer.eval()

progress_bar = tqdm(range(0, len(test_dataloader)), desc="Testing progress")
time_lists = []
peak_memory_list = []
reserved_memory_list = []
for idx, (rec_data, test_data, scene_info) in enumerate(test_dataloader):
    torch.cuda.reset_peak_memory_stats(device)
    scene_name, dataset_name = scene_info["scene_name"][0], scene_info["dataset_name"][0]
    scene_dir = os.path.join(os.path.dirname(args.model_path), dataset_name + "_Raylet_fit_top{}_view{}".format(args.topk, rec_view_num), args.split, scene_name)
    os.makedirs(scene_dir, exist_ok = True)
    ades, rmses, rel_ades, sq_rel_ades = [], [], [], []
    a1s, a2s, a3s = [], [], []
    image_names = []
    full_dict = {args.method_name:{}}
    per_view_dict = {args.method_name:{}}

    images = rec_data["images"].float()
    gt_c2ws = rec_data["c2ws"].float()
    gt_Ks = rec_data["Ks"].float()
    gt_depths = rec_data["depths"].float()

    test_gt_dists = test_data["ray_dists"].to(device).float()
    test_gt_c2ws = test_data["c2ws"].to(device).float()
    test_gt_Ks = test_data["Ks"].to(device).float()
    test_images = test_data["images"].permute(0, 1, 4, 2, 3).float() / 255
    test_dpts = test_data["depths"].to(device).float()
    test_org_Ks = test_data["org_Ks"].to(device).float()

    with torch.no_grad():
        rec_images, rec_depths, rec_Ks, rec_c2ws = images.to(device), gt_depths.to(device), gt_Ks.to(device), gt_c2ws.to(device)

        rec_masks = rec_depths > 0
        gt_points3d = create_pt3d(rec_depths, rec_Ks, rec_c2ws, torch.eye(4, device=device).unsqueeze(0).repeat(args.batch_size, 1, 1))

        rec_H, rec_W = rec_images.shape[-2:]

        with torch.cuda.amp.autocast(dtype=dtype):
            aggregated_tokens_list, ps_idx, prediction = model(rec_images)
            pose_enc, points, confs = prediction["pose_enc"], prediction["world_points"], prediction["world_points_conf"]

        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, rec_images.shape[-2:])
        b, v, _, _ = extrinsic.shape
        extrinsic = extrinsic.reshape(b * v, 3, 4)
        cam_to_world = closed_form_inverse_se3(extrinsic)
        cam_to_world = cam_to_world.reshape(b, v, 4, 4)

        MAX_POINTS_FOR_QUANTILE = 518*392*71
        if confs.numel() > MAX_POINTS_FOR_QUANTILE:
            indices = torch.randperm(confs.numel(), device=confs.device)[:MAX_POINTS_FOR_QUANTILE]
            conf_threshold_reg = torch.quantile(confs.flatten()[indices], 85 / 100.0)
        else:
            conf_threshold_reg = torch.quantile(confs, 85 / 100.0)

        conf_mask_reg = confs >= conf_threshold_reg
        final_mask_pred = rec_masks.squeeze() & conf_mask_reg
        pred_pts_tensor = points[final_mask_pred]
        gt_pts_tensor_icp = gt_points3d[final_mask_pred]

        del gt_points3d, final_mask_pred

        R, t, scene_scale_fit = roma.rigid_points_registration(pred_pts_tensor, gt_pts_tensor_icp, compute_scaling=True)

        transform_fit_gt2vggt = torch.eye(4, device=device).unsqueeze(0).repeat(args.batch_size, 1, 1)
        transform_fit_gt2vggt[:, :3, :3] = R.T
        transform_fit_gt2vggt[:, :3, 3] = -1 * (R.T @ t) / scene_scale_fit
        del gt_pts_tensor_icp, pred_pts_tensor

        geometry_tokens_list = []
        for vggt_token in aggregated_tokens_list:
            geo_token = vggt_token[:, :, ps_idx:, :]
            geometry_tokens_list.append(geo_token)
        del aggregated_tokens_list
        torch.cuda.synchronize()
        prepare_time_start = time.time()
        with torch.cuda.amp.autocast(dtype=dtype):
            geometry_tokens = aggregate_net(geometry_tokens_list)

        with torch.cuda.amp.autocast(dtype=dtype):
            local_geometry_tokens = local_aggregate_net(geometry_tokens_list)
        points = points.reshape(args.batch_size, -1, 3)
        confs = confs.reshape(args.batch_size, -1, 1)
        coords_list, feats_batch, pc_len, grid_coord = [], [], [], []
        local_geometry_tokens = local_geometry_tokens.reshape(args.batch_size, -1, local_geometry_tokens.shape[-1])
        cor_patch_idx = get_patch_index(rec_images.shape[-2], rec_images.shape[-1], 14, 14, rec_view_num, device=device).flatten(0, 1)
        for batch_id in range(args.batch_size):
            per_pc, per_conf = points[batch_id], confs[batch_id]

            originCorner = torch.floor(per_pc.min(0)[0] * (1/args.voxel_size)) * args.voxel_size
            grids = torch.div(per_pc - originCorner, args.voxel_size, rounding_mode="trunc").int()
            unq, inverse_map, _, unique_map = torch_unique(grids, dim=0)
            ds_pts = per_pc[unique_map]
            features = local_geometry_tokens[batch_id][cor_patch_idx[unique_map]]

            coords_list.append(ds_pts)
            feats_batch.append(features)
            pc_len.append(ds_pts.shape[0])
            grid_coord.append(torch.cat([torch.ones_like(unq[:, :1]) * batch_id, unq], dim=-1))

        pt_inputs = {}
        coords_batch = torch.cat(coords_list, dim=0)
        feats_batch = torch.cat(feats_batch, dim=0)
        grid_coord = torch.cat(grid_coord, dim=0)
        pt_inputs["batch"] = grid_coord[..., 0]
        pt_inputs["feat"] = feats_batch.float()
        pt_inputs['grid_coord'] = grid_coord[..., 1:]
        pt_inputs["coord"] = coords_batch

        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            pt_output = point_transformer(pt_inputs)

        counts = torch.bincount(pt_inputs["batch"])
        decomposed_out_fea = list(torch.split(pt_output, counts.tolist()))

        coords_batch = pad_sequence(coords_list, batch_first=True, padding_value=0.0)
        feats_batch = pad_sequence(decomposed_out_fea, batch_first=True, padding_value=0.0).to(dtype)
        pc_len = torch.tensor(pc_len, device=device).long()

        voxel_grid = torch_knnquery.VoxelGrid(
        voxel_size=(0.05, 0.05, 0.05),
        voxel_scale=(2, 2, 2),
        kernel_size=(3, 3, 3),
        max_points_per_voxel=20,
        max_occ_voxels_per_example=610000,
    )

        voxel_grid.set_pointset(
            points=coords_batch,
            actual_num_points_per_example=torch.tensor(ds_pts.shape[0], dtype=torch.int32).to(device)
            )
    torch.cuda.synchronize()
    prepare_time_end = time.time()
    print(f"Prepare time: {prepare_time_end - prepare_time_start:.4f} seconds")
    for test_idx in range(test_gt_dists.shape[1]):
        with torch.no_grad():
            select_c2ws, select_Ks, select_gts = test_gt_c2ws[:, test_idx:test_idx+1], test_gt_Ks[:, test_idx:test_idx+1], test_gt_dists[:, test_idx:test_idx+1]
            select_dpts = test_dpts[:, test_idx:test_idx+1]
            select_org_K = test_org_Ks[:, test_idx:test_idx+1]
            vggt_select_c2ws = select_c2ws.clone()
            vggt_select_c2ws[:, :, :3, 3] = vggt_select_c2ws[:, :, :3, 3] / scene_scale_fit
            vggt_select_c2ws = transform_fit_gt2vggt.unsqueeze(1) @ vggt_select_c2ws

            _, m, _, _ = vggt_select_c2ws.shape
            vggt_select_w2cs = torch.linalg.inv(vggt_select_c2ws).unsqueeze(2)
            start = time.time()
            points_coord = points.reshape(b, 1, v, -1, 3)
            homo_points_coord = torch.cat([points_coord, torch.ones_like(points_coord[..., :1])], dim=-1)
            points_per_cam = (homo_points_coord @ vggt_select_w2cs.transpose(-1, -2))[...,:3]
            z_coords = points_per_cam[..., 2]
            z_mask = z_coords > 0.1
            uv_coords = (select_Ks.unsqueeze(2) @ points_per_cam.transpose(-1, -2)).transpose(-1, -2) / points_per_cam[..., 2:]
            uv_coords = torch.round(uv_coords).long()
            target_W, target_H = select_dpts.shape[-1], select_dpts.shape[-2]
            frustum_mask = (uv_coords[..., 0] >= 0) & (uv_coords[..., 0] < target_W) & (uv_coords[..., 1] >= 0) & (uv_coords[..., 1] < target_H)

            frustum_mask = frustum_mask & z_mask
            uv_coords[..., 0][~frustum_mask], uv_coords[..., 1][~frustum_mask], z_coords[~frustum_mask] = 0, 0, MAX_DEPTH

            pixel_flatten = uv_coords[..., 1] * target_W + uv_coords[..., 0]
            per_view_min_depth = torch.full((b*m*v, target_H * target_W), MAX_DEPTH, device=device)
            per_view_min_depth = per_view_min_depth.scatter_reduce_(-1, pixel_flatten.reshape(b*m*v, -1), z_coords.reshape(b*m*v, -1), reduce="amin")
            per_view_min_depth = per_view_min_depth.reshape(b, m, v, target_H * target_W)
            min_depth = per_view_min_depth.min(dim=2, keepdim=True)[0]

            min_depth = min_depth.reshape(b, m, 1, target_H, target_W)
            patchify_min_depth = einops.rearrange(min_depth, 'b m v (h p1) (w p2) -> b m v (h w) (p1 p2)', p1 = args.surface_patch_size, p2=args.surface_patch_size)
            patchify_per_view_min_depth = einops.rearrange(per_view_min_depth.reshape(b, m, v, target_H, target_W), 'b m v (h p1) (w p2) -> b m v (h w) (p1 p2)', p1 =args.surface_patch_size, p2=args.surface_patch_size)
            patch_valid_mask = (patchify_min_depth > MIN_DEPTH) & (patchify_min_depth < MAX_DEPTH)
            patch_overlap_mask = (patchify_per_view_min_depth - patchify_min_depth).abs() < 0.05
            patch_overlap_ratio = (patch_overlap_mask.float()*patch_valid_mask.float()).sum(-1) / (patch_valid_mask.float().sum(-1) + 1e-8)
            per_view_overlap_mask = patch_overlap_ratio > args.ref2target_overlap_thresh
            per_view_overlap_ratio = per_view_overlap_mask.float().mean(-1)

            mean_overlap_ratio = per_view_overlap_ratio
            _, neighbor_idx = torch.topk(mean_overlap_ratio, k=args.neighbor_num, dim=-1, largest=True)

            B, V, PD1, PD2 = cam_to_world.shape
            _, M, N = neighbor_idx.shape
            _, _, KD1, KD2 = intrinsic.shape
            expand_neighbor_idx = neighbor_idx.flatten(-2).unsqueeze(-1).expand(-1, -1, PD1 * PD2)
            neighbor_c2ws = torch.gather(cam_to_world.flatten(-2), dim=1, index=expand_neighbor_idx).view(B, M, N, PD1, PD2)
            expand_neighbor_idx = neighbor_idx.flatten(-2).unsqueeze(-1).expand(-1, -1, KD1 * KD2)
            neighbor_Ks = torch.gather(intrinsic.flatten(-2), dim=1, index=expand_neighbor_idx).view(B, M, N, KD1, KD2)

            local_neighbor_c2ws = torch.linalg.inv(vggt_select_c2ws).unsqueeze(2) @ neighbor_c2ws

            local_neighbor_c2ws, neighbor_Ks = local_neighbor_c2ws.reshape(B*M, N, PD1, PD2), neighbor_Ks.reshape(B*M, N, KD1, KD2)
            ray_o, ray_d = compute_rays(local_neighbor_c2ws, neighbor_Ks, device=device)
            pose_cond = get_posed_input(ray_o = ray_o, ray_d = ray_d, method = args.ray_encode_type)

            expand_neighbor_idx = neighbor_idx.flatten(-2).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, geometry_tokens.shape[-2], geometry_tokens.shape[-1])
            neighbor_geometry_tokens = torch.gather(geometry_tokens, dim=1, index=expand_neighbor_idx).reshape(B, M, N, geometry_tokens.shape[-2], geometry_tokens.shape[-1])
            neighbor_geometry_tokens = neighbor_geometry_tokens.flatten(0, 1)

            with torch.cuda.amp.autocast(dtype=dtype):
                scene_tokens = scene_rec_network(pose_cond, neighbor_geometry_tokens)

            local_c2w = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).expand(B*M, 1, -1, -1)
            select_Ks = select_Ks.reshape(B*M, 1, select_Ks.shape[-2], select_Ks.shape[-1])
            novel_ray_o, novel_ray_d = compute_rays(local_c2w, select_Ks, h=target_H, w=target_W, device=device)

        if args.normal:
            torch.set_grad_enabled(True)
            sph_coord = cam_coord2sph(novel_ray_d.squeeze().permute(1, 2, 0), normalize=False).reshape(-1, 1, 2)
            sph_coord = sph_coord.requires_grad_(True)
            novel_ray_d = cam_sph2coord(sph_coord).squeeze().reshape(target_H, target_W, 3).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
        else:
            torch.set_grad_enabled(False)

        novel_ray_params = get_posed_input(ray_o = novel_ray_o, ray_d = novel_ray_d, method = args.ray_encode_type)
        with torch.cuda.amp.autocast(dtype=dtype):
            render_dists = render_net(novel_ray_params, scene_tokens, (rec_H, rec_W))

        render_dists = render_dists.transpose(0, 1).squeeze().reshape(-1, 1, 1)

        if select_dpts.shape != select_gts.shape:
            render_dists = render_dists.reshape_as(select_dpts)
            render_dists = torch.nn.functional.interpolate(render_dists, size=select_gts.shape[-2:], mode='nearest').reshape(-1, 1, 1)
        novel_ray_o, novel_ray_d = compute_rays(local_c2w, select_org_K, h=select_gts.shape[-2], w=select_gts.shape[-1], device=device)
        novel_ray_o, novel_ray_d = novel_ray_o.squeeze().reshape(3, -1).transpose(0, 1).unsqueeze(1), novel_ray_d.squeeze().reshape(3, -1).transpose(0, 1).unsqueeze(1)
        novel_ray_o_homo = torch.cat([novel_ray_o, torch.ones_like(novel_ray_o[..., :1])], dim=-1)
        novel_ray_o_vggt = (novel_ray_o_homo @ vggt_select_c2ws.squeeze().transpose(-1, -2))[..., :3]
        novel_ray_d_vggt = novel_ray_d @ vggt_select_c2ws.squeeze()[:3, :3].transpose(-1, -2)
        render_dists = render_dists.squeeze().reshape(-1, 1, 1)

        if args.topk > 1:
            render_dists = torch.tile(render_dists, (1, args.topk, 1))
            sample_delta = torch.randn_like(render_dists[:, :args.topk - 1]) * 0.1
            sample_delta = torch.cat([sample_delta, torch.zeros_like(render_dists[:, 0:1])], dim=1)
            raylet_sample_dists = render_dists + sample_delta
        else:
            raylet_sample_dists = render_dists

        NUM_PIXEL = select_gts.shape[-2] * select_gts.shape[-1]
        iter_num = math.ceil(NUM_PIXEL / args.iter_step)
        out_dist, out_normal = [], []
        for i in range(0, iter_num):
            batch_ray_o = novel_ray_o_vggt[i*args.iter_step : min((i+1)*args.iter_step, NUM_PIXEL)]
            batch_ray_d = novel_ray_d_vggt[i*args.iter_step : min((i+1)*args.iter_step, NUM_PIXEL)]
            batch_render_dists = raylet_sample_dists[i*args.iter_step : min((i+1)*args.iter_step, NUM_PIXEL)]
            hit_points = (batch_ray_o + batch_ray_d * batch_render_dists).unsqueeze(0)

            sparse_knn_inds, _, ray_mask = voxel_grid.query(
                                    raypos=hit_points.contiguous(),
                                    k=args.knn_num,
                                    radius_limit_scale=5,
                                    max_shading_points_per_ray=args.topk
                                    )
            sparse_knn_inds = sparse_knn_inds.long()
            knn_inds = torch.zeros((hit_points.shape[1], args.topk, args.knn_num), dtype=torch.long, device=device)
            knn_inds[ray_mask.squeeze().long()>0] = sparse_knn_inds
            knn_inds[knn_inds==-1] = 0
            knn_inds = knn_inds.unsqueeze(0)

            knn_pts = ds_pts[knn_inds]
            knn_vec = (knn_pts - hit_points.unsqueeze(-2))
            knn_dists = torch.linalg.norm(knn_vec, dim=-1, keepdim=True)
            knn_dir = (knn_vec / knn_dists + 1e-8)
            knn_rel = torch.cat([knn_dir, knn_dists], dim=-1).flatten(-2)

            knn_feats = torch.gather(feats_batch.unsqueeze(-2).unsqueeze(-2).expand(-1, -1, args.topk, args.knn_num, -1), 1, knn_inds.unsqueeze(-1).expand(-1, -1, -1, -1, feats_batch.shape[-1]))
            with torch.cuda.amp.autocast(dtype=dtype):
                render_pkg_net = raylet_render_net(batch_ray_d.unsqueeze(0), hit_points, knn_pts.flatten(-2), knn_rel, knn_feats)

            if args.topk == 1:
                dist_delta = render_pkg_net[...,0].squeeze(0)
                render_dist = dist_delta + batch_render_dists.squeeze(-1)
            else:
                dist_delta = render_pkg_net[...,0].squeeze(0)
                unc = render_pkg_net[...,1].squeeze(0)
                unc = torch.softmax(unc, dim=-1)
                render_dist = ((dist_delta+batch_render_dists.squeeze(-1))*unc).sum(-1)
            if args.normal:
                normal = get_surface_normal(render_dist, sph_coord, slice_range = (i*args.iter_step, min((i+1)*args.iter_step, NUM_PIXEL))).detach()
                out_normal.append(normal)
            render_dist = render_dist.detach()
            out_dist.append(render_dist)

        out_dist = torch.cat(out_dist).reshape(select_gts.shape[-2], select_gts.shape[-1])
        torch.cuda.synchronize()
        end = time.time()
        time_lists.append(end - start)
        if args.normal and args.save_normal:
            out_normal = torch.cat(out_normal).reshape(select_gts.shape[-2], select_gts.shape[-1], 3)
            out_normal = -out_normal
            render_normal = to_normalmap(out_normal).squeeze()
            imageio.imwrite(os.path.join(scene_dir, scene_info["image_names"][test_idx:test_idx+1][0][0].split(".")[0] + ".png"), render_normal)

        out_dist =  out_dist.squeeze().reshape(-1, 1) * scene_scale_fit
        fit_distance = out_dist.reshape_as(select_gts)

    peak_memory_bytes = torch.cuda.max_memory_allocated(device)
    reserv_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
    peak_memory_gb = peak_memory_bytes / (1024 ** 3)
    peak_memory_list.append(peak_memory_gb)
    reserved_memory_list.append(reserv_gb)
    torch.cuda.empty_cache()
    progress_bar.set_postfix({"FPS": f"{1/np.mean(time_lists):.{5}f}"})
    progress_bar.update(1)

progress_bar.close()
print("Finished testing.")
print("Average inference time per view: {:.4f} seconds".format(np.mean(time_lists)))
print("Standard deviation of inference time per view: {:.4f} seconds".format(np.std(time_lists)))
print("FPS: {:.2f}".format(1/np.mean(time_lists)))
print("Average peak memory usage per view: {:.4f} GB".format(np.mean(peak_memory_list)))
print("Average reserved memory usage per view: {:.4f} GB".format(np.mean(reserved_memory_list)))
