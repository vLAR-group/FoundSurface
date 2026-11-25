import torch
import torch.nn as nn
from itertools import repeat
import collections.abc
from functools import partial
import torch.nn.functional as F
from raydf.my_dpt_head import DPTHead
from vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class Attention(nn.Module):

    def __init__(self, dim, rope=None, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., fused_attn = True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope
        self.fused_attn = fused_attn

    def forward(self, x, xpos):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1,3)
        q, k, v = [qkv[:,:,i] for i in range(3)]

        if self.rope is not None:
            q = self.rope(q, xpos)
            k = self.rope(k, xpos)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttention(nn.Module):

    def __init__(self, dim, rope=None, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., fused_attn = True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.projq = nn.Linear(dim, dim, bias=qkv_bias)
        self.projk = nn.Linear(dim, dim, bias=qkv_bias)
        self.projv = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.fused_attn = fused_attn
        self.rope = rope

    def forward(self, query, key, value, qpos, kpos):
        B, Nq, C = query.shape
        Nk = key.shape[1]
        Nv = value.shape[1]

        q = self.projq(query).reshape(B,Nq,self.num_heads, C// self.num_heads).permute(0, 2, 1, 3)
        k = self.projk(key).reshape(B,Nk,self.num_heads, C// self.num_heads).permute(0, 2, 1, 3)
        v = self.projv(value).reshape(B,Nv,self.num_heads, C// self.num_heads).permute(0, 2, 1, 3)

        if self.rope is not None:
            q = self.rope(q, qpos)
            k = self.rope(k, kpos)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class PatchEmbed(nn.Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)

    Args:
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
    """

    def __init__(
        self,
        patch_size = (16, 16),
        in_chans = 3,
        embed_dim = 768,
        norm_layer = None,
        flatten_embedding = True,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _,  H, W = x.shape
        patch_H, patch_W = self.patch_size

        assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
        assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"

        x = self.proj(x)
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)
        return x

class RenderBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_mem=True, rope=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.cross_attn = CrossAttention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm_y = norm_layer(dim) if norm_mem else nn.Identity()

    def forward(self, x, y, xpos=None, ypos=None):
        b, v, n, c = x.shape
        x = x.view(b * v, n, c)
        if xpos is not None:
            xpos = xpos.view(b * v, n, 2)

        x = x + self.drop_path(self.attn(self.norm1(x), xpos))
        y_ = self.norm_y(y)

        x = x.view(b, v, n, c)
        x = x.reshape(b, v*n, c)

        if xpos is not None:
            xpos = xpos.view(b, v*n, 2)

        x = x + self.drop_path(self.cross_attn(self.norm2(x), y_, y_, xpos, ypos))
        x = x + self.drop_path(self.mlp(self.norm3(x)))

        x = x.view(b, v, n, c)
        return x, y

class CondNet(nn.Module):
    def __init__(self, dec_embed_dim, geo_token_dim = 2048+512, patch_in = 6, pos_emb = False):
        super(CondNet, self).__init__()
        self.pose_emb_layer = PatchEmbed(patch_size=(14, 14), in_chans=patch_in, embed_dim=512)
        self.map_layer = nn.Sequential(
            nn.Linear(geo_token_dim, dec_embed_dim),
            nn.ReLU(),
            nn.Linear(dec_embed_dim, dec_embed_dim)
        )
    def forward(self, pose_cond, geo_tokens_uncond):
        b, v, _, h, w = pose_cond.shape
        pose_cond = pose_cond.reshape(b * v, -1, h, w)
        pose_tokens = self.pose_emb_layer(pose_cond)
        pose_tokens = pose_tokens.reshape(b, v, -1, self.pose_emb_layer.embed_dim)

        geometry_cond_tokens = torch.cat([pose_tokens, geo_tokens_uncond], dim=-1)
        b, v, n, c = geometry_cond_tokens.shape
        geo_tokens = self.map_layer(geometry_cond_tokens)

        return geo_tokens

class RenderDPTNet(nn.Module):
    def __init__(self, dec_embed_dim, dec_num_heads, dec_depth, mlp_ratio, norm_layer = partial(nn.LayerNorm, eps=1e-6), norm_im2_in_dec = True, pos_emb = True,
                        rope_freq=0, patch_size=(16, 16), intermediate_layer_idx=[0,1,2,3]):
        super(RenderDPTNet, self).__init__()
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None
        self.pos_emb = pos_emb
        self.path_size = patch_size
        self.intermediate_layer_idx = intermediate_layer_idx

        self.pose_emb_layer = PatchEmbed(patch_size=patch_size, in_chans=6, embed_dim=dec_embed_dim)
        self.dec_norm = norm_layer(dec_embed_dim)
        self.dec_blocks = nn.ModuleList([
            RenderBlock(dec_embed_dim, dec_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer, norm_mem=norm_im2_in_dec, rope=self.rope)
            for i in range(dec_depth)])

        self.dense_dist_head = DPTHead(dim_in=dec_embed_dim, output_dim=1, activation="exp", conf_activation="expp1", patch_size=patch_size[0], intermediate_layer_idx = [0,1,2,3])

    def forward(self, ray_params, scene_tokens, rec_size):
        rec_p_H, rec_p_W = rec_size[0] // 14, rec_size[1] // 14
        b, v, c, h, w = ray_params.shape
        ray_params = ray_params.reshape(b * v, -1, h, w)
        ray_tokens = self.pose_emb_layer(ray_params)
        scene_b, scene_v, scene_n, scene_c = scene_tokens.shape
        scene_tokens = scene_tokens.reshape(scene_b, scene_v*scene_n, scene_c)
        if self.rope is not None:
            x_pos = self.position_getter(b * v, h // self.path_size[0], w // self.path_size[1], device=ray_params.device)
            y_pos = self.position_getter(scene_b * scene_v, rec_p_H, rec_p_W, device=ray_params.device).reshape(scene_b, -1, 2)
        ray_tokens = ray_tokens.view(b, v, -1, self.pose_emb_layer.embed_dim)

        token_list = []
        for i, blk in enumerate(self.dec_blocks):
            ray_tokens, _ = blk(ray_tokens, scene_tokens, x_pos, y_pos)
            if i in self.intermediate_layer_idx:
                if i == self.intermediate_layer_idx[-1]:
                    ray_tokens = self.dec_norm(ray_tokens)
                token_list.append(ray_tokens.contiguous())
        dense_dists = self.dense_dist_head(token_list, (b, v, c, h, w), patch_start_idx=0)
        return dense_dists

class SceneAggNetMLP(nn.Module):
    def __init__(self, dec_embed_dim, input_dim = 2048*4):
        super(SceneAggNetMLP, self).__init__()
        self.map_layer = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, dec_embed_dim)
        )

    def forward(self, vggt_feature_list):
        vggt_feature_list = torch.cat(vggt_feature_list, dim=-1)
        scene_tokens = self.map_layer(vggt_feature_list)
        return scene_tokens
