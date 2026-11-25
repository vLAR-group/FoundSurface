import torch
import torch.nn as nn

class RaySurfDNetMLPVGGTReduceMeanFuse(nn.Module):
    def __init__(self, gs_layer = 3, dist_layer = 5, W=256, map_layer_dim=228-32, geo_token_dim=2048, reduce_dim=128, out_dim = 1):
        super(RaySurfDNetMLPVGGTReduceMeanFuse, self).__init__()

        self.gs_encoder = nn.Sequential(nn.Linear(W, W), nn.ReLU(inplace=True))
        for _ in range(1, gs_layer):
            self.gs_encoder.append(nn.Linear(W, W))
            self.gs_encoder.append(nn.ReLU(inplace=True))

        self.map_layer = nn.Sequential(nn.Linear(map_layer_dim, W), nn.ReLU(inplace=True))
        self.dist_dense = nn.Sequential(nn.Linear(W, out_dim))
        for i in range(dist_layer):
            self.dist_dense.append(nn.ReLU(inplace=True))
            self.dist_dense.append(nn.Linear(W, W))
        self.dist_dense = self.dist_dense[::-1]
        self.geo_feats_map_layer = nn.Sequential(
            nn.Linear(geo_token_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

    def forward(self, lf_input, hit_pt, knn_pts, knn_vec, knn_feats):
        knn_feats = self.geo_feats_map_layer(knn_feats)
        knn_feats = knn_feats.mean(-2)
        lf_input = torch.tile(lf_input, [1, 1, hit_pt.shape[2], 1])
        geo_input = torch.cat([lf_input, hit_pt, knn_pts, knn_vec, knn_feats], dim=-1)
        geo_input = self.map_layer(geo_input)
        cond_feats = self.gs_encoder(geo_input)
        outputs = self.dist_dense(cond_feats)

        return outputs

