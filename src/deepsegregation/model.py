"""Point cloud segmentation models.

Includes:
1. Baseline PointMLP for fast smoke tests.
2. Hierarchical PointNet++ (SSG) with Set Abstraction and Feature Propagation
   written in pure PyTorch (no custom C++/CUDA extension required), providing
   neighborhood surface context to break the 20% mIoU ceiling on K80 / CPU.
"""

from __future__ import annotations


def build_point_mlp(input_features: int = 3, num_classes: int = 3):
    try:
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("PointMLP requires PyTorch") from exc
    if input_features < 3 or num_classes < 2:
        raise ValueError("invalid model dimensions")

    class PointMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_features, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(64, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(128, num_classes),
            )

        def forward(self, features):
            shape = features.shape
            if features.ndim == 3:
                features = features.reshape(-1, shape[-1])
            if features.ndim != 2 or features.shape[-1] != input_features:
                raise ValueError("features must have shape (N,F) or (B,N,F)")
            logits = self.network(features)
            return logits.reshape(*shape[:-1], num_classes)

    return PointMLP()


def build_pointnet2_ssg(input_features: int = 3, num_classes: int = 3):
    """Build a pure-PyTorch PointNet++ Single-Scale Grouping (SSG) segmentation model."""
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError("PointNet2SSG requires PyTorch") from exc

    if input_features < 3 or num_classes < 2:
        raise ValueError("invalid model dimensions")

    def square_distance(src, dst):
        """Calculate squared Euclidean distance between each pair of points."""
        B, N, _ = src.shape
        _, M, _ = dst.shape
        dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
        dist += torch.sum(src ** 2, -1).view(B, N, 1)
        dist += torch.sum(dst ** 2, -1).view(B, 1, M)
        return torch.clamp(dist, min=0.0)

    def index_points(points, idx):
        """Index points given an index tensor."""
        device = points.device
        B = points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
        new_points = points[batch_indices, idx, :]
        return new_points

    def farthest_point_sample(xyz, npoint):
        """Iterative farthest point sampling."""
        device = xyz.device
        B, N, _ = xyz.shape
        npoint = min(npoint, N)
        centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
        distance = torch.ones(B, N, device=device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
        batch_indices = torch.arange(B, dtype=torch.long, device=device)
        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
            dist = torch.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]
        return centroids

    def query_ball_point(radius, nsample, xyz, new_xyz):
        """Find nsample points within radius distance."""
        device = xyz.device
        B, N, _ = xyz.shape
        _, S, _ = new_xyz.shape
        group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat([B, S, 1])
        sqrdists = square_distance(new_xyz, xyz)
        group_idx[sqrdists > radius ** 2] = N
        group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
        group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
        mask = group_idx == N
        group_idx[mask] = group_first[mask]
        return group_idx

    class PointNetSetAbstraction(nn.Module):
        def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
            super().__init__()
            self.npoint = npoint
            self.radius = radius
            self.nsample = nsample
            self.group_all = group_all
            self.mlp_convs = nn.ModuleList()
            self.mlp_bns = nn.ModuleList()
            last_channel = in_channel
            for out_channel in mlp:
                self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel

        def forward(self, xyz, points):
            xyz = xyz.contiguous()
            if points is not None:
                points = points.contiguous()

            B, N, C = xyz.shape
            if self.group_all:
                new_xyz = torch.zeros(B, 1, 3, device=xyz.device)
                grouped_xyz = xyz.view(B, 1, N, 3)
                if points is not None:
                    new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
                else:
                    new_points = grouped_xyz
            elif self.npoint is None or self.npoint >= N:
                new_xyz = xyz
                if points is not None:
                    combined = torch.cat([xyz, points], dim=-1)
                else:
                    combined = xyz
                new_points = combined.unsqueeze(2)
            else:
                fps_idx = farthest_point_sample(xyz, self.npoint)
                new_xyz = index_points(xyz, fps_idx)
                idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
                grouped_xyz = index_points(xyz, idx)
                grouped_xyz_norm = grouped_xyz - new_xyz.view(B, self.npoint, 1, 3)

                if points is not None:
                    grouped_points = index_points(points, idx)
                    new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
                else:
                    new_points = grouped_xyz_norm

            # Shape: (B, C_in, nsample, npoint)
            new_points = new_points.permute(0, 3, 2, 1)
            for conv, bn in zip(self.mlp_convs, self.mlp_bns):
                new_points = F.relu(bn(conv(new_points)))
            new_points = torch.max(new_points, 2)[0]  # Max pool over neighbors
            new_points = new_points.permute(0, 2, 1)   # (B, npoint, C_out)
            return new_xyz, new_points

    class PointNetFeaturePropagation(nn.Module):
        def __init__(self, in_channel, mlp):
            super().__init__()
            self.mlp_convs = nn.ModuleList()
            self.mlp_bns = nn.ModuleList()
            last_channel = in_channel
            for out_channel in mlp:
                self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
                self.mlp_bns.append(nn.BatchNorm1d(out_channel))
                last_channel = out_channel

        def forward(self, xyz1, xyz2, points1, points2):
            """
            Interpolate points2 features from xyz2 onto xyz1 locations,
            concatenate with skip features points1, and pass through MLP.
            """
            B, N, C = xyz1.shape
            _, S, _ = xyz2.shape

            if S == 1:
                interpolated_points = points2.repeat(1, N, 1)
            else:
                dists = square_distance(xyz1, xyz2)
                dists, idx = dists.sort(dim=-1)
                dists, idx = dists[:, :, :3], idx[:, :, :3]
                dist_recip = 1.0 / (dists + 1e-10)
                norm = torch.sum(dist_recip, dim=2, keepdim=True)
                weight = dist_recip / norm
                interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

            if points1 is not None:
                new_points = torch.cat([points1, interpolated_points], dim=-1)
            else:
                new_points = interpolated_points

            new_points = new_points.permute(0, 2, 1)
            for conv, bn in zip(self.mlp_convs, self.mlp_bns):
                new_points = F.relu(bn(conv(new_points)))
            return new_points.permute(0, 2, 1)

    class PointNet2SSG(nn.Module):
        def __init__(self):
            super().__init__()
            extra_features = input_features - 3
            # Set Abstraction 1
            self.sa1 = PointNetSetAbstraction(npoint=256, radius=0.2, nsample=32,
                                             in_channel=3 + extra_features, mlp=[32, 32, 64])
            # Set Abstraction 2
            self.sa2 = PointNetSetAbstraction(npoint=64, radius=0.4, nsample=32,
                                             in_channel=64 + 3, mlp=[64, 64, 128])
            # Set Abstraction 3 (Global feature)
            self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None,
                                             in_channel=128 + 3, mlp=[128, 256, 256], group_all=True)

            # Feature Propagation layers
            self.fp3 = PointNetFeaturePropagation(in_channel=256 + 128, mlp=[128, 128])
            self.fp2 = PointNetFeaturePropagation(in_channel=128 + 64, mlp=[128, 64])
            self.fp1 = PointNetFeaturePropagation(in_channel=64 + extra_features, mlp=[64, 64])

            # Output classification head
            self.conv1 = nn.Conv1d(64, 64, 1)
            self.bn1 = nn.BatchNorm1d(64)
            self.drop1 = nn.Dropout(0.3)
            self.conv2 = nn.Conv1d(64, num_classes, 1)

        def forward(self, features):
            # Input features shape: (B, N, F) or (N, F)
            is_2d = (features.ndim == 2)
            if is_2d:
                features = features.unsqueeze(0)

            B, N, F_dim = features.shape
            xyz = features[:, :, :3]
            points = features[:, :, 3:] if F_dim > 3 else None

            # Hierarchical Set Abstraction Encoder
            l1_xyz, l1_points = self.sa1(xyz, points)
            l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
            l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

            # Feature Propagation Decoder
            l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
            l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
            l0_points = self.fp1(xyz, l1_xyz, points, l1_points)

            # Head
            feat = l0_points.permute(0, 2, 1)  # (B, C, N)
            x = self.drop1(F.relu(self.bn1(self.conv1(feat))))
            x = self.conv2(x)                 # (B, num_classes, N)
            logits = x.permute(0, 2, 1)       # (B, N, num_classes)

            if is_2d:
                return logits.squeeze(0)
            return logits

    return PointNet2SSG()
