import torch
import torch.nn.functional as F

def depth_2_normal(depth):
    """
    Convert depth map to surface normal map using K^{-1} projection.

    Args:
        depth (torch.Tensor): (1, 1, H, W)

    Returns:
        torch.Tensor: (1, 3, H, W), normalized surface normals
    """

    device = depth.device
    B, _, H, W = depth.shape
    assert B == 1, "Batch size 1 only supported"

    # Step 1: Camera intrinsics
    fx, fy = 518.8579, 519.4696
    cx, cy = 325.5824, 253.7362

    # Camera intrinsic matrix
    K = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ], device=device)

    K_inv = torch.inverse(K)  # (3, 3)

    # Step 2: Pixel grid (u, v)
    y, x = torch.meshgrid(torch.arange(H, device=device),
                          torch.arange(W, device=device), indexing='ij')
    ones = torch.ones_like(x)
    pixel_coords = torch.stack((x, y, ones), dim=0).reshape(3, -1).type(torch.float32)  # (3, H*W)

    # Step 3: Back-project to camera space
    depth_flat = depth[0, 0].reshape(-1)  # (H*W,)
    cam_points = (K_inv @ pixel_coords) * depth_flat  # (3, H*W)
    cam_coords = cam_points.reshape(3, H, W)  # (3, H, W)

    # Step 4: Central difference
    # cam_coords_padded = F.pad(cam_coords, (1, 1, 1, 1), mode='constant', value =0)
    cam_coords_padded = F.pad(cam_coords, (1, 1, 1, 1), mode='replicate')
    dx = cam_coords_padded[:, 1:-1, 2:] - cam_coords_padded[:, 1:-1, :-2] 
    dy = cam_coords_padded[:, 2:, 1:-1] - cam_coords_padded[:, :-2, 1:-1] 

    # Step 5: Cross product and normalize
    normal = torch.cross(dx, dy, dim=0)  # (3, H, W)
    norm = torch.norm(normal, dim=0, keepdim=True) + 1e-8
    normal = normal / norm

    # # Step 6: Flip direction to face camera
    # if torch.mean(normal[2]) > 0:
    #     normal = -normal

    return normal.unsqueeze(0)  # (1, 3, H, W)

