import torch
import torch.nn.functional as F
import os
import numpy as np

def depth_2_normal(depth):
    """
    depth shape (1, 1, H, W)
    # Intrinsic matrix
    # f_x, f_y = 5.1885790117450188e+02, 5.1946961112127485e+02
    # c_x, c_y = 3.2558244941119034e+02, 2.5373616633400465e+02    
    Convert depth map to surface normal map using K^{-1} projection.
    """

    output_path = './output/normal_chk'
    os.makedirs(output_path, exist_ok= True)    
    
    # print(depth.shape)
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
    # print(pixel_coords)
    pixel_coords_np = pixel_coords.cpu().numpy()
    # np.save('./output/normal_chk/pixel_coord.npy', pixel_coords_np)
    

    # Step 3: Back-project to camera space
    # print(depth[0, 0].shape)
    depth_flat = depth[0, 0].reshape(-1)  # (H*W,)
    cam_points = (K_inv @ pixel_coords) * depth_flat  # (3, H*W)
    # cam_points = (K_inv @ pixel_coords) 
    # print(cam_points)
    # cam_points_np = cam_points.cpu().numpy()
    # np.save('output/normal_chk/cam_points.npy', cam_points_np)
    
    cam_coords = cam_points.reshape(3, H, W)  # (3, H, W)
    # print(cam_coords.shape)
    # cam_coords_np = cam_coords.permute(1, 2, 0).cpu().numpy()
    # cam_coords_np2 = cam_coords_np[:,:,2]
    # np.save('output/normal_chk/cam_coords.npy',cam_coords_np2)
    
    # Step 4: Compute surface normals using central differences
    cam_coords_padded = F.pad(cam_coords, (1, 1, 1, 1), mode='replicate')  # (3, H+2, W+2)

    dx = cam_coords_padded[:, 1:-1, 2:] - cam_coords_padded[:, 1:-1, :-2]  # (3, H, W)
    dy = cam_coords_padded[:, 2:, 1:-1] - cam_coords_padded[:, :-2, 1:-1]  # (3, H, W)

    normal = torch.cross(dx, dy, dim=0)  # (3, H, W)
    normal = F.normalize(normal, dim=0)

    # normal_np = normal.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
    # np.save('./output/normal_chk/normal.npy', normal_np)

    sol  =normal.unsqueeze(0) #(1, 3, 480, 640)
    # print(sol.shape)
    return sol