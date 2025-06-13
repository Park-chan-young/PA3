import torch
import torch.nn.functional as F
import os
import numpy as np

def depth_2_normal(depth):

    # Intrinsic matrix 
    # f_x, f_y = 5.1885790117450188e+02, 5.1946961112127485e+02
    # c_x, c_y = 3.2558244941119034e+02, 2.5373616633400465e+02

    f_x, f_y = 518.8579, 519.4696
    c_x, c_y = 325.5824, 253.7362
    
    output_path = './output/normal_chk'
    os.makedirs(output_path, exist_ok= True)    
    depth = depth.type(torch.float32).cuda()
    # print(depth.shape) #(1, 1, H, W)
    _, _, H, W = depth.shape

    # Camera intrinsic matrix
    K = torch.tensor([
        [f_x, 0, c_x],
        [0, f_y, c_y],
        [0,  0,  1]
    ]).cuda()

    K_inv = torch.inverse(K)

    # D 좌표계로... 표현하기
    x_cord  = torch.arange(W, device=depth.device).repeat(H, 1).type(torch.float32) * depth[0, 0]
    # print(x_cord.shape)
    y_cord = torch.arange(H, device=depth.device).unsqueeze(1).repeat(1, W).type(torch.float32) *depth[0 , 0]
    # print(y_cord.shape)
    z_cord = depth[0, 0] # z는 depth scale 값으로
    pixel_coords = torch.stack((x_cord, y_cord, z_cord), dim=0).reshape(3, -1).type(torch.float32)  # 행렬곱으 위해, (3, H*W)
    # print(pixel_coords)
    # pixel_coords_np = pixel_coords.cpu().numpy()
    # np.save('./output/normal_chk/pixel_coord.npy', pixel_coords_np)
    
    # print(depth[0, 0].shape)
    cam_points = K_inv @ pixel_coords # (3, H*W)
    # cam_points = (K_inv @ pixel_coords) 
    # print(cam_points)
    # cam_points_np = cam_points.cpu().numpy()
    # np.save('output/normal_chk/cam_points.npy', cam_points_np)
    
    cam_coords = cam_points.reshape(3, H, W)  # (3, H, W)
    # print(cam_coords.shape)
    # cam_coords_np = cam_coords.permute(1, 2, 0).cpu().numpy()
    # cam_coords_np2 = cam_coords_np[:,:,2]
    # np.save('output/normal_chk/cam_coords.npy',cam_coords_np2)
    
    cam_coords_pad = F.pad(cam_coords, (1, 1, 1, 1), mode='replicate')  # (3, H+2, W+2)
    # print(cam_coords_pad)
    horizontal_vec = cam_coords_pad[:, 1:-1, 2:] - cam_coords_pad[:, 1:-1, :-2]  # (3, H, W)
    vertical_vec =  cam_coords_pad[:, :-2, 1:-1] - cam_coords_pad[:, 2:, 1:-1]  # (3, H, W)

    normal = torch.cross(vertical_vec, horizontal_vec, dim=0)  # (3, H, W)
    normal = F.normalize(normal, dim=0)

    # normal_np = normal.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
    # np.save('./output/normal_chk/normal.npy', normal_np)

    pred_normal  =normal.unsqueeze(0) #(1, 3, 480, 640)
    # print(pred_normal.shape)
    return pred_normal