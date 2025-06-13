# Computer vision PA3 CODE!

# You should submit this file with your report.
# You can chanage the function names and add your own functions.
# You can also add your own libraries.
# You should submit Initial depth and refined depth image as npy files !!!!
# Write your name and student ID below

# Student ID: 20251155
# Name: 박찬영 (Park Chanyoung)


#### Libraries ####
import numpy as np
import torch
import cv2
from PIL import Image
import os
import torch.nn as nn
from unet_model_final import UNet
from HoleFilling import hole_filling 
from Depth2Normal import depth_2_normal

# ############# Step 1 #############
# def hole_filling(sparse_depth):

#     pass
#     # Todo
    
############# Step 1 #############






#### Load Data ####

# sparse_depth, rgb, ... = ...

#### Load Data ####






# ############# Step 2 #############
# UNet = None

# ############# Step 2 #############





# ############# Step 3 #############
# def depth_2_normal(depth):
#     pass
#     # Todo

#     # Intrinsic matrix
#     # f_x, f_y = 5.1885790117450188e+02, 5.1946961112127485e+02
#     # c_x, c_y = 3.2558244941119034e+02, 2.5373616633400465e+02

############# Step 3 #############





############# Step 4 #############

# Traing UNet 

# Loss_Normal = None
# Loss_sparse = None
# alpha, beta = None, None
# Loss = alpha * Loss_Normal + beta * Loss_sparse



############# Step 4 #############


def main():
    ####setting PA1 과제를 참고하였다.....
    example_path = './data/data_submission'
    # example_path = './data/data_submission'
    rgb_path = os.path.join(example_path, 'rgb.png')
    sparse_depth_path = os.path.join(example_path, 'sparse_depth.npy')
    normal_path = os.path.join(example_path, 'normal.npy')
    # gt_path = os.path.join(example_path, 'gt.npy')

    rgb = cv2.imread(os.path.join(example_path, "rgb.png"))
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb = rgb.transpose(2, 0, 1) / 255.0 
    rgb = torch.from_numpy(rgb).unsqueeze(0).type(torch.float32).cuda()
    # print(rgb.shape) #(1, 3, H, W)

    sparse_depth = np.load(sparse_depth_path)
    sparse_depth = torch.from_numpy(sparse_depth).unsqueeze(0).float().cuda() 
    # print(sparse_depth.shape) #( 1, H, W)

    normal = np.load(normal_path)
    normal = torch.from_numpy(normal).permute(2, 0, 1).unsqueeze(0).float().cuda()  
    # print(normal.shape)
    
    # gt = np.load(gt_path)
    # gt = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0).float().cuda()  # (1, 1, H, W)

    output_path = './output'
    os.makedirs(output_path, exist_ok=True)

    #### predict용 inital_depth 그리고 inital depth 구하기
    init_depth = hole_filling(sparse_depth)  # (1, 1, H, W)
    unet_input = torch.cat([rgb, init_depth], dim=1)
    np.save('./output/Inital_depth.npy', init_depth.squeeze().cpu().numpy())

    ####### 저장값 불러서 예측gkrl
    model = UNet(in_channels=4, out_channels=1).cuda()
    model.load_state_dict(torch.load('./output/unet_trained.pth'))
    model.eval()

    unet_input = torch.cat([rgb, init_depth], dim=1)

    with torch.no_grad():
        pred_depth = model(unet_input)  # (1, 1, H, W)
        np.save('./output/Final_refined_depth.npy', pred_depth.squeeze().cpu().numpy())


if __name__ == '__main__':
    main()

