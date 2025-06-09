# Computer vision PA3 CODE!

# You should submit this file with your report.
# You can chanage the function names and add your own functions.
# You can also add your own libraries.
# You should submit Initial depth and refined depth image as npy files !!!!
# Write your name and student ID below

# Student ID: 20251155
# Name: 박찬영 (Park Chanyoung)


#### Libraries ####
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from PIL import Image
import torch.nn as nn
from torch.utils.data import DataLoader
from data_loading import PA3Dataset
from HoleFilling import hole_filling 
from unet_model import UNet
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
    ######## sSettings ###########
    example_path = './data/data_example'
    submission_path = './data/data_submission'
    rgb_path = os.path.join(example_path, 'rgb.png')
    sparse_depth_path = os.path.join(example_path, 'sparse_depth.npy')
    normal_path = os.path.join(example_path, 'normal.npy')
    gt_path = os.path.join(example_path, 'gt.npy')
    
    inf_rgb_path = os.path.join(submission_path, 'rgb.png')
    inf_sparse_path = os.path.join(submission_path, 'sparse_depth.npy')
    inf_normal_path = os.path.join(submission_path, 'normal.npy')
    
    rgb = Image.open(rgb_path).convert('RGB')
    rgb = np.array(rgb).astype(np.float32) / 255.0
    rgb = torch.from_numpy(rgb).permute(2, 0, 1)
    
    sparse_depth = np.load(sparse_depth_path)
    sparse_depth = torch.from_numpy(sparse_depth).unsqueeze(0).type(torch.float32)
    normal = np.load(normal_path)
    normal = torch.from_numpy(normal).permute(2, 0, 1).float()
    
    gt = np.load(gt_path)
    gt = torch.from_numpy(gt).unsqueeze(0).type(torch.float32)
    
    inf_rgb = Image.open(inf_rgb_path).convert('RGB')
    inf_rgb = np.array(inf_rgb).astype(np.float32) / 255.0
    inf_rgb = torch.from_numpy(inf_rgb).permute(2, 0, 1)

    inf_sparse_depth = np.load(inf_sparse_path)
    inf_sparse_depth = torch.from_numpy(inf_sparse_depth).unsqueeze(0).type(torch.float32)
    inf_normal = np.load(inf_normal_path)
    inf_normal = torch.from_numpy(inf_normal).permute(2, 0, 1).float()  
    
    
    # #data_loading    
    # dataset = PA3Dataset('./data/data_exaple')
    # loader = DataLoader(dataset, batch_size = 1, shuffle = False)
    
    # for batch in loader:
    #     rgb = batch['rgb']
    #     sparse = batch['sparse_depth']
    #     normal = batch['normal']

      
    
    output_path = './output'
    os.makedirs(output_path, exist_ok= True)
    
    
    inital_tensor = hole_filling(sparse_depth)
    # inital_depth = inital_tensor.squeeze().cpu().numpy()
    
    # np.save(os.path.join(output_path, 'inital_depth.npy'), inital_depth)
    
    unet_input = torch.cat([rgb, inital_tensor.squeeze(0)], dim=0)
    model = UNet()
    unet_input = unet_input.unsqueeze(0)
    
    # with torch.no_grad():
    #     predicted_depth = model(unet_input) #출력 shape: (1, 1, H, W) 
    #     predicted_depth_np = predicted_depth.squeeze().cpu().numpy() #중간 결과 확인용

    # gt = gt.unsqueeze(0)    
    
    predicted_normal = depth_2_normal(predicted_depth) #depth_2_normal (1, 3, H, W)
    predicted_normal_np = predicted_normal.squeeze(0).permute(1, 2, 0).cpu().numpy()
    np.save(os.path.join(output_path, 'predicted_normal.npy'), predicted_normal_np)





if __name__ == '__main__':
    main()   
    

