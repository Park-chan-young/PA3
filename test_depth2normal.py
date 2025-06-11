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



def main():

    dataset = PA3Dataset('./data/data_example')
    loader = DataLoader(dataset, batch_size = 1, shuffle = False)
    
    for batch in loader:
        rgb = batch['rgb']
        sparse = batch['sparse_depth']
        normal = batch['normal']

    rgb = rgb.squeeze(0).cuda()
    sparse = sparse.squeeze(0).cuda() #(1,H, W)
    normal = normal.squeeze(0).cuda()
    init_depth = hole_filling(sparse) #(1, 1, H, W)
    init_depth_np = init_depth.squeeze().cpu().numpy()
    
    gt_depth = np.load('./data/data_example/gt.npy') #(H, W)
    gt = torch.from_numpy(gt_depth).unsqueeze(0).type(torch.float32) #(1, H, W)
    
    init_normal = depth_2_normal(init_depth) # (1, 3, H, W)
    gt_normal = depth_2_normal(gt.unsqueeze(0)) # (1, 3, H, W)
    
    init_normal_np = init_normal.squeeze().permute(1, 2, 0).cpu().numpy()
    gt_normal_np = gt_normal.squeeze().permute(1, 2, 0).cpu().numpy()
    
    # np.save('./output/init_normal.npy', init_normal_np)
    np.save('./output/gt_normal.npy', gt_normal_np)
    
    
if __name__ == '__main__':
    main()