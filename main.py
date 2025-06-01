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


############# Step 1 #############
def hole_filling(sparse_depth):
    
    pass
    # Todo
    
############# Step 1 #############






#### Load Data ####

# sparse_depth, rgb, ... = ...

#### Load Data ####






############# Step 2 #############
UNet = None

############# Step 2 #############





############# Step 3 #############
def depth_2_normal(depth):
    pass
    # Todo

    # Intrinsic matrix
    # f_x, f_y = 5.1885790117450188e+02, 5.1946961112127485e+02
    # c_x, c_y = 3.2558244941119034e+02, 2.5373616633400465e+02

############# Step 3 #############





############# Step 4 #############

# Traing UNet 

Loss_Normal = None
Loss_sparse = None
alpha, beta = None, None
Loss = alpha * Loss_Normal + beta * Loss_sparse



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
    sparse_depth = torch.from_numpy(sparse_depth).unsqueeze(0).type(float)
    normal = np.load(normal_path)
    normal = torch.from_numpy(normal).permute(2, 0, 1).float()
    
    gt = np.load(gt_path)
    gt = torch.from_numpy(gt).unsqueeze(0).float()
    
    inf_rgb = Image.open(inf_rgb_path).convert('RGB')
    inf_rgb = np.array(inf_rgb).astype(np.float32) / 255.0
    inf_rgb = torch.from_numpy(inf_rgb).permute(2, 0, 1)

    inf_sparse_depth = np.load(inf_sparse_path)
    inf_sparse_depth = torch.from_numpy(inf_sparse_depth).unsqueeze(0).type(float)
    inf_normal = np.load(inf_normal_path)
    inf_normal = torch.from_numpy(inf_normal).permute(2, 0, 1).float()    
    
    pass



if __name__ == '__main__':
    main()   
    

