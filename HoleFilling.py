import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from PIL import Image
import torch.nn as nn

# def hole_filling(sparse_depth): # using torch.nn.functional
#     kernel_size = 5
#     cnt_inter = 8
    
#     device =sparse_depth.device
#     fill_depth = sparse_depth.clone()
#     fill_depth = fill_depth.unsqueeze(0).type(torch.float32)
    
#     kernel = torch.ones((1, 1, kernel_size, kernel_size), device = device).type(torch.float32) # 단지 평균 값을 구하기 위해서 1로 구성된 kernel 생성
#     padding = (kernel_size-1)//2
    
    
#     for _ in range(cnt_inter):
#         valid_mask = (fill_depth > 0).type(torch.float32) # 1인 값을 세기 위해, 평균을 내야 하닌까
#         depth_sum = F.conv2d(fill_depth, kernel, stride= 1, padding=padding)
#         valid_count = F.conv2d(valid_mask, kernel, stride = 1, padding=padding)
#         avg_depth = depth_sum / valid_count
#         fill_depth = torch.where((fill_depth == 0) & (valid_count > 0), avg_depth, fill_depth)
    
#     fill_depth = fill_depth.squeeze(0)
            
#     return fill_depth

def hole_filling(sparse_depth): #using torch.nn
    kernel_size = 7
    cnt_inter = 10
    device =sparse_depth.device
    
    inital_depth = sparse_depth.clone()
    inital_depth = inital_depth.unsqueeze(0).type(torch.float32) #(1, 1, H, W)
    padding = (kernel_size-1)//2
    
    #평균용 conv 구하기
    avg_conv = nn.Conv2d(1, 1, kernel_size, stride= 1, padding = padding, bias = False)
    avg_conv.weight.data.fill_(1.0)
    avg_conv.weight.requires_grad = False
    avg_conv.to(device)
    
    for _ in range(cnt_inter):
        
        sumed_depth = avg_conv(inital_depth)
        
        cnt_mask = (inital_depth > 0).type(torch.float32)
        cnt_depth = avg_conv(cnt_mask)
        
        avg_depth = sumed_depth/ (cnt_depth + 1e-6)
        
        inital_depth = torch.where((inital_depth == 0) & (cnt_depth > 0), avg_depth, inital_depth)
        
    return inital_depth

    