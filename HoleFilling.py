import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from PIL import Image
import torch.nn as nn


def hole_filling(sparse_depth):
    
    kernel_size = 7
    cnt_iter = 10
    inital_depth = sparse_depth
    # print(inital_depth.shape) #(1, 1, H, W)
    inital_depth = inital_depth.unsqueeze(0).type(torch.float32) #(1, 1, H, W)
    padding = (kernel_size-1)//2
    
    # 평균 필터 생성하기
    avg_conv = nn.Conv2d(1, 1, kernel_size, stride= 1, padding = padding, bias = False)
    avg_conv.cuda()
    avg_conv.weight.data.fill_(1.0)
    avg_conv.weight.requires_grad = False

    # 보간 수행하기
    for _ in range(cnt_iter):

        mask_depth = (inital_depth > 0)
        mask_depth = mask_depth.type(torch.float32) 
        cnt_depth = avg_conv(mask_depth)
        # print(cnt_depth)
        
        sumed_depth = avg_conv(inital_depth)
        # print(sumed_depth)
        # print("min/max:", np.min(sumed_depth), "/", np.max(sumed_depth))
        
        avg_depth = sumed_depth/ (cnt_depth + 1e-5)
        # print(avg_depth)
        # print("min/max:", np.min(avg_depth), "/", np.max(avg_depth))
        
        inital_depth = torch.where((inital_depth == 0), avg_depth, inital_depth)
        # print("min/max:", np.min(inital_depth), "/", np.max(inital_depth))
        

    return inital_depth

    