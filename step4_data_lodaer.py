import torch
import numpy as np
import cv2
import os
from HoleFilling import hole_filling 

class Step4Dataset(torch.utils.data.Dataset):
    def __init__(self, root="./data/data_submission/"):
        
        # RGB PA1 과제 참고
        rgb = cv2.imread(os.path.join(root, "rgb.png"))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = rgb.transpose(2, 0, 1) / 255.0  # (3, H, W)
        self.rgb = torch.from_numpy(rgb).type(torch.float32)

        
        sparse = np.load(os.path.join(root, "sparse_depth.npy"))
        self.sparse = torch.from_numpy(sparse).unsqueeze(0).type(torch.float32).cuda() #(1, H, W)

        
        initial = hole_filling(self.sparse)  # (1, 1, H, W)
        self.initial = initial.squeeze(0)

        # gt = np.load(os.path.join(root, "gt.npy"))
        # self.gt = torch.from_numpy(gt).unsqueeze(0).type(torch.float32) #(1, H, W)
        
        normal = np.load(os.path.join(root, "normal.npy")) 
        self.normal = torch.from_numpy(normal).permute(2, 0, 1).type(torch.float32) # (3, H, W)

    def __len__(self):
        return 30  # 하나 이미지를 여러번 수행하려고 설정하였다.

    def __getitem__(self, idx):
        return self.rgb, self.sparse, self.initial, self.normal
