import torch
import numpy as np
import cv2
import os
from HoleFilling import hole_filling  # Step1 함수 (sparse → initial)

class Step4Dataset(torch.utils.data.Dataset):
    def __init__(self, root="./data/data_example/"):
        # Load RGB
        rgb = cv2.imread(os.path.join(root, "rgb.png"))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = rgb.transpose(2, 0, 1) / 255.0  # (3, H, W)

        # Load Sparse Depth
        sparse = np.load(os.path.join(root, "sparse_depth.npy"))
        if sparse.ndim == 2:
            sparse = np.expand_dims(sparse, axis=0)

        # Step1: Hole Filling → initial depth
        sparse_tensor = torch.FloatTensor(sparse)
        initial = hole_filling(sparse_tensor)  # (1, 1, H, W)
        initial = initial.squeeze(0).numpy()  # (1, H, W)

        # Load GT Depth
        gt = np.load(os.path.join(root, "gt.npy"))
        if gt.ndim == 2:
            gt = np.expand_dims(gt, axis=0)

        # Load GT Normal
        normal = np.load(os.path.join(root, "normal.npy"))  # (3, H, W)

        # Tensor 변환
        self.rgb = torch.FloatTensor(rgb)         # (3, H, W)
        self.initial = torch.FloatTensor(initial) # (1, H, W)
        self.sparse = torch.FloatTensor(sparse)
        self.gt = torch.FloatTensor(gt)
        self.normal = torch.FloatTensor(normal)

    def __len__(self):
        return 100  # 반복 학습용

    def __getitem__(self, idx):
        input_tensor = torch.cat([self.rgb, self.initial], dim=0)  # (4, H, W)
        return input_tensor, self.sparse, self.gt, self.normal
