import numpy as np
import torch
import cv2
import os
from HoleFilling import hole_filling 

class SingleExampleDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir="./data/data_example/"):
        self.root = root_dir

        # RGB 이미지 불러오기 및 전처리
        rgb = cv2.imread(os.path.join(self.root, "rgb.png"))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)        # BGR → RGB
        rgb = rgb.transpose(2, 0, 1) / 255.0               # (3, H, W)

        # Sparse Depth 로딩 및 채널 차원 보장
        sparse = np.load(os.path.join(self.root, "sparse_depth.npy"))  # (H, W) or (1, H, W)
        if sparse.ndim == 2:
            sparse = np.expand_dims(sparse, axis=0)  # (1, H, W)
            temp_sparse = torch.from_numpy(sparse).cuda()
            first_depth = hole_filling(temp_sparse)
            first_depth = first_depth.squeeze(0).cpu().numpy()

        # GT Depth 로딩 및 채널 차원 보장
        gt = np.load(os.path.join(self.root, "gt.npy"))  # (H, W) or (1, H, W)
        if gt.ndim == 2:
            gt = np.expand_dims(gt, axis=0)

        # 텐서 변환
        self.input = torch.FloatTensor(np.concatenate([rgb, first_depth], axis=0))  # (4, H, W)
        self.gt = torch.FloatTensor(gt)  # (1, H, W)

    def __len__(self):
        return 100  # 반복 학습을 위한 샘플 수

    def __getitem__(self, idx):
        return self.input, self.gt
