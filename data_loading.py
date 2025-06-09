from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class PA3Dataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        
        self.root_dir = Path(root_dir)
        self.rgb_path = self.root_dir / 'rgb.png'
        self.sparse_depth_path = self.root_dir / 'sparse_depth.npy'
        self.normal_path = self.root_dir / 'normal.npy'
        self.transform = transform

    def __len__(self):
        # 단일 샘플이므로 1
        return 1

    def __getitem__(self, idx):
        # RGB image
        rgb = Image.open(self.rgb_path).convert('RGB')
        rgb = np.array(rgb).astype(np.float32) / 255.0
        rgb = torch.from_numpy(rgb).permute(2, 0, 1) #(3, H, W)

        # Sparse depth
        sparse_depth = np.load(self.sparse_depth_path)
        sparse_depth = torch.from_numpy(sparse_depth).unsqueeze(0).type(torch.float32) #(1 ,H, W)

        # GT surface normal
        normal = np.load(self.normal_path) 
        normal = torch.from_numpy(normal).permute(2, 0, 1).type(torch.float32) # (3, H, W)

        sample = {
            'rgb': rgb,
            'sparse_depth': sparse_depth,
            'normal': normal
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
