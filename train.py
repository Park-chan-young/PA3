import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
from unet_model import UNet
from HoleFilling import hole_filling
from Depth2Normal import depth_2_normal
from torch.utils.data import DataLoader
from data_loading import PA3Dataset
import torch.nn.functional as F

# class DepthDataset(Dataset):
#     def __init__(self, rgb_path, sparse_path, gt_path, normal_path):
#         self.rgb = torch.from_numpy(np.array(Image.open(rgb_path)).astype(np.float32) / 255.0).permute(2, 0, 1)
#         self.sparse = torch.from_numpy(np.load(sparse_path)).unsqueeze(0).float()
#         self.gt = torch.from_numpy(np.load(gt_path)).unsqueeze(0).float()
#         self.normal = torch.from_numpy(np.load(normal_path)).permute(2, 0, 1).float()

#     def __len__(self):
#         return 1  # Single example

#     def __getitem__(self, idx):
#         return self.rgb, self.sparse, self.gt, self.normal

def normal_l2_loss(pred, gt):
    return ((pred - gt) ** 2).mean()

def train():
    # # Set paths
    # rgb_path = './data/data_example/rgb.png'
    # sparse_path = './data/data_example/sparse_depth.npy'
    # gt_path = './data/data_example/gt.npy'
    # normal_path = './data/data_example/normal.npy'

    # Load dataset
    # dataset = DepthDataset(rgb_path, sparse_path, gt_path, normal_path)
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    dataset = PA3Dataset('./data/data_example')
    loader =  DataLoader(dataset, batch_size= 1, shuffle= False)
    
    for batch in loader:
        rgb = batch['rgb']
        sparse = batch['sparse_depth']
        normal = batch['normal']
    
    # Model, optimizer, loss
    model = UNet().cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    alpha = 0.01  # sparse loss weight
    beta = 1.0   # normal loss weight

    num_epochs = 500
    model.train()
    rgb = rgb.squeeze(0).cuda()
    sparse = sparse.squeeze(0).cuda()
    normal = normal.squeeze(0).cuda()
    init_depth =hole_filling(sparse)
    unet_input = torch.cat([rgb, init_depth.squeeze(0)], dim=0).unsqueeze(0).cuda() #(4, H, W)

    for epoch in range(num_epochs):
        pred_depth = model(unet_input)  # (1, 1, H, W)

        # Loss_sparse
        mask = (sparse > 0).float()
        loss_sparse = (mask * torch.abs(pred_depth - sparse)).sum() / (mask.sum() + 1e-6)

        # Loss_normal
        pred_normal = depth_2_normal(pred_depth)
        loss_normal = normal_l2_loss(pred_normal, normal.unsqueeze(0).cuda())
        

        # Total loss
        loss = alpha * loss_sparse + beta * loss_normal

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == num_epochs - 1:
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "./output/unet_trained.pth")

if __name__ == '__main__':
    train()
