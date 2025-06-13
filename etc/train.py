import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import os
from unet_model_final import UNet
from HoleFilling import hole_filling
from Depth2Normal import depth_2_normal
from data_loading import PA3Dataset
import torch.nn.functional as F

def normal_l2_loss(pred, gt):
    # (1, 3, H, W)
    diff = (pred - gt) ** 2              # (1, 3, H, W)
    l2_per_pixel = diff.sum(dim=1)       # (1, H, W), 벡터 L2 제곱합
    return l2_per_pixel.mean()           # 전체 픽셀 평균

def normal_cosine_loss(pred_normal, gt_normal):
    # 정규화 이미 되어 있다고 가정
    cos_sim = (pred_normal * gt_normal).sum(dim=1)  # (B, H, W)

    # 유효 영역만 loss에 포함 (0 vector 등 제외)
    valid_mask = (gt_normal.abs().sum(dim=1) > 0).float()
    return (1 - cos_sim * valid_mask).sum() / (valid_mask.sum() + 1e-6)


def depth_l1_loss(pred_d, sparse_d, weight=5.0):
    mask = (sparse_d > 0).float()
    diff = torch.abs(pred_d - sparse_d)
    
    weighted_diff = weight * mask * diff + (1 - mask) * diff  # 유효한 픽셀만 강조
    loss = weighted_diff.mean()
    return loss


def depth_train():
    # Load dataset
    dataset = PA3Dataset('./data/data_example')
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Model, optimizer
    model = UNet(in_channels=4, out_channels=1).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    model.train()

    alpha = 1.0  # sparse loss weight
    beta = 0.5 # normal loss weight
    num_epochs = 100

    # Logging setup
    os.makedirs('./test_log', exist_ok=True)
    log_file = open("./test_log/train_log.txt", "w")
    log_file.write(f"alpha: {alpha}, beta: {beta}\n")
    print(f"alpha: {alpha}, beta: {beta}")

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in loader:
            rgb = batch['rgb'].cuda()
            sparse = batch['sparse_depth'].squeeze(0).cuda()
            normal = batch['normal'].cuda()
            ggt = batch['gt'].cuda()

            init_depth = hole_filling(sparse)      # (1, 1, H, W)
            unet_input = torch.cat([rgb,init_depth], dim=1)  # (1, 4, H, W)

            pred_depth = model(unet_input)  # (1, 1, H, W)

            # Loss_sparse
            loss_sparse = depth_l1_loss(pred_depth, sparse.unsqueeze(0)) #(1, 1, 480, 640)

            # Loss_normal
            pred_normal = depth_2_normal(pred_depth) # pred_normal (1, 3, 480, 640)
            # print(pred_normal.shape)
            loss_normal = normal_cosine_loss(pred_normal, normal) #(1, 3, 480, 640)

            # Total loss
            loss = alpha * loss_sparse + beta * loss_normal
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 1 == 0 or epoch == num_epochs - 1:
            log = f"Epoch [{epoch}/{num_epochs}], Loss: {epoch_loss:.4f}"
            print(log)
            log_file.write(log + "\n")

    log_file.close()

    # Save model
    os.makedirs('./output', exist_ok=True)

    torch.save(model.state_dict(), "./output/unet_trained.pth")

if __name__ == '__main__':
    depth_train()
