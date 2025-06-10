import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import os
from unet_model import UNet
from HoleFilling import hole_filling
from Depth2Normal import depth_2_normal
from data_loading import PA3Dataset
import torch.nn.functional as F

def normal_l2_loss(pred, gt):
    return ((pred - gt) ** 2).mean()

def normal_cosine_loss(pred, gt):
    pred = F.normalize(pred, dim=1)
    gt = F.normalize(gt, dim=1)
    return 1 - (pred * gt).sum(dim=1).mean()

def depth_l1_loss(pred_d, sparse_d):
    mask = (sparse_d > 0).float()
    loss_sparse_ = (mask * torch.abs(pred_d - sparse_d)).sum() / (mask.sum() + 1e-6)
    return loss_sparse_   

def depth_train():
    # Load dataset
    dataset = PA3Dataset('./data/data_example')
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Model, optimizer
    model = UNet().cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    model.train()

    alpha = 1.0  # sparse loss weight
    beta = 0.5   # normal loss weight
    num_epochs = 500

    # Logging setup
    os.makedirs('./test_log', exist_ok=True)
    log_file = open("./test_log/train_log.txt", "w")
    log_file.write(f"alpha: {alpha}, beta: {beta}\n")
    print(f"alpha: {alpha}, beta: {beta}")

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in loader:
            rgb = batch['rgb'].squeeze(0).cuda()
            sparse = batch['sparse_depth'].squeeze(0).cuda()
            normal = batch['normal'].squeeze(0).cuda()

            init_depth = hole_filling(sparse)  # (1, 1, H, W)
            unet_input = torch.cat([rgb, init_depth.squeeze(0)], dim=0).unsqueeze(0).cuda()  # (1, 4, H, W)

            pred_depth = model(unet_input)  # (1, 1, H, W)

            # Loss_sparse
            loss_sparse = depth_l1_loss(pred_depth, sparse.unsqueeze(0))

            # Loss_normal
            pred_normal = depth_2_normal(pred_depth)
            loss_normal = normal_l2_loss(pred_normal, normal.unsqueeze(0))

            # Total loss
            loss = alpha * loss_sparse + beta * loss_normal
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 50 == 0 or epoch == num_epochs - 1:
            log = f"Epoch [{epoch}/{num_epochs}], Loss: {epoch_loss:.4f}"
            print(log)
            log_file.write(log + "\n")

    log_file.close()

    # Save model
    os.makedirs('./output', exist_ok=True)
    torch.save(model.state_dict(), "./output/unet_trained.pth")

if __name__ == '__main__':
    depth_train()
