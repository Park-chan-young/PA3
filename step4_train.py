import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from step4_data_lodaer import Step4Dataset
from unet_model_final import UNet
from Depth2Normal import depth_2_normal  # torch 기반 normal 추출 함수
import numpy as np

# # Loss 함수들
def sparse_depth_loss(pred, sparse):

    mask = (sparse > 0).float() 
    diff = torch.abs(pred - sparse)
    loss = (mask * diff).sum() / (mask.sum() + 1e-6)
    return loss

# def normal_cosine_loss(pred_normal, gt_normal):

#     pred = nn.functional.normalize(pred_normal, dim=0)
#     gt = nn.functional.normalize(gt_normal, dim=0)
#     cos_sim = (pred * gt).sum(dim=0)  # (H, W)
#     loss = 1 - cos_sim  # (H, W)
#     return loss.mean()

def normal_l2_loss(pred_normal, gt_normal):
 
    diff = (pred_normal - gt_normal) ** 2  # (3, H, W)
    loss = diff.mean()
    return loss


def train_step4():

    #데이터 받아오기
    dataset = Step4Dataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    # Unet
    model = UNet(in_channels=4, out_channels=1).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # alpha, beta 값 조절하였지만... alpha 20 ----100 ---- 1다 해보고 조절했는데 잘 동작하지 않음
    alpha, beta = 8, 1.0 
    # gamma = 1.0
    num_epochs = 500
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        #같은 이미지로 100번 학습...
        for rgb, sparse_depth, inital_depth, gt_normal in dataloader: 
            rgb = rgb.cuda() #(1, 3, H, W)
            sparse_depth = sparse_depth.cuda() #(1, 1, H, W)
            inital_depth = inital_depth.cuda() #(1, 1, H, W)
            # print(inital_depth.shape)
            gt_normal = gt_normal.cuda() #(1, 3, H, W)
            unet_input = torch.cat([rgb, inital_depth], dim = 1)
            pred = model(unet_input)  # Step2 output: (1, 1, H, W)
            # pred = torch.clamp(pred, min=0.0)
            # Loss1
            l1_pred = pred.squeeze()
            # print(l1_pred.shape)
            l1_sparse_depth = sparse_depth.squeeze() #(H, W)
            # print(l1_sparse_depth.shape) 
            loss_sparse = sparse_depth_loss(l1_pred, l1_sparse_depth)

            # Step3 → normal
            pred_normal = depth_2_normal(pred)  # (1, 3, H, W)
            l2_pre_normal = pred_normal.squeeze(0)
            # print(l2_pre_normal.shape)
            l2_gt_normal = gt_normal.squeeze(0) #(3, H, W)
            # print(l2_gt_normal.shape)
            loss_normal = normal_l2_loss(l2_pre_normal, l2_gt_normal)
            # loss_normal_g = normal_cosine_loss(l2_pre_normal, l2_gt_normal)
            loss = alpha*loss_sparse + beta*loss_normal
            # loss = alpha * loss_sparse + beta * loss_normal + gamma * loss_normal_g

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Epoch {epoch+1}] Total Loss: {total_loss:.4f}")

    # Save model & prediction
    torch.save(model.state_dict(), "./output/unet_trained.pth")
    # with torch.no_grad():
    #     pred = model(unet_input).cpu().squeeze().numpy() #(1, 1, H, W) --> (H, W)
    #     np.save('./output/final_refine_depth.npy', pred)
    # print("Saved final refined depth to ./output/final_refine_depth.npy")

if __name__ == "__main__":
    train_step4()
