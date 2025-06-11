import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from step4_data_lodaer import Step4Dataset
from unet_model_final import UNet
from Depth2Normal import depth_2_normal  # torch 기반 normal 추출 함수

# Loss 함수들
def sparse_l1_loss(pred, sparse):
    mask = (sparse > 0).float()
    return (mask * torch.abs(pred - sparse)).sum() / (mask.sum() + 1e-6)

def normal_cosine_loss(pred_normal, gt_normal):
    pred = nn.functional.normalize(pred_normal, dim=1)
    gt = nn.functional.normalize(gt_normal, dim=1)
    return 1 - (pred * gt).sum(dim=1).mean()

def train_step4():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = Step4Dataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = UNet(in_channels=4, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    alpha, beta = 100.0, 1.0
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for x, sparse, _, normal_gt in dataloader:
            x = x.to(device)  # (1, 4, H, W)
            sparse = sparse.to(device)
            if normal_gt.ndim == 3:
                normal_gt = normal_gt.unsqueeze(0)  # (1, 3, H, W)

            normal_gt = normal_gt.to(device)

            pred = model(x)  # Step2 output: (1, 1, H, W)

            # Loss1
            loss_sparse = sparse_l1_loss(pred, sparse)

            # Step3 → normal
            pred_normal = depth_2_normal(pred)  # (1, 3, H, W)
            loss_normal = normal_cosine_loss(pred_normal, normal_gt)

            loss = alpha * loss_sparse + beta * loss_normal

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Epoch {epoch+1}] Total Loss: {total_loss:.4f}")

    # Save model & prediction
    torch.save(model.state_dict(), "./output/unet_trained.pth")
    with torch.no_grad():
        pred = model(x).cpu().numpy()
        np.save('./output/final_refine_depth.npy', pred[0])
    print("Saved final refined depth to ./output/final_refine_depth.npy")

if __name__ == "__main__":
    train_step4()
