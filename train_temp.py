from torch.utils.data import DataLoader
from unet_model_final import UNet
from data_loader import SingleExampleDataset  # 파일명에 맞게 import
import torch
import torch.nn as nn
import torch.optim as optim

def train():
    dataset = SingleExampleDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = UNet(in_channels=4, out_channels=1)
    model = model.cuda() if torch.cuda.is_available() else model

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss()

    for epoch in range(10):
        total_loss = 0.0
        for x, gt in dataloader:
            x = x.cuda() if torch.cuda.is_available() else x
            gt = gt.cuda() if torch.cuda.is_available() else gt

            pred = model(x)
            loss = criterion(pred, gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "./output/unet_trained.pth")

if __name__ == "__main__":
    train()
