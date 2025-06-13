import numpy as np
import torch
import matplotlib.pyplot as plt


def main():
    gt = torch.from_numpy(np.load("./data/data_example/normal.npy")).float()
    pred = torch.from_numpy(np.load("./output/predicted_normal.npy")).float()

    gt = torch.nn.functional.normalize(gt, dim=1)
    pred = torch.nn.functional.normalize(pred, dim=1)

    cos_sim = (gt * pred).sum(dim=1).squeeze(0).clamp(-1 + 1e-6, 1 - 1e-6)
    angle_error = torch.rad2deg(torch.acos(cos_sim)).squeeze().numpy()  # shape: (H, W)


    plt.figure(figsize=(10, 6))
    plt.imshow(angle_error, cmap="plasma", origin="upper")
    plt.colorbar(label="Angular Error (°)")
    plt.title("Angular Error between GT and Predicted Normals")
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()