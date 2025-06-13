import torch
import numpy as np
import cv2
import os
from unet_model_final import UNet  # train 때와 같은 구조여야 함

def load_input(root_dir="./data/data_example/"):
    # 1. RGB 불러오기
    rgb = cv2.imread(os.path.join(root_dir, "rgb.png"))
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb = rgb.transpose(2, 0, 1) / 255.0  # (3, H, W)

    # 2. Sparse Depth 불러오기
    sparse = np.load(os.path.join(root_dir, "sparse_depth.npy"))  # (H, W) 또는 (1, H, W)
    if sparse.ndim == 2:
        sparse = np.expand_dims(sparse, axis=0)  # (1, H, W)

    # 3. Stack + Tensor 변환
    input_tensor = np.concatenate([rgb, sparse], axis=0)  # (4, H, W)
    input_tensor = torch.FloatTensor(input_tensor).unsqueeze(0)  # (1, 4, H, W)

    return input_tensor

def predict(model_path="./output/unet_trained.pth", save_path="./output/final_refine_depth.npy"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 불러오기
    model = UNet(in_channels=4, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 입력 로딩
    input_tensor = load_input().to(device)

    # 추론
    with torch.no_grad():
        pred = model(input_tensor)  # (1, 1, H, W)

    # 저장
    pred_np = pred.squeeze().cpu().numpy()  # (1, H, W)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, pred_np)
    print(f"Saved predicted depth to: {save_path}")

if __name__ == "__main__":
    predict()
