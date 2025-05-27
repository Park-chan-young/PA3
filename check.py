import numpy as np
import matplotlib.pyplot as plt

# 파일 경로
sparse_path = "./data/data_example/normal.npy"

# .npy 파일 불러오기
sparse_depth = np.load(sparse_path)

# 배열 정보 확인
print("Shape:", sparse_depth.shape)
print("Data type:", sparse_depth.dtype)
print("Min/Max:", np.min(sparse_depth), "/", np.max(sparse_depth))
print(sparse_depth)

# # 이미지로 저장
# plt.imshow(sparse_depth, cmap='viridis')
# plt.colorbar()
# plt.title("Sparse Depth")
# plt.savefig("normal_sub.png")


# # 1. Load normal.npy
# normal = np.load('./data/data_example/normal.npy')  # Already (H, W, 3)

# # 2. Normalize [-1, 1] → [0, 1]
# normal_img_vis = (normal + 1.0) / 2.0
# normal_img_vis = np.clip(normal_img_vis, 0.0, 1.0)

# # 3. Save as RGB image
# plt.imsave("normal_visual.png", normal_img_vis)
