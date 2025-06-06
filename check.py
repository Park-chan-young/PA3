import numpy as np
import matplotlib.pyplot as plt

# 파일 경로
sparse_path = "./output/inital_depth.npy"

# .npy 파일 불러오기
sparse_depth = np.load(sparse_path)

# 배열 정보 확인
print("Shape:", sparse_depth.shape)
print("Data type:", sparse_depth.dtype)
print("Min/Max:", np.min(sparse_depth), "/", np.max(sparse_depth))
print(sparse_depth)

# 이미지로 저장
plt.imshow(sparse_depth, cmap='viridis')
plt.colorbar()
plt.title("inital Depth")
plt.savefig("inital_depth.png")


# # # 1. Load normal.npy
# # normal = np.load('./data/data_example/normal.npy')  # Already (H, W, 3)

# # # 2. Normalize [-1, 1] → [0, 1]
# # normal_img_vis = (normal + 1.0) / 2.0
# # normal_img_vis = np.clip(normal_img_vis, 0.0, 1.0)

# # # 3. Save as RGB image
# # plt.imsave("normal_visual.png", normal_img_vis)


# import numpy as np
# import matplotlib.pyplot as plt

# sparse_depth = np.load("./output/temp_depth.npy")

# nonzero = sparse_depth[sparse_depth > 0]
# vmin, vmax = np.min(nonzero), np.max(nonzero)

# masked_sparse = np.ma.masked_where(sparse_depth == 0, sparse_depth)

# rows, cols = np.where(sparse_depth > 0)
# values = sparse_depth[rows, cols]

# plt.imshow(masked_sparse, cmap='plasma', vmin=vmin, vmax=vmax, interpolation='nearest')
# plt.colorbar()
# plt.title("Sparse Depth (Block Style)")
# plt.axis('off')
# plt.savefig("sparse_depth_block.png", bbox_inches='tight', pad_inches=0)

