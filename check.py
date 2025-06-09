import numpy as np
import matplotlib.pyplot as plt

# 파일 경로
sparse_path = "./output/gt_normal.npy"
# sparse_path = "./data/data_example/normal.npy"
# sparse_path = "./output/inital_depth.npy"
# .npy 파일 불러오기
sparse_depth = np.load(sparse_path)

# 배열 정보 확인
print("Shape:", sparse_depth.shape)
print("Data type:", sparse_depth.dtype)
print("Min/Max:", np.min(sparse_depth), "/", np.max(sparse_depth))
# print(sparse_depth)

# 이미지로 저장
plt.imshow(sparse_depth, cmap='viridis')
plt.colorbar()
plt.title("inital Depth")
plt.savefig("./test/gt_normal.png")


