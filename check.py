import numpy as np
import matplotlib.pyplot as plt

# 파일 경로
# sparse_path = "./output/chk_normal.npy"
# sparse_path = "./data/data_example/sparse_depth.npy"
# sparse_path = "./output/normal_chk/cam_coords.npy"
# sparse_path = "./data/data_example/normal.npy"
sparse_path = "./output/Inital_depth.npy"
# sparse_path = './output/Final_refined_depth.npy'
# sparse_path = './output/chk_normal.npy'
# .npy 파일 불러오기
sparse_depth = np.load(sparse_path)
# sparse_depth = sparse_depth.squeeze()
# 배열 정보 확인
print("Shape:", sparse_depth.shape)
print("Data type:", sparse_depth.dtype)
print("Min/Max:", np.min(sparse_depth), "/", np.max(sparse_depth))
# print(sparse_depth)

# 이미지로 저장
plt.imshow(sparse_depth, cmap='viridis')
plt.colorbar()
# plt.title("final Depth")
# plt.title("final depth")
# plt.savefig("./test/final_refine_depth.png")

plt.title("Inital_depth")
plt.savefig("./test/Inital_depth.png")

