import numpy as np
import matplotlib.pyplot as plt

# 파일 경로 설정
file_path = './output/final_refine_depth.npy'  # 경로는 필요에 따라 수정

# Depth map 불러오기
depth_map = np.load(file_path)      # shape: (1, H, W)
depth_image = depth_map[0]          # shape: (H, W)

# 시각화 및 저장
plt.figure(figsize=(8, 6))
plt.imshow(depth_image, cmap='viridis')
plt.colorbar(label='Depth (meters)')
plt.title('Refined Depth Map Visualization')
plt.axis('off')

# 이미지 저장
plt.savefig('./test/unet_check_data.png', bbox_inches='tight', pad_inches=0.1)
plt.close()
