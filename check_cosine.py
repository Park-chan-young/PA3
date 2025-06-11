import numpy as np
import torch
import torch.nn.functional as F

pred = np.load('output/predicted_normal.npy')
gt = np.load('data/data_example/normal.npy')

pred = torch.tensor(pred.reshape(-1, 3))
pred = pred
gt = torch.tensor(gt.reshape(-1, 3))

cos_sim = F.cosine_similarity(pred, gt, dim=1)
print("평균 코사인 유사도:", cos_sim.mean().item())
