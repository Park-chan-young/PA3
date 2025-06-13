import numpy as np


def eval_rmse(pred_depth_path, gt_depth_path, mask_zero=True):

    pred = np.load(pred_depth_path)
    gt = np.load(gt_depth_path)

    if mask_zero:
        mask = gt > 0
        pred = pred[mask]
        gt = gt[mask]

    rmse = np.sqrt(np.mean((pred - gt) ** 2))
    return rmse

if __name__ == "__main__":
    pred_depth_path = './output/final_refine_depth.npy'
    init_depth_path = './output/inital_depth.npy'
    gt_depth_path = './data/data_example/gt.npy'
    rmse_value = eval_rmse(pred_depth_path, gt_depth_path)
    rmse_value_init_depth = eval_rmse(init_depth_path, gt_depth_path)
    print(f"RMSE: {rmse_value:.6f}")
    print(f"RMSE_init_depth: {rmse_value_init_depth:.6f}")