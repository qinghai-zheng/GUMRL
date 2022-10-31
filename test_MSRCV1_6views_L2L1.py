from utils.Dataset import Dataset
from model_6views_L2L1 import model
from utils.print_result import print_result
import os
import scipy.io as scio

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
'''
Demo for MSRCV1 dataset
'''
if __name__ == '__main__':
    data = Dataset('MSRCV1_6views')
    x1, x2, x3, x4, x5, x6, gt = data.load_data_6views()
    x1 = data.normalize(x1, 0)
    x2 = data.normalize(x2, 0)
    x3 = data.normalize(x3, 0)
    x4 = data.normalize(x4, 0)
    x5 = data.normalize(x5, 0)
    x6 = data.normalize(x6, 0)
    n_clusters = len(set(gt))

    act_dg1, act_dg2, act_dg3, act_dg4, act_dg5, act_dg6 = 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid'
    dims_dg1 = [32, 512, 1302]
    dims_dg2 = [32, 48]
    dims_dg3 = [32, 128, 512]
    dims_dg4 = [32, 100]
    dims_dg5 = [32, 256]
    dims_dg6 = [32, 210]

    k_RCC = 5
    para = 0.1
    batch_size = 64
    lr_dg = 1.0e-3
    lr_h = 1.0e-3
    epochs_total = 20
    act = [act_dg1, act_dg2, act_dg3, act_dg4, act_dg5, act_dg6]
    dims = [dims_dg1, dims_dg2, dims_dg3, dims_dg4, dims_dg5, dims_dg6]
    lr = [lr_dg, lr_h]
    epochs_h = 50
    epochs = [epochs_total, epochs_h]

    H, gt = model(x1, x2, x3, x4, x5, x6, gt, para, dims, act, lr, epochs, batch_size, k_RCC)

    scio.savemat('./results/MSRCV1_6views_L2L1.mat', mdict={'H': H, 'gt': gt})
    print_result(n_clusters, H, gt)