from utils.Dataset import Dataset
from model_3views_L2L1 import model
from utils.print_result import print_result
import os
import scipy.io as scio

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
'''
Demo for ORL dataset
'''
if __name__ == '__main__':
    data = Dataset('ORL_3views')
    x1, x2, x3, gt = data.load_data_3views()
    x1 = data.normalize(x1, 0)
    x2 = data.normalize(x2, 0)
    x3 = data.normalize(x3, 0)
    n_clusters = len(set(gt))

    act_dg1, act_dg2, act_dg3= 'sigmoid', 'sigmoid', 'sigmoid'
    dims_dg1 = [100, 512, 4096]
    dims_dg2 = [100, 512, 3304]
    dims_dg3 = [100, 512, 6750]

    para_beta = 0.01
    k_RCC = 5
    batch_size = 128
    lr_dg = 1.0e-3
    lr_h = 1.0e-3
    epochs_total = 20
    act = [act_dg1, act_dg2, act_dg3]
    dims = [dims_dg1, dims_dg2, dims_dg3]
    lr = [lr_dg, lr_h]
    epochs_h = 50
    epochs = [epochs_total, epochs_h]

    H, gt = model(x1, x2, x3, gt, para_beta, dims, act, lr, epochs, batch_size,k_RCC)

    scio.savemat('./results/ORL_3views_L2L1.mat', mdict={'H': H, 'gt': gt})
    print_result(n_clusters, H, gt)