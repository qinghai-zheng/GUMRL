from utils.cluster import cluster
import warnings

warnings.filterwarnings('ignore')

def print_result(n_clusters, H, gt, count=10):
    acc_avg, acc_std, nmi_avg, nmi_std, ri_avg, ri_std, f1_avg, f1_std, precision_avg, precision_std = cluster(n_clusters, H, gt, count=count)
    print('clustering h : nmi = {:.4f}, acc = {:.4f}, f1 = {:.4f}, precision = {:.4f}, ri = {:.4f}'.format
          (nmi_avg, acc_avg, f1_avg, precision_avg, ri_avg))

