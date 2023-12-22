import torch
from sklearn.cluster import KMeans
# import psutil
# import ray
# from scipy.spatial.distance import cdist

#Initialize device:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # Initialize ray:
# num_cpus = psutil.cpu_count(logical=False)
# ray.init(num_cpus=num_cpus, log_to_driver=False)

def k_means_anchors(k, train_loader):
    """
    Extract anchors for multipath/covernet using k-means on train set trajectories
    gt_y: [num_v, op_len, 2]
    train_loader: CarlaData
    """

    trajectories = []
    rotate_imat= []
    for i, data in enumerate(train_loader):
        trajectories.append(data.y)
        rotate_imat.append(data.rotate_imat)

    traj_all = torch.cat(trajectories, dim=0)
    rotate_imat_all = torch.cat(rotate_imat, dim=0)
    traj_all_rot = torch.matmul(traj_all, rotate_imat_all)

    clustering = KMeans(n_clusters=k).fit(traj_all_rot.reshape((traj_all_rot.shape[0], -1)))
    op_len, op_dim = traj_all_rot.shape[1], traj_all_rot.shape[2]
    anchors = torch.zeros((k, op_len, op_dim)).to(device)
    for i in range(k):
        anchors[i] = torch.mean(traj_all_rot[clustering.labels_==i], axis=0)
    # for i in range(traj_all_rot.shape[0]):
    #     plt.plot(traj_all_rot[i, :, 0], traj_all_rot[i, :, 1])
    # for i in range(anchors.shape[0]):
    #     plt.plot(anchors[i, :, 0], anchors[i, :, 1])
    
    return anchors


def bivariate_gaussian_activation(ip: torch.Tensor) -> torch.Tensor:
    """
    Activation function to output parameters of bivariate Gaussian distribution
    """
    mu_x = ip[..., 0:1]
    mu_y = ip[..., 1:2]
    sig_x = ip[..., 2:3]
    sig_y = ip[..., 3:4]
    rho = ip[..., 4:5]
    sig_x = torch.exp(sig_x)
    sig_y = torch.exp(sig_y)
    rho = torch.tanh(rho)
    out = torch.cat([mu_x, mu_y, sig_x, sig_y, rho], dim = -1)

    return out