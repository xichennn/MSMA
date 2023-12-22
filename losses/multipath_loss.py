import torch
import torch.nn as nn
from metrics.metric import min_ade, traj_nll

class NLLloss(nn.Module):
    """
    MTP loss modified to include variances. Uses MSE for mode selection. Can also be used with
    Multipath outputs, with residuals added to anchors.
    """
    def __init__(self, alpha=0.2, use_variance=True):
        """
        Initialize MTP loss
        :param args: Dictionary with the following (optional) keys
            use_variance: bool, whether or not to use variances for computing regression component of loss,
                default: False
            alpha: float, relative weight assigned to classification component, compared to regression component
                of loss, default: 1
        """
        super(NLLloss, self).__init__()
        self.use_variance = use_variance
        self.alpha = alpha

    def forward(self, y_pred, y_true, log_probs, anchors):
        """
        params:
        :y_pred: [num_nodes, num_modes, op_len, 2]  
        :y_true: [num_nodes, op_len, 2]
        :log_probs: probability for each mode [N_B, N_M]
        :anchors: [num_modes, op_len, 2]
        where N_B is batch_size, N_M is num_modes, N_T is target_len 
        """

        
        num_nodes = y_true.shape[0]
        trajectories = y_pred
        anchor_probs = log_probs
        
        #find the nearest anchor mode to y_true
        #[1, num_modes, op_len, 2] - [num_nodes, 1, op_len, 2] = [num_nodes, num_modes, op_len, 2]
        distance_to_anchors = torch.sum(torch.linalg.vector_norm(anchors.unsqueeze(0) - y_true.unsqueeze(1),
                    dim=-1),dim=-1) #[num_nodes, num_modes]

        nearest_mode = distance_to_anchors.argmin(dim=-1) #[num_nodes]
        nearest_mode_indices = torch.stack([torch.arange(num_nodes,dtype=torch.int64),nearest_mode],dim=-1)

        loss_cls = -log_probs[torch.arange(num_nodes),nearest_mode].squeeze() #[N_B]
        
        trajectories_xy = y_pred + anchors.unsqueeze(0)
        # l2_norm = (torch.norm(trajectories_xy[:, :, :, :2] - y_true.unsqueeze(1), p=2, dim=-1)).sum(dim=-1)  # [num_nodes, num_modes]

        nearest_trajs = trajectories_xy[torch.arange(num_nodes),nearest_mode,:,:].squeeze()
        residual_trajs = y_true - nearest_trajs

        loss_reg = torch.mean(torch.square(residual_trajs[:,:,0])+torch.square(residual_trajs[:,:,1]), dim=-1)
        dx = residual_trajs[:,:,0]
        dy = residual_trajs[:,:,1]

        loss = loss_reg + self.alpha * loss_cls
        loss = torch.mean(loss)

        return loss


