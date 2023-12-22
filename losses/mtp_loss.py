# source: https://github.com/nachiket92/PGP/blob/main/metrics/mtp_loss.py
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

    def forward(self, y_pred, y_gt, log_probs):
        """
        params:
        :y_pred: [num_vehs, num_modes, op_len, op_dim]  
        :y_gt: [num_vehs, op_len, 2]
        :log_probs: probability for each mode [num_vehs, num_modes]
        :alpha: float, relative weight assigned to classification component, compared to regression component
                    of loss, default: 1
        """
        alpha = self.alpha
        use_variance = self.use_variance
        # Obtain mode with minimum ADE with respect to ground truth:
        op_len = y_pred.shape[2]
        pred_params = 5 if use_variance else 2

        errs, inds = min_ade(y_pred, y_gt)
        inds_rep = inds.repeat(op_len, pred_params, 1, 1).permute(3, 2, 0, 1)

        # Calculate MSE or NLL loss for trajectories corresponding to selected outputs:
        traj_best = y_pred.gather(1, inds_rep).squeeze(dim=1)
        # # devectorize traj_best
        # for i in range(1,50):
        #     traj_best[:,i,:] += traj_best[:,i-1,:]

        if use_variance:
            l_reg = traj_nll(traj_best, y_gt)
        else:
            l_reg = errs

        # Compute classification loss
        l_class = - torch.squeeze(log_probs.gather(1, inds.unsqueeze(1)))

        loss = l_reg + alpha * l_class
        loss = torch.mean(loss)

        return loss
    


