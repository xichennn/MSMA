#architecture picture in test.py on colab
import matplotlib.pyplot as plt
def visualize_centerline(centerline) -> None:
    """Visualize the computed centerline.
    Args:
        centerline: Sequence of coordinates forming the centerline
    """
    line_coords = list(zip(*centerline))
    lineX = line_coords[0]
    lineY = line_coords[1]
    plt.plot(lineX, lineY, "--", color="grey", alpha=1, linewidth=1, zorder=0)
    # plt.text(lineX[0], lineY[0], "s")
    # plt.text(lineX[-1], lineY[-1], "e")
    plt.axis("equal")

def visualize_map(lane_strs, lane_vecs, lane_idcs):
    for i in range(1, len(lane_idcs.unique())):
        lane_start = lane_strs[lane_idcs == i]
        vecs = lane_vecs[lane_idcs == i]
        lane_end = lane_start + vecs
        lane = torch.vstack([lane_start, lane_end[-1,:].reshape(-1, 2)])
        visualize_centerline(lane)

def visualize_traj(prediction, gt, prob, best_mode=True):
    """
    prediction: [num_nodes, num_modes, op_len, 2]
    gt: [num_nodes, op_len, 2]
    prob: [num_nodes, num_modes]
    """
    n, m = prediction.shape[0], prediction.shape[1]

    if best_mode:
      # prs, inds = torch.max(prob, dim=1)

      # for i in range(n):
      #   plt.plot(prediction[i,inds[i],:,0], prediction[i,inds[i],:,1])
      #   plt.text(prediction[i,inds[i],-1,0], prediction[i,inds[i],-1,1],
      #            "{:.2f}".format(prs[i].item()))
      #   plt.plot(gt[i,:,0], gt[i,:,1],'--')
      l2_norm = (torch.norm(prediction[:, :, :, : 2] - \
                              gt.unsqueeze(1), p=2, dim=-1)).sum(dim=-1)
      best_mode = l2_norm.argmin(dim=-1)
      y_pred_best = prediction[torch.arange(gt.shape[0]), best_mode, :, : 2]
      for i in range(n):
        plt.plot(y_pred_best[i,:,0], y_pred_best[i,:,1],'b')
        plt.plot(gt[i,:,0], gt[i,:,1], c='orange', linestyle='--')
        # circle_ncv = plt.Circle((gt[i,0,0], gt[i,0,1]),
        #               1, color='orange')
        # plt.gca().add_patch(circle_ncv)

    else:
      for i in range(n):
        for j in range(m):
          plt.plot(prediction[i,j,:,0], prediction[i,j,:,1])
        plt.plot(gt[i,:,0], gt[i,:,1], c='orange', linestyle='--')
        circle_ncv = plt.Circle((gt[i,0,0], gt[i,0,1]),
                      1, color='orange')
        plt.gca().add_patch(circle_ncv)

def visualize_gt_traj(gt):
    for i in range(gt.shape[0]):
      plt.plot(gt[i,:,0], gt[i,:,1], c='orange', linestyle='--')
def visualize_pred_traj(pred, prob, best_mode=True):
    n, m = pred.shape[0], pred.shape[1]
    if best_mode:
      prs, inds = torch.max(prob, dim=1)
      for i in range(n):
        plt.plot(pred[i,inds[i],:,0], pred[i,inds[i],:,1])
        plt.text(pred[i,inds[i],-1,0], pred[i,inds[i],-1,1],
                 "{:.2f}".format(prs[i].item()))
    else:
      for i in range(n):
        for j in range(m):
          plt.plot(pred[i,j,:,0], pred[i,j,:,1])

def prediction_viz(sample, batch_size, test_set, predictions, probs, batch, masks, mpr=0):
    """
    prediction: [num_nodes, num_modes, op_len, 2]
    gt: [num_nodes, op_len, 2]
    prob: [num_nodes, num_modes]
    """
    s0, s1 = divmod(sample, batch_size)

    #map viz
    lane_vecs = test_set.get(sample).lane_vectors
    lane_strs = test_set.get(sample).lane_pos
    lane_idcs = test_set.get(sample).lane_idcs
    # visualize_map(lane_strs, lane_vecs, lane_idcs)
    #traj viz
    prediction = predictions[s0][batch[s0]==s1,:].cpu() #[num_nodes, num_modes, op_len, 2]
    prob = probs[s0][batch[s0]==s1,:].cpu() #[num_nodes, num_modes]
    mask = masks[s0][batch[s0]==s1].cpu() #[num_nodes]
    gt = test_set.get(sample).y.cpu() #[num_nodes, op_len, 2]
    orig = test_set.get(sample).positions[:,49,:].unsqueeze(1) #[num_nodes, 1, 2]
    # visualize_traj((prediction+orig.unsqueeze(1))[mask], (gt+orig)[mask], prob[mask], best_mode=True)
    #cav
    cav_ori = (gt+orig)[test_set.get(sample).cav_mask]
    cav_mask = test_set.get(sample).cav_mask
    visualize_traj((prediction+orig.unsqueeze(1))[cav_mask], (gt+orig)[cav_mask], prob[cav_mask], best_mode=True)
    for i in range(cav_ori.shape[0]):
      # plt.plot(cav_ori[i,:,0], cav_ori[i,:,1], 'r')
      # circle_cav = plt.Circle((cav_ori[i,0,0], cav_ori[i,0,1]),
      #                 1, color='r')
      l1, = plt.plot(cav_ori[i,0,0], cav_ori[i,0,1], marker=(4, 0, 90), color="r",markersize=5)
      circle_commu = plt.Circle((cav_ori[i,0,0], cav_ori[i,0,1]),
                      65, color='honeydew')
      circle_sensor = plt.Circle((cav_ori[i,0,0], cav_ori[i,0,1]),
                40, color='bisque')
      plt.gca().add_patch(circle_commu)
      plt.gca().add_patch(circle_sensor)
      # plt.gca().add_patch(circle_cav)
    #ncv
    ncv_ori = (gt+orig)[test_set.get(sample).sensor_mask]
    ncv_mask = test_set.get(sample).sensor_mask
    # for i in range(ncv_ori.shape[0]):
    for i in [0,2,3,4,5,7]:
      # plt.plot(ncv_ori[i,:,0], ncv_ori[i,:,1], c='orange')
      l2, = plt.plot(ncv_ori[i,0,0], ncv_ori[i,0,1], marker="o",color="darkorange",markersize=5)
      visualize_traj((prediction+orig.unsqueeze(1))[ncv_mask][i].unsqueeze(0), (gt+orig)[ncv_mask][i].unsqueeze(0), prob[ncv_mask][i].unsqueeze(0), best_mode=True)
      circle_ncv = plt.Circle((ncv_ori[i,0,0], ncv_ori[i,0,1]),
                      1, color='orange')
      plt.gca().add_patch(circle_ncv)
    #cv
    cv_ori = (gt+orig)[test_set.get(sample).commu_mask]
    cv_mask = test_set.get(sample).commu_mask
    
    for i in range(1, cv_ori.shape[0]):
      # plt.plot(cv_ori[i,:,0], cv_ori[i,:,1], 'g')
      l3, = plt.plot(cv_ori[i,0,0], cv_ori[i,0,1], marker="*",color="g",markersize=5)
      visualize_traj((prediction+orig.unsqueeze(1))[cv_mask][i].unsqueeze(0), (gt+orig)[cv_mask][i].unsqueeze(0), prob[cv_mask][i].unsqueeze(0), best_mode=True)
      # circle_cv = plt.Circle((cv_ori[i,0,0], cv_ori[i,0,1]),
      #                 1, color='g')
      # plt.gca().add_patch(circle_cv)

    # #hist_cav
    # positions_cav = test_set.get(sample).positions[[test_set.get(sample).cav_mask]]
    # for i in range(positions_cav.shape[0]):
    #   plt.plot(positions_cav[i,20:50,0], positions_cav[i,20:50,1], 'r--',linewidth=2)
    # #hist_ncv
    # positions_ncv = test_set.get(sample).positions[[test_set.get(sample).sensor_mask]]
    # for i in range(positions_ncv.shape[0]):
    #   plt.plot(positions_ncv[i,20:50,0], positions_ncv[i,20:50,1], c='orange', linestyle='--',linewidth=2)
    # #hist_cv
    # positions_cv = test_set.get(sample).positions[[test_set.get(sample).commu_mask]]
    # for i in range(1, positions_cv.shape[0]):
    #   plt.plot(positions_cv[i,20:50,0], positions_cv[i,20:50,1], 'g--',linewidth=2)
    # # visualize_gt_traj(gt+orig)
    # # visualize_pred_traj((prediction+orig.unsqueeze(1))[mask], prob[mask])
    plt.axis('equal')
    plt.axis('off')
    # # plt.ylim((-60,80))
    # # plt.xlim((-80,60))
    # # plt.xlabel("position_x(m)")
    # # plt.ylabel("position_y(m)")
    # # plt.title('mpr={}'.format(mpr))
sample=452
prediction_viz(sample, batch_size, test4, predictions_cav4_cav4, probs_cav4_cav4, batch_cav4_cav4, mask_cav4_cav4, mpr=0.4)

