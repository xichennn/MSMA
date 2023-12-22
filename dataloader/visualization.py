import matplotlib.pyplot as plt
from carla_scene_process import CarlaData
import torch
import numpy as np

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

def get_rotate_invariant_trajs(data: CarlaData):

    rotate_mat = torch.empty(data.num_nodes, 2, 2)
    sin_vals = torch.sin(data['rotate_angles'])
    cos_vals = torch.cos(data['rotate_angles'])
    rotate_mat[:, 0, 0] = cos_vals
    rotate_mat[:, 0, 1] = -sin_vals
    rotate_mat[:, 1, 0] = sin_vals
    rotate_mat[:, 1, 1] = cos_vals

    xrot = torch.bmm(data.positions[:,20:50,:], rotate_mat) 
    yrot = torch.bmm(data.y, rotate_mat) 
    # for i in range(xrot.shape[0]):
    #     plt.plot(xrot[i,:,0], xrot[i,:,1])
    #     plt.plot(data.x_sensor[i,:,0], data.x_sensor[i,:,1],'--')
    # for i in range(yrot.shape[0]):
    #     plt.plot(yrot[i,:,0], yrot[i,:,1])
    #     plt.plot(data.y[i,:,0], data.y[i,:,1],'--')

    return xrot, yrot, rotate_mat
def viz_devectorize(xrot_vec):
    """
    xrot_vec: rotated vector [N,30,2]
    """
    x_devec = torch.cumsum(xrot_vec, dim=1)
    # translate back to original location
    x_devec_ori = x_devec - x_devec[:,-1,:]
    for i in range(x_devec_ori.shape[0]):
        plt.plot(x_devec_ori[i,:,0], x_devec_ori[i,:,1])

def local_invariant_scenes(data: CarlaData):
    xrot, yrot, rotate_mat = get_rotate_invariant_trajs(data)
    lane_str, lane_vectors = data.lane_pos, data.lane_vectors
    lane_idcs = data.lane_idcs
    # # visualize the centerlines
    # lane_pos = data.lane_pos
    # lane_vectors = data.lane_vectors
    # lane_idcs = data.lane_idcs
    # for i in torch.unique(lane_idcs):
    #     lane_str = lane_pos[lane_idcs == i]
    #     lane_vector = lane_vectors[lane_idcs == i]
    #     lane_end = lane_str + lane_vector
    #     lane = torch.vstack([lane_str, lane_end[-1,:].reshape(-1, 2)])
    #     visualize_centerline(lane)

    #rotate locally
    edge_index = data.lane_actor_index

    lane_rotate_mat = rotate_mat[edge_index[1]]
    lane_vectors_rot = torch.bmm(lane_vectors[edge_index[0]].unsqueeze(-2), lane_rotate_mat).squeeze(-2) #[#, 2]
    lane_pos_rot = torch.bmm(lane_str[edge_index[0]].unsqueeze(-2), lane_rotate_mat).squeeze(-2) #[#, 2]
    
    #viz local map and traj
    for i in range(data.num_nodes):
        #traj viz
        plt.plot(xrot[i,:,0], xrot[i,:,1])
        plt.text(xrot[i,-1,0], xrot[i,-1,1], "q")
        #map viz
        lane_idx_i = (edge_index[1] == i).nonzero().squeeze()
        for j in lane_idx_i:
        # lane_str_i = lane_pos_rot[edge_index[1] == i]
            lane_str_i = lane_pos_rot[j].unsqueeze(0) #[1,2]
            # lane_vector_i = lane_vectors_rot[edge_index[1] == i]
            lane_vector_i = lane_vectors_rot[j].unsqueeze(0)
            lane_end_i = lane_str_i + lane_vector_i
            lane_i = torch.vstack([lane_str_i, lane_end_i])
            visualize_centerline(lane_i)
    

    #for each agent, get self-centered maps
    for i in range(xrot.shape[0]):
        lane_vector_i = lane_vectors_rot[edge_index[1]==i]
        lane_pos_i = lane_pos_rot[edge_index[1]==i]
        lane_end_i = lane_vector_i + lane_pos_i
        lane_i = torch.vstack([lane_pos_i, lane_end_i[-1,:].reshape(-1, 2)]) #[L, 2]
        
        visualize_centerline(lane_i)

    # visualize the centerlines
    lane_pos = data.lane_pos
    lane_vectors = data.lane_vectors
    lane_idcs = data.lane_idcs
    for i in torch.unique(lane_idcs):
        lane_str = lane_pos[lane_idcs == i]
        lane_vector = lane_vectors[lane_idcs == i]
        lane_end = lane_str + lane_vector
        lane = torch.vstack([lane_str, lane_end[-1,:].reshape(-1, 2)])
        visualize_centerline(lane)

    for i in range(data.x.shape[0]):
        lane_vector_i = lane_vectors[edge_index[0]][edge_index[1]==i]
        lane_pos_i = lane_str[edge_index[0]][edge_index[1]==i]
        lane_end_i = lane_vector_i + lane_pos_i
        lane_i = torch.vstack([lane_pos_i, lane_end_i[-1,:].reshape(-1, 2)]) #[L, 2]
        
        visualize_centerline(lane_i)

def viz_lane_rot():
    pass

def tensor_viz(node_features_all, cav_mask, commu_mask, sensor_mask):

    axes = [8, 16, 3]
    filled = np.ones(axes, dtype=np.bool)
    colors = np.empty(axes + [4], dtype=np.float32)
    alpha = 0.5
    colors[:] = [1, 1, 1, alpha]
    colors[cav_mask,:,0] = [1, 0, 0, alpha]
    colors[commu_mask,:,1] = [0, 1, 0, alpha]
    colors[sensor_mask,:,2] = [0, 0, 1, alpha]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.voxels(filled, facecolors=colors, edgecolors='grey',shade=True)
    plt.show()
    plt.axis('off')


