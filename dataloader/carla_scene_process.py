"""
process the '.csv' files, save as '.pt' files
"""
import os
import sys
import numpy as np
import pandas as pd
import copy
from os.path import join as pjoin

# sys.path.append('/Users/xichen/Documents/paper2-traj-pred/carla-data/MSMA')
from dataloader.utils import lane_segment, load_xml
from dataloader.utils.lane_sampling import Spline2D, visualize_centerline
import matplotlib.pyplot as plt

from typing import List, Optional, Tuple

import torch
torch.manual_seed(30)
import torch.nn as nn
from torch_geometric.data import Data, HeteroData
from torch_geometric.data import Dataset
from typing import Callable, Dict, List, Optional, Tuple, Union
from itertools import permutations, product

class scene_processed_dataset(Dataset):
    def __init__(self, 
                 root:str,
                 split:str,
                 transform: Optional[Callable] = None,
                 mpr:float = 0.) ->None:
        
        self._split = split
        self.mprs = [0., 0.2, 0.4, 0.6, 0.8]
        self.mprs_dir = ["mpr0", "mpr2", "mpr4", "mpr6", "mpr8"]

        self.mpr_dir = self.mprs_dir[self.mprs.index(mpr)]

        if split == 'train':
            self._directory = 'train'
        elif split == 'val':
            self._directory = 'val'
        elif split == 'test':
            self._directory = 'test'
        else:
            raise ValueError(split + ' is not valid')
        self.root = root
        self._raw_file_names = os.listdir(self.raw_dir)
        self._processed_file_names = [os.path.splitext(f)[0] + '.pt' for f in self.raw_file_names]
        self._processed_paths = [os.path.join(self.processed_dir, f) for f in self._processed_file_names]
        super(scene_processed_dataset, self).__init__(root)

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.mpr_dir, self._directory)

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.mpr_dir, self._directory)

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return self._raw_file_names

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self._processed_file_names

    @property
    def processed_paths(self) -> List[str]:
        return self._processed_paths

    def process(self) -> None:
        pass

    def len(self) -> int:
        return len(self._raw_file_names)

    def get(self, idx) -> Data:
        return torch.load(self.processed_paths[idx])

class scene_process():
    def __init__(self, split="train", obs_len=50, obs_range=50,
                 fut_len=50, cv_range=50, av_range=30, map_radius=75,
                 local_radius=30, mpr=0, noise_var=0.1, delay_frame=1,
                 normalized=True,save_dir=None, 
                 csv_folder="scene_mining_intermediate/small_data"):
        
        self.COLOR_DICT = {"CAV": "#d33e4c", "CV": "g", "NCV": "darkorange"}
        self.split = split
        self.obs_len = obs_len
        self.obs_range = obs_range
        self.fut_len = fut_len
        self.cv_range = cv_range
        self.av_range = av_range
        self.map_radius = map_radius
        self.local_radius = local_radius
        self.mpr = mpr
        self.noise_var = noise_var
        self.delay_frame = delay_frame
        self.normalized = normalized
        self.save_dir = save_dir
        self.abs_path = os.path.dirname(__file__)
        self.loader = CarlaCSVLoader(pjoin(self.abs_path,"../../{}".format(csv_folder)))

    def __getitem__(self, idx, dir_post="train"):
        f_path = self.loader.seq_list[idx]
        df = pd.read_csv(f_path)
        path, seq_f_name_ext = os.path.split(f_path)

        return self.process_and_save(df, seq_id=idx, file_name=seq_f_name_ext, 
                                     dir_=pjoin(self.abs_path,"../../scene_mining_cav/", self.save_dir, dir_post))
    
    def process_and_save(self, df, seq_id, file_name, dir_=None):
        """
        save the feature in the data sequence in a single csv files
        :param dataframe: DataFrame, the data frame
        :param set_name: str, the name of the folder name, exp: train, eval, test
        :param file_name: str, the name of csv file
        :param dir_: str, the directory to store the csv file
        :return:
        """
        df_processed = self.process(df, seq_id)
        self.save(df_processed, file_name, dir_)

    def save(self, df, file_name, dir_=None):
        """
        save the feature in the data sequence in a single csv files
        :param df: DataFrame, the dataframe encoded
        :param set_name: str, the name of the folder name, exp: train, eval, test
        :param file_name: str, the name of csv file
        :param dir_: str, the directory to store the csv file
        :return:
        """
        if not os.path.exists(dir_):
            os.makedirs(dir_)

        torch.save(df, os.path.join(dir_, file_name + '.pt'))
      
    def get_map_polygon_bbox(self):
        rel_path = "../carla_data/Town03.osm"
        roads = load_xml.load_lane_segments_from_xml(pjoin(self.abs_path, rel_path))
        polygon_bboxes, lane_starts, lane_ends = load_xml.build_polygon_bboxes(roads)
        self.roads = roads
        self.polygon_bboxes = polygon_bboxes
        self.lane_starts = lane_starts
        self.lane_ends = lane_ends

    def process(self, df, seq_id):
        self.get_map_polygon_bbox()

        kwargs = self.get_scene_feats(df, seq_id, self.map_radius, self.local_radius, self.split)
        data = CarlaData(**kwargs)

        return data

    def get_scene_feats(self, df, seq_id, radius, local_radius, split="train"):

        # filter out actors that are unseen during the historical time steps
        timestamps = list(np.sort(df['frame'].unique()))
        historical_timestamps = timestamps[: 50]
        historical_df = df[df['frame'].isin(historical_timestamps)]
        actor_ids = list(historical_df['vid'].unique())
        
        # filter out unmoved actors
        # actor_ids = self.remove_unmoved_ids(df, actor_ids)

        df = df[df['vid'].isin(actor_ids)]
        num_nodes = len(actor_ids)

        objs = df.groupby(['vid', 'obj_type_mpr_02', 'obj_type_mpr_04', 'obj_type_mpr_06', 'obj_type_mpr_08', 'in_av_range']).groups
        keys = list(objs.keys())
    
        vids = [x[0] for x in keys]
        actor_indices = [vids.index(x) for x in actor_ids]
        obj_type_02 = [keys[i][1] for i in actor_indices]
        obj_type_04 = [keys[i][2] for i in actor_indices]
        obj_type_06 = [keys[i][3] for i in actor_indices]
        obj_type_08 = [keys[i][4] for i in actor_indices]
        in_av_range = [keys[i][5] for i in actor_indices]

        # obj_type_02 = [x[1] for x in keys]


        # cav_idx = obj_type.index("cav")
        cav_idx = np.where(np.asarray(obj_type_02)=="cav")[0] #np array
        # ncv_idx = np.where(np.asarray(obj_type)=="ncv")[0] #np array
        # cv_idx = np.where(np.asarray(obj_type)=="cv")[0] #np array

        cav_df = df[df['obj_type_mpr_02'] == 'cav'].iloc
        # cv_df = df[df['obj_type_mpr_02'] == 'cv'].iloc
        # ncv_df = df[df['obj_type_mpr_02'] == 'ncv'].iloc

        # make the scene centered at CAV
        origin = torch.tensor([cav_df[49]['position_x'], cav_df[49]['position_y']], dtype=torch.float)
        cav_heading_vector = origin - torch.tensor([cav_df[48]['position_x'], cav_df[48]['position_y']], dtype=torch.float)
        theta = torch.atan2(cav_heading_vector[1], cav_heading_vector[0])
        rotate_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                                    [torch.sin(theta), torch.cos(theta)]])
        
        # initialization
        x = torch.zeros(num_nodes, 100, 2, dtype=torch.float)
        edge_index = torch.LongTensor(list(permutations(range(num_nodes), 2))).t().contiguous()
        padding_mask = torch.ones(num_nodes, 100, dtype=torch.bool)
        bos_mask = torch.zeros(num_nodes, 50, dtype=torch.bool)
        rotate_angles = torch.zeros(num_nodes, dtype=torch.float)

        for actor_id, actor_df in df.groupby('vid'):
            node_idx = actor_ids.index(actor_id)
            node_steps = [timestamps.index(timestamp) for timestamp in actor_df['frame']]
            padding_mask[node_idx, node_steps] = False
            if padding_mask[node_idx, 49]:  # make no predictions for actors that are unseen at current timestep
                padding_mask[node_idx, 50:] = True
            xy = torch.from_numpy(np.stack([actor_df['position_x'].values, actor_df['position_y'].values], axis=-1)).float() #[100,2]
            x[node_idx, node_steps] = torch.matmul(xy - origin, rotate_mat)
            node_historical_steps = list(filter(lambda node_step: node_step < 50, node_steps))
            if len(node_historical_steps) > 1:  # calculate the heading of the actor (approximately)
                heading_vector = x[node_idx, node_historical_steps[-1]] - x[node_idx, node_historical_steps[-2]]
                rotate_angles[node_idx] = torch.atan2(heading_vector[1], heading_vector[0])
            else:  # make no predictions for the actor if the number of valid time steps is less than 2
                padding_mask[node_idx, 50:] = True

        bos_mask[:, 0] = ~padding_mask[:, 0]
        bos_mask[:, 1: 50] = padding_mask[:, : 49] & ~padding_mask[:, 1: 50]

        #positions are transformed absolute x, y coordinates
        positions = x.clone()

        #reformat encode strs and bools, CAV:1, CV:2, NCV:3
        obj_type_mapping = {"cav":1, "cv":2, "ncv":3}
        obj_type_02_ = torch.tensor([obj_type_mapping[x] for x in obj_type_02])
        obj_type_04_ = torch.tensor([obj_type_mapping[x] for x in obj_type_04])
        obj_type_06_ = torch.tensor([obj_type_mapping[x] for x in obj_type_06])
        obj_type_08_ = torch.tensor([obj_type_mapping[x] for x in obj_type_08])
        in_av_range_ = torch.tensor([1 if in_av_range[i]==True else 0 for i in range(len(in_av_range))])

        #get masks for different data sources
        types = [obj_type_02_, obj_type_04_, obj_type_06_, obj_type_08_]
        mprs = [0.2, 0.4, 0.6, 0.8]
        cav_mask, commu_mask, sensor_mask = self.get_masks(self.mpr, mprs, types, in_av_range_)
        positions_hist = positions[:,:50,:].clone()
        x_cav = positions_hist[cav_mask][:,20:50,:]
        x_commu = positions_hist[commu_mask]
        x_sensor = positions_hist[sensor_mask]

        #inject errors to different data sources
        x_sensor_noise, padding_mask_noise = self.get_noisy_x(x_sensor, padding_mask[sensor_mask], self.noise_var)
        x_commu_delay, padding_mask_delay = self.get_delayed_x(x_commu, padding_mask[commu_mask], self.delay_frame)
        
        #get vectorized x
        x_cav_vec = self.get_vectorized_x(x_cav, padding_mask[cav_mask][:,20:50])
        x_commu_delay_vec = self.get_vectorized_x(x_commu_delay, padding_mask_delay)
        x_sensor_noise_vec = self.get_vectorized_x(x_sensor_noise, padding_mask_noise)


        y = torch.where((padding_mask[:, 49].unsqueeze(-1) | padding_mask[:, 50:]).unsqueeze(-1),
                                torch.zeros(num_nodes, 50, 2),
                                x[:, 50:] - x[:, 49].unsqueeze(-2))


        lane_pos, lane_vectors, lane_idcs,lane_actor_index, lane_actor_attr = \
            self.get_lane_feats(origin, rotate_mat, num_nodes, positions, radius, local_radius)
        
        #get rotate-invariant matrix
        rotate_imat = torch.empty(num_nodes, 2, 2)
        sin_vals = torch.sin(rotate_angles)
        cos_vals = torch.cos(rotate_angles)
        rotate_imat[:, 0, 0] = cos_vals
        rotate_imat[:, 0, 1] = -sin_vals
        rotate_imat[:, 1, 0] = sin_vals
        rotate_imat[:, 1, 1] = cos_vals

        return {
            'x_cav': x_cav_vec,  # [1, 30, 2]
            'x_commu': x_commu_delay_vec,  # [N1, 30, 2]
            'x_sensor': x_sensor_noise_vec,  # [N2, 30, 2]
            'cav_mask': cav_mask,  # [N]
            'commu_mask': commu_mask,  # [N]
            'sensor_mask': sensor_mask,  # [N]
            'positions': positions,  # [N, 100, 2]
            'edge_index': edge_index,  # [2, N x (N - 1)]
            'y': y,  # [N, 50, 2]
            'y_commu_ori': x_commu_delay[:,-1,:], #[N,2]
            'y_sensor_ori': x_sensor_noise[:,-1,:], #[N,2]
            'seq_id': seq_id, #int
            # 'file_name': file_name, #str
            'num_nodes': num_nodes,
            'padding_mask': padding_mask,  # [N, 100]
            'bos_mask': bos_mask,  # [N, 50]
            'rotate_angles': rotate_angles,  # [N]
            'rotate_imat': rotate_imat, #[N, 2, 2]
            'lane_vectors': lane_vectors,  # [L, 2]
            'lane_pos': lane_pos, #[L, 2]
            'lane_idcs': lane_idcs, #[L]
            'lane_actor_index': lane_actor_index, 
            'lane_actor_attr': lane_actor_attr,
            'mpr': self.mpr,
            # 'obj_type_02': obj_type_02_, #tensor(str) [N]
            # 'obj_type_04': obj_type_04_, #tensor(str) [N]
            # 'obj_type_06': obj_type_06_, #tensor(str) [N]
            # 'obj_type_08': obj_type_08_, #tensor(str) [N]
            # 'in_av_range': in_av_range_, #tensor(bool) [N]
            'origin': origin.unsqueeze(0),
            'theta': theta,
            'rotate_mat': rotate_mat
        }
    
    def get_lane_feats(self, origin, rotate_mat, num_nodes, positions, radius=75, local_radius=30):

        road_ids = load_xml.get_road_ids_in_xy_bbox(self.polygon_bboxes, self.lane_starts, self.lane_ends, self.roads, origin[0], origin[1], radius)
        road_ids = copy.deepcopy(road_ids)

        lanes=dict()
        for road_id in road_ids:
            road = self.roads[road_id]
            ctr_line = torch.from_numpy(np.stack(((self.roads[road_id].l_bound[:,0]+self.roads[road_id].r_bound[:,0])/2, 
                            (self.roads[road_id].l_bound[:,1]+self.roads[road_id].r_bound[:,1])/2),axis=-1))
            ctr_line = torch.matmul(ctr_line.float() - origin, rotate_mat)

            x, y = ctr_line[:,0], ctr_line[:,1]

            """getting polygons requires original centerline"""
            polygon, _, _ = load_xml.build_polygon_bboxes({road_id: self.roads[road_id]})
            polygon_x = torch.from_numpy(np.array([polygon[:,0],polygon[:,0],polygon[:,2],polygon[:,2],polygon[:,0]]))
            polygon_y = torch.from_numpy(np.array([polygon[:,1],polygon[:,3],polygon[:,3],polygon[:,1],polygon[:,1]]))
            polygon_reshape = torch.cat([polygon_x,polygon_y],dim=-1) #shape(5,2)

            road.centerline = ctr_line
            road.polygon = torch.matmul(polygon_reshape.float() - origin.reshape(-1, 2),rotate_mat).float() 
            lanes[road_id] = road

        lane_ids = list(lanes.keys())
        lane_pos, lane_vectors = [], []
        for lane_id in lane_ids:
            lane = lanes[lane_id]
            ctrln = lane.centerline
            lane_pos.append(ctrln[:-1])#lane starting point
            lane_vectors.append(ctrln[1:]-ctrln[:-1])#length between waypoints

        lane_idcs = []
        count = 0
        for i, position in enumerate(lane_pos):
            lane_idcs.append(i*torch.ones(len(position)))
            count += len(position)

        lane_idcs = torch.cat(lane_idcs, dim=0)
        lane_pos = torch.cat(lane_pos, dim=0)
        lane_vectors = torch.cat(lane_vectors, dim=0)

        lane_actor_index = torch.LongTensor(list(product(torch.arange(lane_vectors.size(0)), \
                                                         torch.arange(num_nodes)))).t().contiguous()
        lane_actor_attr = \
        lane_pos[lane_actor_index[0]] - positions[:,49,:][lane_actor_index[1]]
        mask = torch.norm(lane_actor_attr, p=2, dim=-1) < local_radius
        lane_actor_index = lane_actor_index[:, mask]
        lane_actor_attr = lane_actor_attr[mask]
        

        return lane_pos, lane_vectors, lane_idcs, lane_actor_index, lane_actor_attr

    
    def get_vectorized_x(self, x0, padding_mask):
        '''
        x: torch.Tensor: [n, 30, 2]
        padding_mask: torch.Tensor:[n, 30]
        '''
        x = x0.clone()
        x[:, 1: 30] = torch.where((padding_mask[:, : 29] | padding_mask[:, 1: 30]).unsqueeze(-1),
                                    torch.zeros(x.shape[0], 29, 2),
                                    x[:, 1: 30] - x[:, : 29])
        x[:, 0] = torch.zeros(x.shape[0], 2)

        return x
    
    def get_masks(self, mpr, mprs, types, in_av_range):
        #ncv in av range 
        #and all cv
        if mpr == 0: 
            cav_mask = types[0]==1
            commu_mask = torch.zeros(cav_mask.shape)==True
            sensor_mask = (types[0]!=1) & (in_av_range==1)
        else:
            type_idx = mprs.index(mpr)
            cav_mask = types[type_idx]==1
            commu_mask = types[type_idx]==2
            sensor_mask = (types[type_idx]!=1) & (in_av_range==1)
    
        return cav_mask, commu_mask, sensor_mask
    
    def get_noisy_x(self, x, padding_mask, var=0.1):
        """
        get noisy feats for sensor data
        x: torch.Tensor of shape(n, 50, 2)

        return 
        noise_x: torch.Tensor of shape(n, 30, 2)
        """
        noise = torch.normal(0, var, x.shape)
        
        return (x+noise)[:,20:,:], padding_mask[:,20:]

    def get_delayed_x(self, x, padding_mask, lag=1):
        """
        get delayed feats of communication data
        x: torch tensor of shape(n, 50, 2)
        lag: number of frames in [0:20]

        return 
        delayed_x: torch.Tensor of shape(n, 30, 2)
        """
        if lag<0 or lag>20:
            raise Exception("lag must be in the range(0,20)")

        delayed_x = x[:,20-lag:-lag,:]

        return delayed_x, padding_mask[:, 20-lag:50-lag]
    
    def visualize_rotation_invariance(self, x, rotate_angles):

        rotate_mat = torch.empty(x.shape[0], 2, 2)
        sin_vals = torch.sin(rotate_angles)
        cos_vals = torch.cos(rotate_angles)
        rotate_mat[:, 0, 0] = cos_vals
        rotate_mat[:, 0, 1] = -sin_vals
        rotate_mat[:, 1, 0] = sin_vals
        rotate_mat[:, 1, 1] = cos_vals

        xrot = torch.bmm(x, rotate_mat)
        for i in range(xrot.shape[0]):
            plt.plot(xrot[i,:50,0], xrot[i,:50,1], 'b')
            plt.plot(xrot[i,50:,0], xrot[i,50:,1], 'b--')
            plt.plot(x[i,:50,0], x[i,:50,1],'r')
            plt.plot(x[i,50:,0], x[i,50:,1],'r--')
        return xrot
    
    def visualize_data(self, data):
        """
        visualize the extracted data, and exam the data
        """
        fig = plt.figure(0, figsize=(8, 7))
        fig.clear()

        # visualize the centerlines
        lane_pos = data.lane_pos
        lane_vectors = data.lane_vectors
        lane_idcs = data.lane_idcs
        for i in np.unique(lane_idcs):
            lane_str = lane_pos[lane_idcs == i]
            lane_vector = lane_vectors[lane_idcs == i]
            lane_end = lane_str + lane_vector
            lane = np.vstack([lane_str, lane_end[-1,:].reshape(-1, 2)])
            self.visualize_centerline(lane)

        # visualize the trajectory
        hist = data.positions[:,:50,:]
        fut = data.positions[:,50:,:]
        obj_type = np.zeros((hist.shape[0]))+3
        obj_type[data.cav_mask] = 1
        obj_type[data.commu_mask] = 2

        for i in range(hist.shape[0]):
            self.plot_traj(hist[i], fut[i], obj_type[i], i)

        plt.xlabel("Map X")
        plt.ylabel("Map Y")
        plt.savefig('scene_process.png', dpi=fig.dpi)
        plt.show(block=False)
        plt.pause(5)

    def visualize_centerline(self, centerline) -> None:
        """Visualize the computed centerline.
        Args:
            centerline: Sequence of coordinates forming the centerline
        """
        line_coords = list(zip(*centerline))
        lineX = line_coords[0]
        lineY = line_coords[1]
        plt.plot(lineX, lineY, "--", color="grey", alpha=1, linewidth=1, zorder=0)
        plt.text(lineX[0], lineY[0], "s")
        plt.text(lineX[-1], lineY[-1], "e")
        plt.axis("equal")

    def plot_traj(self, obs, pred, obj_type, traj_id):
        
        traj_na = "t{}".format(traj_id) if traj_id else "traj"
        if obj_type==1:
            obj_type = "CAV" 
        elif obj_type==2:
            obj_type = "CV"
        else:
            obj_type = "NCV" 

        plt.plot(obs[:, 0], obs[:, 1], color=self.COLOR_DICT[obj_type], alpha=1, linewidth=1, zorder=15)
        plt.plot(pred[:, 0], pred[:, 1], '--', color=self.COLOR_DICT[obj_type], alpha=1, linewidth=1, zorder=20)
    
class CarlaCSVLoader():
    def __init__(self, root_dir):
        """ Load csv files from root_dir
        param: 
        root_dir: path to the folder containing sequence csv files
        """
        self.counter = 0
        self.seq_list = [pjoin(root_dir, x) for x in os.listdir(root_dir) if "scene" in x]
        self.current_seq = self.seq_list[self.counter]

    def seq_df(self):
        """Get the dataframe for the current sequence."""

        return self.read_csv(self.current_seq)
    def read_csv(self, path):
        """csv reader
        params:
        path: Path to the csv file

        returns:
        dataframe containing the loaded csv
        """
        return pd.read_csv(path)

    
class CarlaData(Data):

    def __init__(self,
                 x_cav: Optional[torch.Tensor] = None,
                 x_commu: Optional[torch.Tensor] = None,
                 x_sensor: Optional[torch.Tensor] = None,
                 cav_mask: Optional[torch.Tensor] = None,
                 commu_mask: Optional[torch.Tensor] = None,
                 sensor_mask: Optional[torch.Tensor] = None,
                 positions: Optional[torch.Tensor] = None,
                 edge_index: Optional[torch.Tensor] = None,
                 edge_attrs: Optional[List[torch.Tensor]] = None,
                 lane_actor_index: Optional[torch.Tensor] = None,
                 lane_actor_attr: Optional[torch.Tensor] = None,
                 y: Optional[torch.Tensor] = None,
                 y_commu_ori: Optional[torch.Tensor] = None,
                 y_sensor_ori: Optional[torch.Tensor] = None,
                 seq_id: Optional[int] = None,
                #  file_name: Optional[str] = None,
                 num_nodes: Optional[int] = None,
                 padding_mask: Optional[torch.Tensor] = None,
                 bos_mask: Optional[torch.Tensor] = None,
                 rotate_angles: Optional[torch.Tensor] = None,
                 rotate_imat: Optional[torch.Tensor] = None,
                 lane_vectors: Optional[torch.Tensor] = None,
                 lane_pos: Optional[torch.Tensor] = None,
                 lane_idcs: Optional[torch.Tensor] = None,
                 mpr: Optional[torch.Tensor] = None,
                 origin: Optional[torch.Tensor] = None,
                 theta: Optional[torch.Tensor] = None,
                 rotate_mat: Optional[torch.Tensor] = None,
                 **kwargs) -> None:
        if x_cav is None:
            super(CarlaData, self).__init__()
            return
        super(CarlaData, self).__init__(x_cav=x_cav, x_commu=x_commu, x_sensor=x_sensor, mpr=mpr,
                                        cav_mask=cav_mask, commu_mask=commu_mask, sensor_mask=sensor_mask,
                                        positions=positions, edge_index=edge_index, rotate_imat=rotate_imat,
                                        edge_attrs=edge_attrs,lane_actor_index=lane_actor_index, 
                                        lane_actor_attr=lane_actor_attr, y=y, y_commu_ori=y_commu_ori,
                                        y_sensor_ori=y_sensor_ori, seq_id=seq_id, num_nodes=num_nodes, 
                                        padding_mask=padding_mask, bos_mask=bos_mask, rotate_angles=rotate_angles, 
                                        lane_vectors=lane_vectors, lane_pos=lane_pos, lane_idcs=lane_idcs, 
                                        theta=theta, rotate_mat=rotate_mat, **kwargs)
        if edge_attrs is not None:
            for t in range(self.x.size(1)):
                self[f'edge_attr_{t}'] = edge_attrs[t]

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'lane_actor_index':
            return torch.tensor([[self['lane_vectors'].size(0)], [self.num_nodes]])
        else:
            return super().__inc__(key, value)

if __name__ == "__main__":

    frame = 31601
    cav = 454
    test = scene_process(map_radius=75, local_radius=30, mpr=0.6, save_dir="mpr6/")
    rel_path = "../../scene_mining/scene_{}_{}".format(frame,cav)
    df = pd.read_csv(os.path.join(test.abs_path, rel_path))
    
    data = test.process(df, 0)
    test.visualize_data(data)

    from visualization import get_rotate_invariant_trajs
    xrot, yrot, rotate_mat = get_rotate_invariant_trajs(data)




