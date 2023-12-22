#test overall model architecture
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys

from ModelNet.utils import MLP, bivariate_gaussian_activation
# from utils import MLP
from typing import Optional, Tuple, Union, Dict
import math
from torch_scatter import scatter_mean, scatter_add

from dataloader.carla_scene_process import CarlaData
from itertools import product
from torch_geometric.utils import subgraph, add_self_loops

class Base_Net(nn.Module):
    def __init__(self, 
                 ip_dim: int=2,
                 historical_steps: int=30,
                 embed_dim: int=16,
                 temp_ff: int=64,
                 spat_hidden_dim: int=64,
                 spat_out_dim: int=64,
                 edge_attr_dim: int=2,
                 map_out_dim: int=64,
                 lane_dim: int = 2,
                 map_local_radius: float=30.,
                 decoder_hidden_dim: int=64,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 num_temporal_layers: int = 4,
                 use_variance: bool = False,
                 device = 'cpu',
                 commu_only = False,
                 sensor_only = False,
                 prediction_mode = None,
                 ) -> None:
        super(Base_Net, self).__init__()
        self.ip_dim = ip_dim
        self.historical_steps = historical_steps
        self.embed_dim = embed_dim
        self.device = device
        self.local_radius = map_local_radius
        self.commu_only = commu_only
        self.sensor_only = sensor_only
        self.prediction_mode = prediction_mode
    
        if self.prediction_mode == "temp_only":
            decoder_in_dim = embed_dim
        elif self.prediction_mode == "temp_spat":
            decoder_in_dim = spat_out_dim
        else:
            decoder_in_dim = spat_out_dim+map_out_dim
        
        #input embedding
        self.ip_emb_cav = MLP(ip_dim, embed_dim)
        self.ip_emb_commu = MLP(ip_dim, embed_dim)
        self.ip_emb_sensor = MLP(ip_dim, embed_dim)
        self.ip_emb_fuse = MLP(ip_dim, embed_dim)
        #temporal encoders
        self.temp_encoder = TemporalEncoder(historical_steps=historical_steps,
                                            embed_dim=embed_dim,
                                            device=device,
                                            num_heads=num_heads,
                                            num_layers=num_temporal_layers,
                                            temp_ff=temp_ff,
                                            dropout=dropout)
        self.feature_fuse = FeatureFuse(embed_dim=embed_dim,
                                        num_heads=num_heads,
                                        dropout=dropout)
        self.spat_encoder = GAT(in_dim=embed_dim, 
                                hidden_dim=spat_hidden_dim, 
                                out_dim=spat_out_dim, 
                                edge_attr_dim=edge_attr_dim, 
                                device=device,
                                num_heads=num_heads, 
                                dropout=dropout)
        self.map_encoder = MapEncoder(lane_dim=lane_dim,
                                      v_dim=spat_out_dim,
                                      out_dim=map_out_dim,
                                      edge_attr_dim=edge_attr_dim,
                                      num_heads=num_heads,
                                      device=device,
                                      dropout=dropout)
        self.decoder = PredictionDecoder(encoding_size=decoder_in_dim,
                                         hidden_size=decoder_hidden_dim,
                                         num_modes=5,
                                         op_len=50,
                                         use_variance=use_variance)
        
    def forward(self, data: CarlaData):

        #temporal encoding
        x_cav, x_commu, x_sensor = data.x_cav, data.x_commu, data.x_sensor #overlapping among different modes
        cav_mask, commu_mask, sensor_mask = data.cav_mask, data.commu_mask, data.sensor_mask
        rotate_imat = data.rotate_imat
        x_cav = torch.bmm(x_cav, rotate_imat[cav_mask]) 
        x_commu = torch.bmm(x_commu, rotate_imat[commu_mask]) 
        x_sensor = torch.bmm(x_sensor, rotate_imat[sensor_mask]) 

        x_cav_, x_commu_, x_sensor_ = self.ip_emb_cav(x_cav), self.ip_emb_commu(x_commu), self.ip_emb_sensor(x_sensor)
        cav_out, commu_out, sensor_out = self.temp_encoder(x_cav_, x_commu_, x_sensor_)
        
        #convert back to original num_nodes given masks
        node_features_all = torch.zeros((data.num_nodes, self.embed_dim)).to(self.device)
        node_features_all[cav_mask] = cav_out
        node_features_all[commu_mask] = commu_out
        node_features_all[sensor_mask] = sensor_out
        #fuse sensor&commu encodings
        mask_fuse = (commu_mask & sensor_mask)
        commu_emd, sensor_emd = self.get_overlap_feature(data, commu_out, sensor_out, mask_fuse, self.embed_dim)
        # commu_relpos, sensor_relpos = self.get_overlap_feature(data, data.x_commu_ori, data.x_sensor_ori, mask_fuse, self.ip_dim)
        # relpos_emd = self.ip_emb_fuse(sensor_relpos-commu_relpos)

        if self.commu_only:
            node_features_all[commu_mask] = commu_out
            # data.y[commu_mask] = data.y_commu
        elif self.sensor_only:
            node_features_all[sensor_mask] = sensor_out
        elif sum(mask_fuse)>0:
            node_features_all[mask_fuse] = self.feature_fuse(commu_emd, sensor_emd)
        
        mask_all = (cav_mask | commu_mask | sensor_mask)

        if self.prediction_mode == "temp_only":
            predictions = self.decoder(node_features_all[mask_all]) #'traj':[nodes_of_interest, 5, 50, 2], 'log_probs':[nodes_of_interest, 5]
            return predictions, mask_all

        edge_index, _ = subgraph(subset=mask_all, edge_index=data.edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=data.num_nodes)
        edge_attr = data['positions'][edge_index[0], 49] - data['positions'][edge_index[1], 49]
        # edge_attr = torch.bmm(edge_attr.unsqueeze(-2), rotate_imat[edge_index[1]]).squeeze(-2)
        spat_out = self.spat_encoder(node_features_all.view(data.num_nodes,-1), edge_index, edge_attr) #[num_nodes, 64]
        
        if self.prediction_mode == "temp_spat":
            predictions = self.decoder(spat_out[mask_all]) #'traj':[nodes_of_interest, 5, 50, 2], 'log_probs':[nodes_of_interest, 5]
            return predictions, mask_all
        #AL encoding
        map_out = self.map_encoder(data, spat_out, mask_all) #[num_nodes, 64]
        final_emd = torch.cat((spat_out, map_out), dim=-1) #[num_nodes, 128]
        
        predictions = self.decoder(final_emd[mask_all]) #'traj':[nodes_of_interest, 5, 50, 2], 'log_probs':[nodes_of_interest, 5]
        return predictions, mask_all
    
    def get_overlap_feature(self, data, commu_f, sensor_f, mask_fuse, dim):
        commu_mask, sensor_mask = data.commu_mask, data.sensor_mask
        commu_feature = torch.zeros((data.num_nodes, dim)).to(self.device)
        sensor_feature = torch.zeros((data.num_nodes, dim)).to(self.device)
        commu_feature[commu_mask] = commu_f
        sensor_feature[sensor_mask] = sensor_f
       
        return commu_feature[mask_fuse], sensor_feature[mask_fuse]


class TemporalEncoder(nn.Module):
    '''
    for each agent, only one fused channel instead of three
    '''
    def __init__(self,
                 historical_steps: int,
                 embed_dim: int,
                 device,
                 num_heads: int=8,
                 num_layers: int=4,
                 temp_ff: int=64,
                 dropout: float=0.1) -> None:
        super(TemporalEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.device = device
        self.historical_steps = historical_steps
        encoder_layer_cav = nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=temp_ff, dropout=dropout, batch_first=True) 
        self.transformer_encoder_cav = nn.TransformerEncoder(encoder_layer=encoder_layer_cav, num_layers=num_layers,
                                                         norm=nn.LayerNorm(embed_dim))
        encoder_layer_sensor = nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=temp_ff, dropout=dropout, batch_first=True)
        self.transformer_encoder_sensor = nn.TransformerEncoder(encoder_layer=encoder_layer_sensor, num_layers=num_layers,
                                                         norm=nn.LayerNorm(embed_dim))
        encoder_layer_commu = nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=temp_ff, dropout=dropout, batch_first=True)
        self.transformer_encoder_commu= nn.TransformerEncoder(encoder_layer=encoder_layer_commu, num_layers=num_layers,
                                                         norm=nn.LayerNorm(embed_dim))
        self.cls_token_cav = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        self.cls_token_commu = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        self.cls_token_sensor = nn.Parameter(torch.Tensor(1, 1, embed_dim))

        self.pos_embed_cav = nn.Parameter(torch.Tensor(1, historical_steps + 1, embed_dim))
        self.pos_embed_commu = nn.Parameter(torch.Tensor(1, historical_steps + 1, embed_dim))
        self.pos_embed_sensor = nn.Parameter(torch.Tensor(1, historical_steps + 1, embed_dim))

        nn.init.normal_(self.cls_token_cav, mean=0., std=.02)
        nn.init.normal_(self.cls_token_commu, mean=0., std=.02)
        nn.init.normal_(self.cls_token_sensor, mean=0., std=.02)
        nn.init.normal_(self.pos_embed_cav, mean=0., std=.02)
        nn.init.normal_(self.pos_embed_commu, mean=0., std=.02)
        nn.init.normal_(self.pos_embed_sensor, mean=0., std=.02)
        # self.apply(init_weights)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, x_cav, x_commu, x_sensor):
        """
        input [batch, seq, feature]
        """
        num_sensor, seq_len = x_sensor.shape[0], x_sensor.shape[1]
        assert seq_len == self.historical_steps

        x_cav, x_commu, x_sensor = self._expand_cls_token(x_cav, x_commu, x_sensor)

        x_cav = x_cav + self.pos_embed_cav
        x_sensor = x_sensor + self.pos_embed_sensor
        x_commu = x_commu + self.pos_embed_commu

        # Apply dropout and layer normalization
        x_cav_t = self.layer_norm(self.dropout(x_cav))
        x_sensor_t = self.layer_norm(self.dropout(x_sensor))
        x_commu_t = self.layer_norm(self.dropout(x_commu))

        # Apply the transformers
        x_cav_temp = self.transformer_encoder_cav(x_cav_t)
        x_commu_temp = self.transformer_encoder_commu(x_commu_t)
        x_sensor_temp = self.transformer_encoder_sensor(x_sensor_t)

        return x_cav_temp[:,-1,:], x_commu_temp[:,-1,:], x_sensor_temp[:,-1,:] #encoding at last timestep

    def _expand_cls_token(self, x_cav, x_commu, x_sensor):
        expand_cls_token_cav= self.cls_token_cav.expand(x_cav.shape[0], -1, -1)
        expand_cls_token_commu= self.cls_token_commu.expand(x_commu.shape[0], -1, -1)
        expand_cls_token_sensor= self.cls_token_sensor.expand(x_sensor.shape[0], -1, -1)

        x_cav = torch.cat((x_cav, expand_cls_token_cav), dim=1)
        x_commu = torch.cat((x_commu, expand_cls_token_commu), dim=1)
        x_sensor = torch.cat((x_sensor, expand_cls_token_sensor), dim=1)

        return x_cav, x_commu, x_sensor

class FeatureFuse(nn.Module):
    """
    cross attention module
    """
    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout=0.1):
        super(FeatureFuse, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.lin_q = nn.Linear(embed_dim, embed_dim)
        self.lin_k = nn.Linear(embed_dim, embed_dim)
        self.lin_v = nn.Linear(embed_dim, embed_dim)
        self.lin_self = nn.Linear(embed_dim, embed_dim)
        self.lin_ih = nn.Linear(embed_dim, embed_dim)
        self.lin_hh = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, commu_enc, sensor_enc):
        query = self.lin_q(sensor_enc).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        key = self.lin_k(commu_enc).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        value = self.lin_v(commu_enc).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        scale = (self.embed_dim // self.num_heads) ** 0.5
        alpha = (query * key).sum(dim=-1) / scale
        alpha = self.softmax(alpha)
        alpha = self.attn_drop(alpha)
        commu_att = (value * alpha.unsqueeze(-1)).reshape(-1, self.embed_dim)
        w = torch.sigmoid(self.lin_ih(sensor_enc) + self.lin_hh(commu_att))
        fused_enc = w * self.lin_self(sensor_enc) + (1-w) * commu_att
        return fused_enc

class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, edge_attr_dim, device, num_heads=8, dropout=0.1):
        super(GAT, self).__init__()

        self.device = device
        self.attention_layers = nn.ModuleList(
            [GATlayer(in_dim, hidden_dim, edge_attr_dim) for _ in range(num_heads)]
        )
        self.out_att = GATlayer(hidden_dim*num_heads, out_dim, edge_attr_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, edge_index, edge_attr):
        x = X
        
        # Concatenate multi-head attentions
        x = torch.cat([att(x, edge_index, edge_attr) for att in self.attention_layers], dim=1) 
        x = F.elu(x)
        x = self.dropout(x)
        x = self.out_att(x, edge_index, edge_attr) # Final attention aggregation
        return F.log_softmax(x, dim=1)
    
class GATlayer(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 out_dim: int,
                 edge_attr_dim: int,
                 dropout: float=0.1) -> None:
        super(GATlayer, self).__init__()
        
        self.W = nn.Linear(embed_dim, out_dim, bias=False)
        self.a = nn.Linear(2*out_dim + edge_attr_dim, 1, bias=False)
        self.edge_attr_dim = edge_attr_dim
        self.dropout = nn.Dropout(dropout)
        self.out_transform = nn.Linear(out_dim, out_dim, bias=False)

    def forward(self, 
                X: torch.Tensor, 
                edge_index: torch.Tensor, 
                edge_attr: torch.Tensor):
        #transform node features
        h = self.W(X)
        N = h.size(0)
        attn_input = self._prepare_attention_input(h, edge_index, edge_attr)
        score_per_edge = F.leaky_relu(self.a(attn_input)).squeeze(1)  # Calculate attention coefficients
        
        #apply dropout to attention weights
        score_per_edge = self.dropout(score_per_edge)
        # softmax
        # Calculate the numerator. Make logits <= 0 so that e^logit <= 1 (this will improve the numerical stability)
        score_per_edge = score_per_edge - score_per_edge.max()
        exp_score_per_edge = score_per_edge.exp()  

        neigborhood_aware_denominator = scatter_add(exp_score_per_edge, edge_index[0], dim=0, dim_size=N)
        neigborhood_aware_denominator = neigborhood_aware_denominator.index_select(0, edge_index[0])
        attentions_per_edge = exp_score_per_edge / (neigborhood_aware_denominator + 1e-16)
        
        # Apply attention weights to source node features and perform message passing
        out_src = h.index_select(0,edge_index[1]) * attentions_per_edge.unsqueeze(dim=1)
        h_prime = scatter_add(out_src, edge_index[0], dim=0, dim_size=N)

        # Apply activation function
        out = F.elu(h_prime)
        return out

    def _prepare_attention_input(self, h, edge_index, edge_attr):
        '''
        h has shape [N, out_dim]
        '''
        src, tgt = edge_index
        attn_input = torch.cat([h.index_select(0,src), h.index_select(0,tgt),  edge_attr], dim=1)

        return attn_input
    
class MapEncoder(nn.Module):
    def __init__(self,
                 lane_dim: int,
                 v_dim: int,
                 out_dim: int,
                 edge_attr_dim: int,
                 num_heads: int,
                 device: str,
                 local_radius: float=30.,
                 dropout: float=0.1) -> None:
        super(MapEncoder, self).__init__()
        self.local_radius = local_radius
        self.device = device
        self.attention_layers = nn.ModuleList(
            [MapEncoderLayer(out_dim, v_dim, edge_attr_dim) for _ in range(num_heads)]
        )
        self.lane_emb = MLP(lane_dim, v_dim) #out_dim = v_enc.size(1)
        self.edge_attr_dim = edge_attr_dim
        self.dropout = nn.Dropout(dropout)
        self.out_transform = nn.Linear(out_dim*num_heads, out_dim, bias=False)

    def forward(self, data: CarlaData, v_enc: torch.Tensor, v_mask: torch.Tensor):

        lane = data.lane_vectors

        lane_actor_mask = torch.cat((v_mask, (torch.ones(lane.size(0))==1).to(self.device)), dim=0)
        data.lane_actor_index[0] += data.num_nodes #lane_actor_index[0]:lane index, lane_actor_index[1]:actor index
        lane_actor_index, lane_actor_attr = subgraph(subset=lane_actor_mask, 
                                       edge_index=data.lane_actor_index, edge_attr=data.lane_actor_attr)
        lane = torch.bmm(lane[lane_actor_index[0]-data.num_nodes].unsqueeze(-2), data.rotate_imat[lane_actor_index[1]]).squeeze(-2)

        lane_enc = self.lane_emb(lane)
        lane_actor_enc = torch.cat((v_enc, lane_enc), dim=0) #shape:[num_veh+num_lane, v_dim]
        # Concat multi-head attentions
        out = torch.cat([att(lane_actor_enc, data.num_nodes, lane.size(0), lane_actor_index, lane_actor_attr) for att in self.attention_layers], dim=1) 
        out = F.elu(out)
        out = self.dropout(out)
        out = self.out_transform(out)

        return out
    
class MapEncoderLayer(nn.Module):
    def __init__(self,
                 v_dim: int,
                 out_dim: int,
                 edge_attr_dim: int,
                 dropout: float=0.1) -> None:
        super(MapEncoderLayer, self).__init__()
        
        self.W = nn.Linear(v_dim, out_dim, bias=False)
        self.a = nn.Linear(2*out_dim + edge_attr_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, 
                lane_actor_enc: torch.Tensor, 
                num_veh: int, 
                num_lane: int,
                lane_actor_index: torch.Tensor,
                lane_actor_attr: torch.Tensor):
        #transform node features
        h = self.W(lane_actor_enc)
        N = h.size(0)
        assert N == num_veh+num_lane

        attn_input = self._prepare_attention_input(h, num_veh,lane_actor_index, lane_actor_attr)
        score_per_edge = F.leaky_relu(self.a(attn_input)).squeeze(1)  # Calculate attention coefficients
        
        #apply dropout to attention weights
        score_per_edge = self.dropout(score_per_edge)
        # softmax
        # Calculate the numerator. Make logits <= 0 so that e^logit <= 1 (this will improve the numerical stability)
        score_per_edge = score_per_edge - score_per_edge.max()
        exp_score_per_edge = score_per_edge.exp()  

        neigborhood_aware_denominator = scatter_add(exp_score_per_edge, lane_actor_index[1], dim=0, dim_size=num_veh)
        neigborhood_aware_denominator = neigborhood_aware_denominator.index_select(0, lane_actor_index[1])
        attentions_per_edge = exp_score_per_edge / (neigborhood_aware_denominator + 1e-16)

        out_src = h[num_veh:] * attentions_per_edge.unsqueeze(dim=1) #shape[num_lane]
        out = scatter_add(out_src, lane_actor_index[1], dim=0, dim_size=num_veh)
        assert out.shape[0] == num_veh

        # Apply activation function
        out = F.elu(out)
        return out
    
    def _prepare_attention_input(self, h, num_v, edge_index, edge_attr):
        '''
        h has shape [N, out_dim]
        '''
        src, tgt = edge_index
        attn_input = torch.cat([h[num_v:], h[:num_v].index_select(0,tgt),  edge_attr], dim=1)

        return attn_input
    
class PredictionDecoder(nn.Module):

    def __init__(self,
                 encoding_size: int,
                 hidden_size: int=64,
                 num_modes: int=5,
                 op_len: int=50,
                 use_variance: bool=False) -> None:
        super(PredictionDecoder, self).__init__()

        self.op_dim = 5 if use_variance else 2
        self.op_len = op_len
        self.num_modes = num_modes
        self.use_variance = use_variance
        self.hidden = nn.Linear(encoding_size, hidden_size)
        self.traj_op = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, self.op_len * self.op_dim * self.num_modes))
        self.prob_op = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, self.num_modes))

        self.leaky_relu = nn.LeakyReLU(0.01)
        self.log_softmax = nn.LogSoftmax(dim=1)


    def forward(self, agg_encoding: torch.Tensor) -> Dict:
        """
        Forward pass for prediction decoder
        :param agg_encoding: aggregated context encoding
        :return predictions: dictionary with 'traj': K predicted trajectories and
            'probs': K corresponding probabilities
        """

        h = self.leaky_relu(self.hidden(agg_encoding))
        num_vehs = h.shape[0] #n_v
        traj = self.traj_op(h) #[n_v, 1250]
        probs = self.log_softmax(self.prob_op(h)) #[n_v, 5]
        traj = traj.reshape(num_vehs, self.num_modes, self.op_len, self.op_dim)
        probs = probs.squeeze(dim=-1)
        traj = bivariate_gaussian_activation(traj) if self.use_variance else traj
        
        predictions = {'traj':traj, 'log_probs':probs}

        return predictions
