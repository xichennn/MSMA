"""mine CAV scenarios from logged carla data"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import torch
import random
# import math
import os
import copy

def get_obj_type_at_mpr(data_df, vids, cav_id, mpr=0.2):
    others = list(set(vids) - set([cav_id]))
    #keep random seed here to ensure the same seed
    random.seed(30)
    cv_ids = random.sample(list(others), int(mpr*len(others)))
    vid_df = data_df["vid"].values
    obj_type_mpr = []
    for v in vid_df:
        if v == cav_id:
            obj_type_mpr.append("cav")
        elif v in cv_ids:
            obj_type_mpr.append("cv")
        else:
            obj_type_mpr.append("ncv")
    return obj_type_mpr

# read the data
data_raw = pd.read_csv("../carla_data/Location.csv", header=None)
header = ["frame","time","vid","type_id","position_x","position_y","position_z","rotation_x","rotation_y","rotation_z","vel_x","vel_y","angular_z"]
map = {idx:header[idx] for idx in range(13)}
data_raw = data_raw.rename(columns = map) 
# make pos_y consistent with map
data_raw["position_y"] = -data_raw["position_y"]
# %%
vids = list(data_raw["vid"].unique())
ts = np.sort(np.unique(data_raw['frame'].values))
random.seed(30)
cv_range = 50
av_range = 30

data_df = data_raw.copy(deep=True)
# segment the scenes into 10s
min_ts = ts[0]
max_ts = ts[-1]
# 5s overlapping among scenes 10s = 100 steps/frames
# remove the scenes where cav is not moving
for cav_id in vids:
    for frame in range(min_ts+50,max_ts-50,50):

        vehicles_at_frame = data_df[data_df["frame"] == frame]
        cav_entry = data_df[(data_df["frame"]==frame) & (data_df["vid"]==cav_id)]
        cav_entry_previous = data_df[(data_df["frame"]==frame-1) & (data_df["vid"]==cav_id)]
        if (cav_entry.position_x.values == cav_entry_previous.position_x.values) and \
            (cav_entry.position_y.values == cav_entry_previous.position_y.values):
            continue

        dist = ((vehicles_at_frame["position_x"].values - cav_entry["position_x"].values)**2
                + (vehicles_at_frame["position_y"].values - cav_entry["position_y"].values)**2)**0.5
        cv_idx = np.where((dist<cv_range)&(dist>0))[0]
        cv_neighbors = vehicles_at_frame["vid"].values[cv_idx]
        #remove unmoved surrounding vehicles 
        vid_ngbr_unmove = []
        for i in range(len(cv_neighbors)):
            vid_ngbr = cv_neighbors[i]
            ngbr_entry = data_df[(data_df["frame"]==frame) & (data_df["vid"]==vid_ngbr)]
            ngbr_entry_previous = data_df[(data_df["frame"]==frame-1) & (data_df["vid"]==vid_ngbr)]
            if (ngbr_entry.position_x.values == ngbr_entry_previous.position_x.values) and \
                (ngbr_entry.position_y.values == ngbr_entry_previous.position_y.values):
                vid_ngbr_unmove.append(vid_ngbr)
        cv_ngbrs_move=list(set(cv_neighbors)-set(vid_ngbr_unmove))

        av_idx = np.where((dist<av_range)&(dist>0))[0]
        av_neighbors = vehicles_at_frame["vid"].values[av_idx]
        av_ngbrs_move = list(set(av_neighbors)-set(vid_ngbr_unmove))

        scene_frames = list(range(frame-50,frame+50))
        scene_vids = [cav_id]+cv_ngbrs_move
        scene_data = copy.deepcopy(data_df[data_df["vid"].isin(scene_vids) & data_df["frame"].isin(scene_frames)])
        
        #mprs
        obj_type_mpr_02 = get_obj_type_at_mpr(scene_data, scene_vids, cav_id, mpr=0.2)
        obj_type_mpr_04 = get_obj_type_at_mpr(scene_data, scene_vids, cav_id, mpr=0.4)
        obj_type_mpr_06 = get_obj_type_at_mpr(scene_data, scene_vids, cav_id, mpr=0.6)
        obj_type_mpr_08 = get_obj_type_at_mpr(scene_data, scene_vids, cav_id, mpr=0.8)
        scene_data["obj_type_mpr_02"] = obj_type_mpr_02
        scene_data["obj_type_mpr_04"] = obj_type_mpr_04
        scene_data["obj_type_mpr_06"] = obj_type_mpr_06
        scene_data["obj_type_mpr_08"] = obj_type_mpr_08

        scene_data["in_av_range"] = scene_data["vid"].isin([cav_id]+av_ngbrs_move).values
        scene_data.to_csv("scene_mining/scene_{}_{}".format(frame, cav_id),index=False)
