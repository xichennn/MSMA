import os
import sys
from os.path import join as pjoin

import torch
from torch_geometric.loader import DataLoader
from torch.optim import Adam, AdamW
from tqdm import tqdm
import math

from dataloader.carla_scene_process import CarlaData, scene_processed_dataset
from ModelNet.msma import Base_Net
from torch_geometric.utils import subgraph
from losses.msma_loss import NLLloss
from utils.optim_schedule import ScheduledOptim

#load/process the data
root = "../carla_data/"
source_dir = "scene_mining"
mpr = 0.8
delay_frame = 1
noise_var = 0.1
save_dir = "scene_mining_cav/mpr8_delay{}_noise{}".format(delay_frame, noise_var)

train_set = scene_processed_dataset(root,
                                    "train",
                                    mpr=mpr,
                                    delay_frame=delay_frame,
                                    noise_var=noise_var,
                                    source_dir=source_dir,
                                    save_dir=save_dir)
val_set = scene_processed_dataset(root,
                                  "val",
                                  mpr=mpr,
                                  delay_frame=delay_frame,
                                  noise_var=noise_var,
                                  source_dir=source_dir,
                                  save_dir=save_dir)
test_set = scene_processed_dataset(root,
                                   "test",
                                   mpr=mpr,
                                   delay_frame=delay_frame,
                                   noise_var=noise_var,
                                   source_dir=source_dir,
                                   save_dir=save_dir)
#args
batch_size = 64
num_workers = 4
horizon = 50
lr = 1e-3
betas=(0.9, 0.999)
weight_decay = 0.0001
warmup_epoch=10
lr_update_freq=10
lr_decay_rate=0.9


log_freq = 10
save_folder = ""
model_path = '../carla_data/scene_mining_cav'
ckpt_path = None
verbose = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Base_Net(ip_dim=2,
               historical_steps=30,
               embed_dim=16,
               temp_ff=64,
               spat_hidden_dim=64,
               spat_out_dim=64,
               edge_attr_dim=2,
               map_out_dim=64,
               lane_dim=2,
               map_local_radius=30,
               decoder_hidden_dim=64,
               num_heads=8,
               dropout=0.1,
               num_temporal_layers=4,
               use_variance=False,
               device="cpu",
               commu_only=False,
               sensor_only=False,
               prediction_mode="all")

#dataloader
train_loader = DataLoader(
                train_set,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
                shuffle=True,
                persistent_workers=True
            )
eval_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers, persistent_workers=True, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers, persistent_workers=True, shuffle=False)

#loss
criterion = NLLloss(alpha=0.5, use_variance=False, device=device)
# anchors = k_means_anchors(5, train_loader)

# init optimizer
optim = AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
optm_schedule = ScheduledOptim(
    optim,
    lr,
    n_warmup_epoch=warmup_epoch,
    update_rate=lr_update_freq,
    decay_rate=lr_decay_rate
)

model = model.to(device)
if verbose:
    print("[MSMATrainer]: Train the mode with single device on {}.".format(device))

# model.load_state_dict(torch.load('{}/trained_models_review/model_mpr{}_noise{}_fuse_{}_2.tar'.format(model_path, mpr, noise_var, model.prediction_mode)))

# iteration
training = model.training
avg_loss = 0.0
avg_loss_val = 0.0
losses_train =[]
losses_val = []

epochs = 100
minVal = math.inf

# %%

for epoch in range(epochs):
    avg_loss = 0.0
    ## Train:_______________________________________________________________________________________________________________________________
    training = True
    # model.train()
    data_iter = tqdm(
    enumerate(train_loader),
    desc="{}_Ep_{}: loss: {:.5e}; avg_loss: {:.5e}".format("train" if training else "eval",
                                                            epoch,
                                                            0.0,
                                                            avg_loss),
    total=len(train_loader),
    bar_format="{l_bar}{r_bar}"
)
    count = 0

    for i, data in data_iter: #next(iter(train_loader))
        data = data.to(device)

        if training:
            optm_schedule.zero_grad()
            predictions, mask = model(data)
            gt = torch.matmul(data.y, data.rotate_imat)[mask]
            loss = criterion(predictions['traj'], gt, predictions['log_probs'])
            loss.backward()
            losses_train.append(loss.detach().item())

            torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
            optim.step()
            # write_log("Train Loss", loss.detach().item() / n_graph, i + epoch * len(train_loader))
            avg_loss += loss.detach().item()
            count += 1

            # print log info
            desc_str = "[Info: Device_{}: {}_Ep_{}: loss: {:.5e}; avg_loss: {:.5e}]".format(
                0,
                "train" if training else "eval",
                epoch,
                loss.item(),
                avg_loss / count)
            data_iter.set_description(desc=desc_str, refresh=True)

    if training:
      learning_rate = optm_schedule.step_and_update_lr()
    if epoch%10==0:
      print("learning_rate: ", learning_rate)
                    # write_log("LR", learning_rate, epoch)


    ## Val:_______________________________________________________________________________________________________________________________
    training = False
    # model.eval()
    avg_loss_val = 0.0
    count_val = 0
    data_iter_val = tqdm(enumerate(eval_loader), desc="{}_Ep_{}: loss: {:.5e}; avg_loss: {:.5e}".format("eval",
                        epoch,
                        0.0,
                        avg_loss_val),
                        total=len(eval_loader),
                        bar_format="{l_bar}{r_bar}"
                        )
    for i, data_val in data_iter_val:
        data_val = data_val.to(device)

        with torch.no_grad():
            predictions_val, mask_val = model(data_val)
            gt_val = torch.matmul(data_val.y, data_val.rotate_imat)[mask_val]
            loss_val = criterion(predictions_val['traj'],
                                 gt_val, predictions_val['log_probs'])

        losses_val.append(loss_val.detach().item())
        avg_loss_val += loss_val.detach().item()
        count_val += 1

        # print log info
        desc_str_val = "[Info: Device_{}: {}_Ep_{}: loss: {:.5e}; avg_loss: {:.5e}]".format(
            0,
            "eval",
            epoch,
            loss_val.item(),
            avg_loss_val / count_val)
        data_iter_val.set_description(desc=desc_str_val, refresh=True)

        if loss_val.item() < minVal:
            minVal = loss_val.item()
            torch.save(model.state_dict(), '{}/trained_models_review/model_mpr{}_noise{}_fuse_{}_3.tar'.format(model_path, mpr, noise_var, model.prediction_mode))
            
# %%
## Test:___________________________________________________________________________________________________________________________________
def test(model, test_loader, epoch):
    """
    make predictions on test dataset

    """
    training = model.training
    training = False
    # model.training = False
    count_test = 0
    avg_loss_test = 0.0
    predictions_test = {}
    gts_test = {}
    batch_info = {}
    probs = {}
    masks = {}
    sensor_masks = {}

    data_iter_test = tqdm(enumerate(test_loader), desc="{}_Ep_{}: loss: {:.5e}; avg_loss: {:.5e}".format("test",
                        epoch,
                        0.0,
                        avg_loss_test),
                        total=len(test_loader),
                        bar_format="{l_bar}{r_bar}"
                        )
    for i, data_test in data_iter_test:
        data_test = data_test.to(device)

        with torch.no_grad():
            pred_test, mask_test = model(data_test) #pred_test: offset to anchors
            gt_test = torch.matmul(data_test.y, data_test.rotate_imat)[mask_test] #aligned at +x axis
            #sum of reg and cls loss for all detected vehs
            loss_test = criterion(pred_test['traj'], \
                                  gt_test, pred_test['log_probs'])

        count_test += 1
        avg_loss_test += loss_test.detach().item()
        #compare predictions for vehs in sensor range when centered at [0,0] but not aligned with x-axis
        predictions_test_i = torch.zeros((mask_test.shape[0], 5, 50, 2)).to(device)
        predictions_test_i[mask_test]= pred_test["traj"]
        predictions_test[i] = torch.matmul(predictions_test_i, \
                                           torch.inverse(data_test.rotate_imat.unsqueeze(1)))
        # predictions_test[i] = torch.matmul(pred_test["traj"] + anchors.unsqueeze(0), \
        #                                    torch.inverse(data_test.rotate_imat[mask_test]))
        batch_info[i] = data_test.batch
        probs_i = torch.zeros((mask_test.shape[0], 5)).to(device)
        probs_i[mask_test] = torch.exp(pred_test['log_probs'])
        probs[i] = probs_i
        # probs[i] = torch.exp(pred_test['log_probs'])
        masks[i] = mask_test
        sensor_masks[i] = data_test.sensor_mask
        gts_test[i] = data_test.y

        # print log info
        desc_str_test = "[Info: Device_{}: {}_Ep_{}: loss: {:.5e}; avg_loss: {:.5e}]".format(
            0,
            "test",
            epoch,
            loss_test.item(),
            avg_loss_test / count_test)
        data_iter_test.set_description(desc=desc_str_test, refresh=True)

    return predictions_test, gts_test, probs, batch_info, masks, sensor_masks

predictions_av_av, gt_av_av, probs_av_av, batch_av_av, mask_av_av, sensor_mask_av_av = test(model, test_loader, 100)
