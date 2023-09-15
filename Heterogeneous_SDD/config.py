import torch.nn.functional as F
from numpy import linalg as LA
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
import numpy as np
import copy
import random




labels=["Biker","Pedestrian","Car","Bus","Skater","Cart"]

one_hot_encoding = {}
for i in range(len(labels)):
    encoding = [0.] * len(labels)
    encoding[len(labels) - 1 - i] = 1.
    one_hot_encoding[labels[i]] = encoding

from sklearn import preprocessing
from numpy.linalg import norm

def cosine_similarity(a, b):
    cos_sim = np.inner(a, b) / ((norm(a) * norm(b)) + 1e-6)
    return cos_sim

def angle_func(data_batch, StandardNorm = True):
    angles = [0.0]
    for i in range(1, data_batch.shape[0]):
        angles.append(cosine_similarity(data_batch[i - 1], data_batch[i]))

    if StandardNorm == False:
        scaler = preprocessing.MinMaxScaler().fit(np.array(angles).reshape(-1, 1))
        scaled_angle = scaler.transform(np.array(angles).reshape(-1, 1)).reshape(-1, 1)
        angle_data = np.concatenate((data_batch, scaled_angle), axis=1)
        data = torch.tensor(np.round(angle_data, 4))
        return data
        
    else:
        scaler = preprocessing.StandardScaler().fit(np.array(angles).reshape(-1, 1))
        scaled_angle = scaler.transform(np.array(angles).reshape(-1, 1)).reshape(-1, 1)
        angle_data = np.concatenate((data_batch, scaled_angle), axis=1)
        data = torch.tensor(np.round(angle_data, 4))
        return data


def mag_angle_func(data_batch, StandardNorm = True):
    angles = [0.0]
    magnitude = [0.0]
    for i in range(1, data_batch.shape[0]):
        angles.append(cosine_similarity(data_batch[i - 1], data_batch[i]))
        magnitude.append(LA.norm([data_batch[i] - data_batch[i - 1]]))
    
    if StandardNorm == False:
        scaler = preprocessing.MinMaxScaler().fit(np.array(angles).reshape(-1, 1))
        scaled_angle = scaler.transform(np.array(angles).reshape(-1, 1)).reshape(-1, 1)
        angle_data = np.concatenate((data_batch, scaled_angle), axis=1)
    
    else:
        scaler = preprocessing.StandardScaler().fit(np.array(angles).reshape(-1, 1))
        scaled_angle = scaler.transform(np.array(angles).reshape(-1, 1)).reshape(-1, 1)
        angle_data = np.concatenate((data_batch, scaled_angle), axis=1)
        
    
    mag = np.array(magnitude).reshape(-1, 1)
    final_data = np.concatenate((angle_data, mag), axis=1)

    return torch.tensor(np.round(final_data, 4))

def data_processor(dataloader):

    _sequence = []

    for batch_idx, bath_data in enumerate(dataloader):
        data = [tensor.cuda() for tensor in bath_data]
        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, V_obs, V_pred) = data
        full_rel = torch.cat((obs_traj_rel.squeeze(0), pred_traj_gt_rel.squeeze(0)), dim=1)
        tmp_rel = torch.zeros((full_rel.shape[0], full_rel.shape[1], 3))
        mag_angle_rel = torch.zeros((full_rel.shape[0], full_rel.shape[1], 4))

        for i in range(tmp_rel.shape[0]):
            tmp_rel[i] = angle_func(full_rel[i].cpu().numpy(), StandardNorm= True)
            mag_angle_rel[i] = mag_angle_func(full_rel[i].cpu().numpy(), StandardNorm = True)

        sequential_data = copy.deepcopy(mag_angle_rel[:, 2:8, :][:,:,2:]).to(device)

        _sequence.append(sequential_data)

    temp = torch.cat(_sequence, 0)
    index = torch.LongTensor(random.sample(range(temp.shape[0]), 50000)).cuda()
    temp_ = torch.index_select(temp, 0, index)

    
    return temp_.permute(1,0,2)

def train_data_processor(dataloader):
    rel_sequence = []
    abs_sequence = []

    for batch_idx, bath_data in enumerate(dataloader):
        data = [tensor.cuda() for tensor in bath_data]
        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, V_obs, V_pred) = data
        full_rel = torch.cat((obs_traj_rel.squeeze(0), pred_traj_gt_rel.squeeze(0)), dim=2)
        full_abs = torch.cat((obs_traj.squeeze(0), pred_traj_gt.squeeze(0)), dim=2)

        rel_sequence.append(full_rel)
        abs_sequence.append(full_abs)

    all_rel = torch.cat(rel_sequence, 0)
    all_abs = torch.cat(abs_sequence, 0)

    return all_rel.permute(0,2,1), all_abs.permute(0,2,1)