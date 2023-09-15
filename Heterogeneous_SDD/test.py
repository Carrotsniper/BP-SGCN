import pickle
import glob
import torch.nn.functional as F
import torch
from torch.utils.data.dataloader import DataLoader
import torch.distributions.multivariate_normal as torchdist
import random
from utils import *
from metrics import * 
from model import TrajectoryModel_attn
import copy

from sklearn import preprocessing
from numpy.linalg import norm
from config import angle_func, mag_angle_func, cosine_similarity
from baseline_model import VRNN
from DEC_model import ClusteringLayer, DEC
import collections
from metrics import * 
from model import TrajectoryModel_attn
import copy

from sklearn import preprocessing
from numpy.linalg import norm
from config import angle_func, mag_angle_func, cosine_similarity, train_data_processor
from baseline_model import VRNN
from DEC_model import ClusteringLayer, DEC
import torch.nn.functional as F
from numpy import linalg as LA
from expert_find import expert_find

import torch.nn.functional as F
from numpy import linalg as LA
device = "cuda" if torch.cuda.is_available() else "cpu"


device = "cuda" if torch.cuda.is_available() else "cpu"

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

@torch.no_grad()
def test(model, loader_test,  expert_rel, expert_abs, KSTEPS=20):

    model.eval()
    ade_all, fde_all, col_all, tcc_all = [], [], [], []

    for batch in loader_test: 
        #Get data
        batch = [tensor.cuda() for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
        loss_mask, V_obs, V_tr = batch

        full_rel = torch.cat((obs_traj_rel.squeeze(0).permute(0,2,1), pred_traj_gt_rel.squeeze(0).permute(0,2,1)), dim=1)
        full_abs = torch.cat((obs_traj.squeeze(0).permute(0,2,1), pred_traj_gt.squeeze(0).permute(0,2,1)), dim=1)
        
        angles = [0]
        end_error, rst = expert_find(
            full_rel.cpu().numpy(),
            full_abs.cpu().numpy(),
            expert_rel.cpu().numpy(),
            expert_abs.cpu().numpy(),
            angles,
            option=2,)
        rst = torch.stack(rst)  # [num_of_objs, 2]
        dest = rst.unsqueeze(1).reshape(obs_traj.shape[1], 1, 2).repeat(1, 8, 1)
        V_obs_new = V_obs.squeeze(0).permute(1,0,2)[:,:,1:] - dest
        num = V_obs[:,:, :, 0].unsqueeze(2).permute(0,1,3,2)
        V_obs = torch.cat([num, V_obs_new.unsqueeze(0).permute(0,2,1,3)], dim=3)



        tmp_rel = torch.zeros((full_rel.shape[0], full_rel.shape[1], 3))
        mag_angle_rel = torch.zeros((full_rel.shape[0], full_rel.shape[1], 4))

        for i in range(tmp_rel.shape[0]):
            tmp_rel[i] = angle_func(full_rel[i].cpu().numpy(), StandardNorm= True)
            mag_angle_rel[i] = mag_angle_func(full_rel[i].cpu().numpy(), StandardNorm = True)
        sequential_data = copy.deepcopy(mag_angle_rel[:, 2:8, :][:,:,2:]).to(device)

        dec = torch.load("./classify_model/classify_model.pt")
        dec.eval()
        
        output = dec(sequential_data.permute(1,0,2)).detach().cpu()
        # label = output.argmax(1)
        out_label = F.gumbel_softmax(output, tau=1, hard=True).to(device)
        # batch_class_onehot = F.one_hot(label, num_classes=6).to(device)
        V_obs = torch.cat((V_obs,torch.broadcast_to(out_label, (1,8,obs_traj_rel.shape[1],6))), dim=3)
        V_tr = torch.cat((V_tr,torch.broadcast_to(out_label, (1,12,obs_traj_rel.shape[1],6))), dim=3)

        identity_spatial = torch.ones((V_obs.shape[1], V_obs.shape[2], V_obs.shape[2])) * torch.eye(
            V_obs.shape[2])
        identity_temporal = torch.ones((V_obs.shape[2], V_obs.shape[1], V_obs.shape[1])) * torch.eye(
            V_obs.shape[1])
        identity_spatial = identity_spatial.cuda()
        identity_temporal = identity_temporal.cuda()
        identity = [identity_spatial, identity_temporal]
        
        V_pred = model(V_obs.to(torch.float32), identity)
        V_obs_traj = obs_traj.permute(0, 3, 1, 2).squeeze(dim=0)
        V_pred_traj_gt = pred_traj_gt.permute(0, 3, 1, 2).squeeze(dim=0)
        mu, cov = generate_statistics_matrices(V_pred.squeeze(dim=0))
        ade_stack, fde_stack, tcc_stack, col_stack = [], [], [], []
        for trial in range(100):
            sample_level = 'scene'
            if sample_level == 'scene':
                r_sample = random.randn(1, KSTEPS, 2)
            else:
                raise NotImplementedError

            r_sample = torch.Tensor(r_sample).to(dtype=mu.dtype, device=mu.device)
            r_sample = r_sample.permute(1, 0, 2).unsqueeze(dim=1).expand((KSTEPS,) + mu.shape)
            V_pred_sample = mu + (torch.linalg.cholesky(cov) @ r_sample.unsqueeze(dim=-1)).squeeze(dim=-1)

            V_absl = V_pred_sample.cumsum(dim=1) + V_obs_traj[[-1], :, :]
            ADEs, FDEs, COLs, TCCs = compute_batch_metric(V_absl*10, V_pred_traj_gt*10)

            ade_stack.append(ADEs.detach().cpu().numpy())
            fde_stack.append(FDEs.detach().cpu().numpy())
            col_stack.append(COLs.detach().cpu().numpy())
            tcc_stack.append(TCCs.detach().cpu().numpy())
        ade_all.append(np.array(ade_stack))
        fde_all.append(np.array(fde_stack))
        col_all.append(np.array(col_stack))
        tcc_all.append(np.array(tcc_stack))

    ade_all = np.concatenate(ade_all, axis=1)
    fde_all = np.concatenate(fde_all, axis=1)
    col_all = np.concatenate(col_all, axis=1)
    tcc_all = np.concatenate(tcc_all, axis=1)

    mean_ade, mean_fde = ade_all.mean(axis=0).mean(), fde_all.mean(axis=0).mean()
    mean_col, mean_tcc = col_all.mean(axis=0).mean(), tcc_all.mean(axis=0).mean()
    return mean_ade, mean_fde, mean_col, mean_tcc


def main():
    ADE_ls, FDE_ls, COL_ls, TCC_ls = [], [], [], []
    KSTEPS = 20
    print('Number of samples:', KSTEPS)
    print("*" * 50)
    root_ = './dec_checkpoints/stanFordDrone'
    

    paths = root_
    print(paths)
    print("*" * 50)
    print("Evaluating model:", paths)

    model_path =  paths + '/val_best.pth'
    args_path = paths + '/args.pkl'
    with open(args_path, 'rb') as f:
        args = pickle.load(f)

    # Data prep
    obs_seq_len = args.obs_len
    pred_seq_len = args.pred_len
    data_set = './dataset/stanford/'

    dset_train = AgentTrajectoryDataset(
        data_set + 'train/',
        obs_len=obs_seq_len,
        pred_len=pred_seq_len,
        skip=1)

    loader_train = DataLoader(
        dset_train,
        batch_size=1,  # This is irrelative to the args batch size parameter
        shuffle=True,
        num_workers=0)

    expert_rel, expert_abs = train_data_processor(loader_train)


    dset_test = AgentTrajectoryDataset(
        data_set + 'test/',
        obs_len=obs_seq_len,
        pred_len=pred_seq_len,
        skip=1)

    loader_test = DataLoader(
        dset_test,
        batch_size=1,  # This is irrelative to the args batch size parameter
        shuffle=False,
        num_workers=1)

    model = TrajectoryModel_attn(number_asymmetric_conv_layer=7, embedding_dims=64, number_gcn_layers=1, dropout=0,
                        obs_len=8, pred_len=12, n_tcn=5, out_dims=5).cuda()
    model.load_state_dict(torch.load(model_path))


    print("Testing ....")
    ADE, FDE, COL, TCC = test(model, loader_test, expert_rel, expert_abs)
    ADE_ls.append(ADE)
    FDE_ls.append(FDE)
    COL_ls.append(COL)
    TCC_ls.append(TCC)

    print("Scene: {} ADE: {:.8f} FDE: {:.8f} COL: {:.8f}, TCC: {:.8f}".format('SDD', ADE, FDE, COL, TCC))
    


if __name__ == '__main__':
    main()
