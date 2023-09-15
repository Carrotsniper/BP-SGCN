from utils_SDD import AgentTrajectoryDataset
from torch.utils.data.dataloader import DataLoader
import numpy as np
import copy
import sys
from sklearn.mixture import GaussianMixture
import argparse
import pathlib
import random
import logging
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import pickle
import torch.nn.functional as F
from numpy import linalg as LA
from config import angle_func, mag_angle_func, cosine_similarity
import math
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import matplotlib.pyplot as plt 
from baseline_model import VRNN
from varc import VARC

import neptune.new as neptune
run = neptune.init(
    project="sc16rl/pretrain",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyOTY2ZmIxYi1kODViLTRmN2EtYmFiOS04NTgxYjU5MzgwZjkifQ==",
)  # your credentials

### Loading Training Data
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


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
        sequential_data = copy.deepcopy(mag_angle_rel[:, 2:8, :][:,:,2:]).cuda()

        _sequence.append(sequential_data)

    temp = torch.cat(_sequence, 0)
    
    return temp.permute(1,0,2)



def pretrain(model, epoch, train_loader,  optimizer, clip, print_every_batch, device):

    ### Initialise losses
    kld_loss = 0
    nll_loss = 0
    train_loss = 0

    ### Training mode
    model.train()

    for batch_idx, batch in enumerate(train_loader):

        ### Data transformation
        batch = [tensor.cuda() for tensor in batch]
        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, V_obs, V_pred) = batch
        full_rel = torch.cat((obs_traj_rel.squeeze(0), pred_traj_gt_rel.squeeze(0)), dim=1)
        tmp_rel = torch.zeros((full_rel.shape[0], full_rel.shape[1], 3))
        mag_angle_rel = torch.zeros((full_rel.shape[0], full_rel.shape[1], 4))
        for i in range(tmp_rel.shape[0]):
            tmp_rel[i] = angle_func(full_rel[i].cpu().numpy(), StandardNorm= True)
            mag_angle_rel[i] = mag_angle_func(full_rel[i].cpu().numpy(), StandardNorm = True)
        

        sequential_data = copy.deepcopy(mag_angle_rel[:, 2:8, :][:,:,2:]).to(device)
                         
        # Forward + backward + optimize
        optimizer.zero_grad()
        kld, nll, h, z_t, (all_enc_mean, all_enc_var), (all_dec_mean, all_dec_var) = model(sequential_data.permute(1,0,2))
        loss = kld + nll
        loss.backward()
        
        ### Visualisation training loss
        if batch_idx % print_every_batch == 0:
            run["train/batch/loss"].log(loss.item())

        nn.utils.clip_grad_norm_(model.parameters(), 20)
        optimizer.step()

        train_loss += loss.item()
        kld_loss += kld.item()
        nll_loss += nll.item()

    avg_loss = train_loss / len(train_loader.dataset)
    run["train/epoch/avg_loss"].log(avg_loss)
    avg_kld_loss = kld_loss / len(train_loader.dataset)
    run["train/epoch/avg_kld_loss"].log(avg_kld_loss)
    avg_nll_loss = nll_loss / len(train_loader.dataset)
    run["train/epoch/avg_nll_loss"].log(avg_nll_loss)

    print('====> Epoch: {} Average loss: {:.4f}'.format(
    epoch, avg_loss))


def main():

    parser = argparse.ArgumentParser(
        description='Pretrain for latent representation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    ### Training settings
    parser.add_argument('--n_epochs', '-e', help='Number of epochs.', type=int, default=100)
    parser.add_argument('--gpu', '-g', help='GPU id. (Negative number indicates CPU)', type=int, default=0)
    parser.add_argument('--learning_rate', '-l', help='Learning Rate.', type=float, default=1e-4)
    parser.add_argument('--out', '-o', help='Output path.', type=str, default='./pretrain.pth')

    ### Model settings
    parser.add_argument('--clip', default=10, type=int, required=False, help='Gradient clipping')
    parser.add_argument('--x_dim', default=2, type=int, required=False, help='Dimension of the input of the single agent')
    parser.add_argument('--h_dim', default=64, type=int, required=False, help='Dimension of the hidden layers')
    parser.add_argument('--z_dim', default=10, type=int, required=False, help='Dimension of the latent variables')
    parser.add_argument('--rnn_dim', default=64, type=int, required=False, help='Dimension of the recurrent layers')
    parser.add_argument('--n_layers', default=2, type=int, required=False, help='Number of recurrent layers')

    ### Miscellaneous
    parser.add_argument('--seed', default=128, type=int, required=False, help='PyTorch random seed')
    parser.add_argument('--print_every_batch', default=10, type=int, required=False, help='How many batches to print loss inside an epoch')
    parser.add_argument('--save_every', default=10, type=int, required=False, help='How often save model checkpoint')

    args = parser.parse_args()

    ### Neptune settings
    params = {
    "lr": args.learning_rate,
    }
    run["parameters"] = params


    ### Device cuda settings
    #if_use_cuda = torch.cuda.is_available() and args.gpu >= 0
    #device = torch.device('cuda:{}'.format(args.gpu) if if_use_cuda else 'cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ### Build datasets
    dataset = './dataset/stanford/'
    dset_train = AgentTrajectoryDataset(
    dataset + 'train/',
            obs_len= 8,
            pred_len= 12,
            skip=1)

    train_loader = DataLoader(
            dset_train,
            batch_size=256,  
            shuffle=True,
            num_workers=0, )


    model = VRNN(args.x_dim, args.z_dim, args.h_dim, args.rnn_dim, args.n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = params['lr'])

    torch.autograd.set_detect_anomaly(True)


    for epoch in range(1, args.n_epochs + 1):

        #training + testing
        pretrain(model, epoch, train_loader,  optimizer, args.clip, args.print_every_batch, device)

        if epoch % args.save_every == 0:
            fn = 'saves/vrnn_state_dict_'+str(epoch)+'.pth'
            torch.save(model.state_dict(), fn)
    

    with torch.no_grad():
        ### All data
        x = data_processor(train_loader).to(device)
        _  , _, h, z_t, (all_enc_mean, all_enc_var), (all_dec_mean, all_dec_var)= model(x)

    model = model.cpu()
    state_dict = model.state_dict()

    gmm_trained = GaussianMixture(n_components=6, covariance_type='diag')
    gmm_trained.fit(z_t.cpu())

    new_model = VARC(args.x_dim, args.z_dim, args.h_dim, args.rnn_dim, args.n_layers, n_classes=6)
    new_model.load_state_dict(state_dict, strict=False)
    new_model._pi.data = torch.log(torch.from_numpy(gmm_trained.weights_)).float()
    new_model.mu.data = torch.from_numpy(gmm_trained.means_).float()
    new_model.logvar.data = torch.log(torch.from_numpy(gmm_trained.covariances_)).float()
    torch.save(new_model.state_dict(), "./varc_parameter.pth")

    # #saving model
    # if epoch % args.save_every == 0:
    #     fn = 'saves_1/vrnn_state_dict_'+str(epoch)+'.pth'
    #     torch.save(model.state_dict(), fn)
    #     print('Saved model to '+fn)

if __name__ == "__main__":
    #manual seed
    def set_random_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    set_random_seed(128)
    main()