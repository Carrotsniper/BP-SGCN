import torch
from soft_DTW import SoftDTW
import numpy as np

def rotate_pc(coords, alpha):
    alpha = alpha * np.pi / 180
    M = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
    return M @ coords



def expert_find(data, data_ori, expert_set, expert_ori, angles=None, option=1, device = 'cuda'):
    # global args
    """
    data: [test_batch, seq, 2]
    expert_set : [train_batch, seq ,2]
    """

    all_min_end = []
    rest_diff = []
    ceriterion = SoftDTW(
        use_cuda=True,
        gamma=2.0,
        normalize=True,
    )

    mse = torch.nn.MSELoss()

    num_of_trajs = data.shape[0]
    # print("Total number of searched data {}".format(num_of_trajs))
    """Pre-process to velocity and accer"""

    gradient_eff = 0.6
    traj_v = np.gradient(np.transpose(data, (0, 2, 1)), gradient_eff, axis=-1)
    traj_a = np.gradient(traj_v, gradient_eff, axis=-1)
    traj_v = torch.from_numpy(traj_v).permute(0, 2, 1).cuda()
    traj_a = torch.from_numpy(traj_a).permute(0, 2, 1).cuda()

    # TODO: apply random rotation here
    extra_data = []
    extra_ori = []
    if angles is not None:
        for ang in angles:
            expert_copy = np.copy(expert_set)
            expert_ori_copy = np.copy(expert_ori)
            B, T, C = expert_copy.shape
            expert_copy = expert_copy.reshape(B * T, C).transpose()
            expert_ori_copy = expert_ori_copy.reshape(B * T, C).transpose()

            expert_copy = rotate_pc(expert_copy, ang).transpose()
            expert_ori_copy = rotate_pc(expert_ori_copy, ang).transpose()
            extra_data.append(expert_copy.reshape(B, T, C))
            extra_ori.append(expert_ori_copy.reshape(B, T, C))

    expert_set = np.concatenate(extra_data, 0)
    expert_ori = np.concatenate(extra_ori, 0)

    expert_traj_v = np.gradient(
        np.transpose(expert_set, (0, 2, 1)), gradient_eff, axis=-1
    )
    expert_traj_a = np.gradient(expert_traj_v, gradient_eff, axis=-1)
    expert_traj_v = torch.from_numpy(expert_traj_v).permute(0, 2, 1).cuda()
    expert_traj_a = torch.from_numpy(expert_traj_a).permute(0, 2, 1).cuda()

    expert_set = torch.from_numpy(expert_set).cuda()
    expert_ori = torch.from_numpy(expert_ori).cuda()
    data = torch.DoubleTensor(data).to(device)
    data_ori = torch.DoubleTensor(data_ori).to(device).squeeze()

    """
        For random few shot ablation study 
    """
    # random_split_ratio = 0.9
    # expert_size = expert_traj_v.shape[0]
    # print(int(expert_size * random_split_ratio))
    # indice = random.sample(range(expert_size), int(expert_size * random_split_ratio))
    # indice = torch.tensor(indice)
    # print(len(set(indice)))
    # expert_traj_v = expert_traj_v[indice]
    # expert_set = expert_set[indice]
    # print(expert_traj_v.shape)

    # t0 = time.time()
    for i in range(num_of_trajs):

        tmp_traj = traj_v[i, :8].unsqueeze(0)
        tmp_traj_abs = data[i, :8].unsqueeze(0)

        expert_num = expert_traj_v.shape[0]

        tmp_traj = tmp_traj.repeat(expert_num, 1, 1)
        tmp_traj_abs = tmp_traj_abs.repeat(expert_num, 1, 1)

        loss = ceriterion(tmp_traj, expert_traj_v[:, :8])

        if option == 1:
            """Opt1: for dtw matching only"""
            min_k, min_k_indices = torch.topk(loss, 20, largest=False)

        elif option == 2:
            """Opt2: for dtw matching + clustering matching"""
            min_k, min_k_indices = torch.topk(loss, 65, largest=False)
            retrieved_expert = expert_set[min_k_indices][:, -1]
            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=20, random_state=0).fit(
                retrieved_expert.cpu().numpy()
            )

        iter_target = min_k_indices

        min_k_end = []
        end_point_appr = []

        """Back to indexing in real coords domain"""
        if option == 1:
            for k in iter_target:
                test_end = data[i, -1]
                exp_end = expert_set[k, -1]
                min_k_end.append(torch.norm(test_end - exp_end, 2))
                end_point_appr.append(exp_end)

            all_min_end.append(min(min_k_end))
            rest_diff.append(end_point_appr[min_k_end.index(min(min_k_end))])
        else:
            for k in kmeans.cluster_centers_:
                test_end = data[i, -1]
                exp_end = torch.from_numpy(k).cuda()

                min_k_end.append(torch.norm(test_end - exp_end, 2))
                end_point_appr.append(exp_end)

            all_min_end.append(min(min_k_end))

            rest_diff.append(end_point_appr[min_k_end.index(min(min_k_end))])

    return all_min_end, rest_diff