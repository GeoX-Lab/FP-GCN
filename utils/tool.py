import torch
import numpy as np
import pickle
from torch_geometric.utils import to_dense_adj, add_self_loops,dense_to_sparse,remove_self_loops
import random


def init_seed(seed=502):
    init_seed = seed
    torch.manual_seed(init_seed)
    torch.cuda.manual_seed(init_seed)
    torch.cuda.manual_seed_all(init_seed)
    random.seed(init_seed)
    np.random.seed(init_seed) # 用于numpy的随机数
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def todeeprobust(data):
    pass

def distur_edge(data,dis_n,file):
    with open(file, 'rb') as f:
        str1 = pickle.load(f)

    _A_obs = to_dense_adj(data.edge_index)
    _A_obs = torch.squeeze(_A_obs)
    n = len(str1)
    # i 表示当前的节点序号
    #j 表示当前节点的候选扰动集个数
    for i in range(n):
        m = len(str1[i])
        m = m if m < dis_n else dis_n
        for j in range(m):
            if str1[i][j]:
                x = str1[i][j][0]
                y = str1[i][j][1]
                _A_obs[x][y] += 1
    #_A_obs[_A_obs > 1] = 1
    _A_obs[_A_obs %2==0] = 0
    _A_obs[_A_obs %2 == 1] = 1
    data.edge_index = remove_self_loops(dense_to_sparse(_A_obs)[0])[0]
    print("ok")



def distur_edge_way2(data,dis_n,file):
    with open(file, 'rb') as f:
        str1 = pickle.load(f)

    _A_obs = to_dense_adj(data.edge_index)
    _A_obs = torch.squeeze(_A_obs)
    n = len(str1)
    # i 表示当前的节点序号
    #j 表示当前节点的候选扰动集个数
    for i in range(n):
        m = len(str1[i])
        m = m if m < dis_n else dis_n
        for j in range(m):
            if str1[i][j]:
                x = str1[i][j][0]
                y = str1[i][j][1]
                _A_obs[x][y] += 1
    #_A_obs[_A_obs > 1] = 1
    _A_obs[_A_obs %2==0] = 0
    _A_obs[_A_obs %2 == 1] = 1
    data.edge_index = remove_self_loops(dense_to_sparse(_A_obs)[0])[0]
    print("ok")