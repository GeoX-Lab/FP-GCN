import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GINConv, GCNConv, SAGPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_sort_pool as gsp

from torch_geometric.utils import to_dense_adj,dropout_adj
from utils.tool import *
from utils.MeanunPooling import v3_MeanunPooling
from config import config
from torch.utils.data import random_split
from tqdm import tqdm
from sklearn import manifold
import matplotlib.pyplot as plt
import argparse

from pyod.models.knn import KNN
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=105)
args = parser.parse_args()

init_seed(args.seed)

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', config.dataname)
# dataset = TUDataset(path, name=config.dataname, use_node_attr=True, cleaned=True).shuffle()
dataset = TUDataset(path, name=config.dataname)
dataset.data, dataset.slices = torch.load(dataset.processed_paths[0])
test_num = int(len(dataset) * 0.1)
train_num = int(len(dataset) * 0.8)
val_num = len(dataset) - train_num - test_num
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_num, val_num,  test_num])
# test_dataset = dataset[-test_num:]
# val_dataset = dataset[train_num: -test_num]
# train_dataset = dataset[:train_num]
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=1, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=1, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=1, shuffle=False)


class Net(torch.nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        in_channels = dataset.num_features
        dim = 128
        merge_dim = dim*2*3
        self.conv1 = GINConv(Seq(Lin(in_channels, dim), ReLU(), Lin(dim, dim)))
        # self.pool1 = SAGPooling(dim, min_score=0.001, GNN=GCNConv)
        self.pool1 = SAGPooling(dim, ratio=0.5, GNN=GCNConv)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        self.conv2 = GINConv(Seq(Lin(dim, dim), ReLU(), Lin(dim, dim)))
        # self.pool2 = SAGPooling(dim, min_score=0.001, GNN=GCNConv)
        self.pool2 = SAGPooling(dim, ratio=0.5, GNN=GCNConv)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        self.conv3 = GINConv(Seq(Lin(dim, dim), ReLU(), Lin(dim, dim)))
        self.bn3 = torch.nn.BatchNorm1d(dim)

        self.conv4 = GINConv(Seq(Lin(dim, dim), ReLU(), Lin(dim, dim)))
        self.unpool1 = v3_MeanunPooling()
        self.bn4 = torch.nn.BatchNorm1d(dim)

        self.conv5 = GINConv(Seq(Lin(dim, dim), ReLU(), Lin(dim, dim)))
        self.unpool2 = v3_MeanunPooling()
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.attn = torch.nn.Linear(merge_dim, 3)
        self.bn6 = torch.nn.BatchNorm1d(merge_dim)
        self.lin1 = torch.nn.Linear(merge_dim, dataset.num_classes)

        self.resweight1 = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.resweight2 = nn.Parameter(torch.zeros(1), requires_grad=True)


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        edge_index, _ = dropout_adj(data.edge_index, p=0.2,
                                    force_undirected=True,
                                    num_nodes=data.num_nodes,
                                    training=self.training)
        edge_index0 = edge_index
        batch0 = batch

        #第A层
        x0 = F.relu(self.conv1(x, edge_index))
        x1, edge_index, _, batch, perm, score = self.pool1(x0, edge_index, None, batch)
        x1 = self.bn1(x1)
        edge_index1 = edge_index
        perm1 = perm
        batch1 = batch

        # 第B层
        x1 = F.relu(self.conv2(x1, edge_index))
        x2, edge_index, _, batch, perm, score = self.pool2(x1, edge_index, None, batch)
        x2 = self.bn2(x2)
        edge_index2 = edge_index
        perm2 = perm
        batch2 = batch

        # P：第B层
        x2 = F.relu(self.conv3(x2, edge_index))
        x2 = self.bn3(x2)

        x3, edge_index = self.unpool1(x1, x2, edge_index1, edge_index2, perm2)
        x3 = x3 + self.resweight1*x1
        x3 = F.relu(self.conv4(x3, edge_index))
        x3 = self.bn4(x3)

        # P：第A层
        x4, edge_index = self.unpool2(x0, x3, edge_index0, edge_index1, perm1)
        x4 = x4 + self.resweight2*x0
        x4 = F.relu(self.conv5(x4, edge_index))
        x4 = self.bn5(x4)

        # x = gmp(x4, batch0)
        x2 = torch.cat([gmp(x2, batch2), gap(x2, batch2)], dim=1)
        x3 = torch.cat([gmp(x3, batch1), gap(x3, batch1)], dim=1)
        x4 = torch.cat([gmp(x4, batch0), gap(x4, batch0)], dim=1)
        m2, m3, m4 = x2.cpu().detach().numpy(), x3.cpu().detach().numpy(), x4.cpu().detach().numpy()
        x = torch.cat([x2, x3, x4], dim=1)

        atten = self.attn(x)
        # atten = torch.sigmoid(atten)
        atten = F.softmax(atten, dim=1)
        # x = torch.mul(atten, x)
        x = torch.cat([x[:, :128]*atten[:, 0].unsqueeze(dim=1), x[:, 128:-128]*atten[:, 1].unsqueeze(dim=1), x[:, -128:]*atten[:, 2].unsqueeze(dim=1)], dim=1)
        x = self.bn6(x)
        m1 = x.cpu().detach().numpy()
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin1(x))
        m5 = x.cpu().detach().numpy()
        return F.log_softmax(x, dim=-1), m1, m2, m3, m4,m5




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
model.load_state_dict(torch.load('../run/GraphFPN_dataset{}_seed{}.pth'.format(config.dataname, args.seed)))

def train(loader):
    model.train()

    loss_all = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)


def test(loader):
    model.eval()

    correct = 0
    num = 0
    data1,data2,data3,data4,data5 = np.empty([0, 768]), np.empty([0, 256]), np.empty([0, 256]),np.empty([0, 256]),np.empty([0, 2]),
    y = []
    for data in loader:
        num +=1
        print(num)
        data = data.to(device)
        pred, m1, m2, m3, m4, m5 = model(data)
        data1 = np.concatenate((data1, m1), axis=0)
        data2 = np.concatenate((data2, m2), axis=0)
        data3 = np.concatenate((data3, m3), axis=0)
        data4 = np.concatenate((data4, m4), axis=0)
        data5 = np.concatenate((data5, m5), axis=0)

        pred = pred.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        y.extend(data.y.cpu().detach().numpy())


    tsne = manifold.TSNE(n_components=2, init='pca', random_state=args.seed)



    W1 = tsne.fit_transform(data2)
    W1, y = delpoint2(W1, y)
    # cValue = ['#fa5a5a', '#f0d264', '#8c4b31', '#1f9baa', '#6698cb', '#cb99c5', '#a1488e', '#bd2158''#82c8a0',
              # '#003366', ]
    cValue = ['#6698cb', '#bd2158']
    # cValue = ['#82c8a0','#003366']
    for i in tqdm(range(W1.shape[0])):
        plt.scatter(W1[i, 0], W1[i, 1], c=cValue[y[i]], marker='o')
    plt.savefig(
        "img/{}_{}.png".format(config.dataname, "data2"),
        dpi=600, format='png')
    plt.show()

    return correct / len(loader.dataset)

def delpoint(data, y):
    clf = KNN(contamination=0.01, method='mean')  # 初始化检测器clf
    x = np.sqrt(np.square(data[:, 0]) + np.square(data[:, 1]))
    clf.fit(x.reshape(-1, 1))  # 使用X_train训练检测器clf
    y_train_pred = clf.labels_  # 返回训练数据上的分类标签 (0: 正常值, 1: 异常值)
    y_train_scores = clf.decision_scores_  # 返回训练数据上的异常值 (分值越大越异常)
    for i in np.where(y_train_pred)[0]:
        data = np.delete(data, i, axis=0)
        del y[i]
    return data, y

def delpoint2(data, y):
    # x = np.sqrt(np.square(data[:, 0]) + np.square(data[:, 1]))
    y = np.array(y)
    x = data[:, 0]
    data = np.delete(data, np.where(x-x.mean() > 3*x.std())[0], axis=0)

    y = np.delete(y, np.where(x-x.mean() > 3*x.std())[0], axis=0)
    # for i in np.where(x-x.mean() > 3*x.std())[0]:
    #     data = np.delete(data, i, axis=0)
    #     del y[i]
    x = data[:, 1]
    data = np.delete(data, np.where(x - x.mean() > 3 * x.std())[0], axis=0)
    y = np.delete(y, np.where(x-x.mean() > 3*x.std())[0], axis=0)
    # for i in np.where(x-x.mean() > 3*x.std())[0]:
    #     data = np.delete(data, i, axis=0)
    #     del y[i]
    return data, y

test_acc = test(train_loader)
# test_acc = test(test_loader)
print(test_acc)
