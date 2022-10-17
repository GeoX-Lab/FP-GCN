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
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=120)
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
    def __init__(self, dim=128):
        super(Net, self).__init__()
        in_channels = dataset.num_features
        self.dim = dim
        self.two_dim = self.dim*2
        self.merge_dim = self.dim*2*3
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

        self.attn1 = torch.nn.Linear(self.two_dim, 2)
        self.bn61 = torch.nn.BatchNorm1d(self.two_dim)


        self.attn2 = torch.nn.Linear(self.two_dim, 2)
        self.bn62 = torch.nn.BatchNorm1d(self.two_dim)


        self.attn3 = torch.nn.Linear(self.two_dim, 2)
        self.bn63 = torch.nn.BatchNorm1d(self.two_dim)


        self.attn = torch.nn.Linear(self.merge_dim, 3)
        self.bn6 = torch.nn.BatchNorm1d(self.merge_dim)
        self.lin1 = torch.nn.Linear(self.merge_dim, dataset.num_classes)

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
        x2_attn = F.softmax(self.attn1(x2), dim=1)
        x2 = torch.cat([x2[:, :self.dim] * x2_attn[:, 0].unsqueeze(dim=1), x2[:, self.dim:] * x2_attn[:, 1].unsqueeze(dim=1)], dim=1)
        x2 = self.bn61(x2)

        x3 = torch.cat([gmp(x3, batch1), gap(x3, batch1)], dim=1)
        x3_attn = F.softmax(self.attn2(x3), dim=1)
        x3 = torch.cat([x3[:, :self.dim] * x3_attn[:, 0].unsqueeze(dim=1), x3[:, self.dim:] * x3_attn[:, 1].unsqueeze(dim=1)], dim=1)
        x3 = self.bn62(x3)

        x4 = torch.cat([gmp(x4, batch0), gap(x4, batch0)], dim=1)
        x4_attn = F.softmax(self.attn3(x4), dim=1)
        x4 = torch.cat([x4[:, :self.dim] * x4_attn[:, 0].unsqueeze(dim=1), x4[:, self.dim:] * x4_attn[:, 1].unsqueeze(dim=1)], dim=1)
        x4 = self.bn63(x4)

        x = torch.cat([x2, x3, x4], dim=1)

        atten = self.attn(x)
        # atten = torch.sigmoid(atten)
        atten = F.softmax(atten, dim=1)
        # x = torch.mul(atten, x)
        x = torch.cat([x[:, :self.two_dim]*atten[:, 0].unsqueeze(dim=1), x[:, self.two_dim:-self.two_dim]*atten[:, 1].unsqueeze(dim=1), x[:, -self.two_dim:]*atten[:, 2].unsqueeze(dim=1)], dim=1)
        x = self.bn6(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin1(x))
        return F.log_softmax(x, dim=-1)




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)


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
    for data in loader:
        data = data.to(device)
        pred = model(data).max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)

max_log, max_acc = '', 0
best_val_acc = test_acc = 0
max_patience, patience = 50, 0
for epoch in tqdm(range(config.epoch)):
    loss = train(train_loader)
    train_acc = test(train_loader)
    val_acc = test(val_loader)
    if val_acc > best_val_acc:
        test_acc = test(test_loader)
        torch.save(model.state_dict(), '../run/GraphFPN_dataset{}_seed{}.pth'.format(config.dataname, args.seed))
        best_val_acc = val_acc
        patience = 0
        if test_acc > max_acc:
            max_acc = test_acc
            max_log = 'Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, val Acc: {:.5f}, Test Acc: {:.5f}'.format(epoch, loss, train_acc, val_acc, test_acc)
    else:
        patience += 1
    if patience > max_patience:
        break
    print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, val Acc: {:.5f}, Test Acc: {:.5f}'.
          format(epoch, loss, train_acc, val_acc, test_acc))
fin_log = 'Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, val Acc: {:.5f}, Test Acc: {:.5f}'.format(epoch, loss, train_acc, val_acc, test_acc)

with open(config.logfile, 'a', encoding='utf-8') as f:
    f.writelines('new-v1-GraphFPN' + '\n')
    f.writelines('dataset:' + config.dataname + '\n')
    f.writelines(max_log+'\n')
    f.writelines(fin_log+'\n')
    f.writelines('-----------------------------------------'+'\n')


