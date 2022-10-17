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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=120)
args = parser.parse_args()

init_seed(args.seed)

path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', config.dataname)
# dataset = TUDataset(path, name=config.dataname, use_node_attr=True, cleaned=True).shuffle()
dataset = TUDataset(path, name=config.dataname,  cleaned=True).shuffle()
test_dataset = dataset[:len(dataset) // config.data_split]
train_dataset = dataset[len(dataset) // config.data_split:]
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=1, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=1, shuffle=False)


class Net(torch.nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        in_channels = dataset.num_features
        dim = 64
        self.conv1 = GINConv(Seq(Lin(in_channels, dim), ReLU(), Lin(dim, dim)))
        # self.pool1 = SAGPooling(dim, min_score=0.001, GNN=GCNConv)
        self.pool1 = SAGPooling(64, ratio=0.5, GNN=GCNConv)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        self.conv2 = GINConv(Seq(Lin(dim, dim), ReLU(), Lin(dim, dim)))
        # self.pool2 = SAGPooling(dim, min_score=0.001, GNN=GCNConv)
        self.pool2 = SAGPooling(64, ratio=0.5, GNN=GCNConv)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        self.conv3 = GINConv(Seq(Lin(dim, dim), ReLU(), Lin(dim, dim)))
        self.bn3 = torch.nn.BatchNorm1d(dim)

        self.conv4 = GINConv(Seq(Lin(64, 64), ReLU(), Lin(64, 64)))
        self.unpool1 = v3_MeanunPooling()
        self.bn4 = torch.nn.BatchNorm1d(dim)

        self.conv5 = GINConv(Seq(Lin(64, 64), ReLU(), Lin(64, 64)))
        self.unpool2 = v3_MeanunPooling()
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.attn = torch.nn.Linear(384, 3)
        self.bn6 = torch.nn.BatchNorm1d(384)
        self.lin1 = torch.nn.Linear(384, dataset.num_classes)

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

        # Layer A
        x0 = F.relu(self.conv1(x, edge_index))
        x1, edge_index, _, batch, perm, score = self.pool1(x0, edge_index, None, batch)
        x1 = self.bn1(x1)
        edge_index1 = edge_index
        perm1 = perm
        batch1 = batch

        # Layer B
        x1 = F.relu(self.conv2(x1, edge_index))
        x2, edge_index, _, batch, perm, score = self.pool2(x1, edge_index, None, batch)
        x2 = self.bn2(x2)
        edge_index2 = edge_index
        perm2 = perm
        batch2 = batch

        # P: Layer B
        x2 = F.relu(self.conv3(x2, edge_index))
        x2 = self.bn3(x2)

        x3, edge_index = self.unpool1(x1, x2, edge_index1, edge_index2, perm2)
        x3 = x3 + self.resweight1*x1
        x3 = F.relu(self.conv4(x3, edge_index))
        x3 = self.bn4(x3)

        # P: layer B
        x4, edge_index = self.unpool2(x0, x3, edge_index0, edge_index1, perm1)
        x4 = x4 + self.resweight2*x0
        x4 = F.relu(self.conv5(x4, edge_index))
        x4 = self.bn5(x4)

        # x = gmp(x4, batch0)
        x2 = torch.cat([gmp(x2, batch2), gap(x2, batch2)], dim=1)
        x3 = torch.cat([gmp(x3, batch1), gap(x3, batch1)], dim=1)
        x4 = torch.cat([gmp(x4, batch0), gap(x4, batch0)], dim=1)

        x = torch.cat([x2, x3, x4], dim=1)

        atten = self.attn(x)
        # atten = torch.sigmoid(atten)
        atten = F.softmax(atten, dim=1)
        # x = torch.mul(atten, x)
        x = torch.cat([x[:, :128]*atten[:, 0].unsqueeze(dim=1), x[:, 128:-128]*atten[:, 1].unsqueeze(dim=1), x[:, -128:]*atten[:, 2].unsqueeze(dim=1)], dim=1)
        x = self.bn6(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin1(x))
        return F.log_softmax(x, dim=-1)




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


def train(epoch):
    model.train()

    loss_all = 0
    for data in train_loader:
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

max_log = ''
max_acc = 0
for epoch in range(config.epoch):
    loss = train(epoch)
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    if test_acc>max_acc:
        max_acc = test_acc
        max_log = 'Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, Test Acc: {:.5f}'.format(epoch, loss, train_acc, test_acc)
    print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, Test Acc: {:.5f}'.
          format(epoch, loss, train_acc, test_acc))
fin_log = 'Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, Test Acc: {:.5f}'.format(epoch, loss, train_acc, test_acc)
with open(config.logfile, 'a', encoding='utf-8') as f:
    f.writelines('v6-GraphFPN' + '\n')
    f.writelines('dataset:' + config.dataname + '\n')
    f.writelines(max_log+'\n')
    f.writelines(fin_log+'\n')
    f.writelines('-----------------------------------------'+'\n')


