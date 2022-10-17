import torch
from torch_geometric.nn import GraphConv
from torch_geometric.nn.pool.topk_pool import topk, filter_adj
from torch_geometric.utils import softmax
import numpy as np
from torch import tensor
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul_
#debug
from torch_geometric.utils import to_dense_adj, dense_to_sparse
class MeanunPooling(MessagePassing):
    def __init__(self):
        super(MeanunPooling, self).__init__()

    @staticmethod
    def norm(edge_index, num_nodes, minus, edge_weight=None, improved=False,
             dtype=None):

        if minus.shape[0] != 0:
            # for i in range(edge_index.shape[1]):
            #     if edge_index[1, i] in minus:
            #         del edge_index[:, i]2691287
            tmp_edge = to_dense_adj(edge_index)[0]
            if tmp_edge.size(0) != tmp_edge.size(1):
                print('error!')
            minus_dense = torch.ones((1, tmp_edge.size(0)), device=tmp_edge.device)
            # minus_dense[0][minus] = 0
            minus_dense.scatter_(1, minus.view(1, -1), 0)
            tmp_edge = tmp_edge * minus_dense
            edge_index = dense_to_sparse(tmp_edge)[0]

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x1, x2, edge_index1, edge_index2, perm, edge_weight=None):
        tmp = torch.zeros(x1.shape, dtype=torch.float32, device=x1.device)
        tmpperm = np.arange(x1.size(0),)
        minus = set(tmpperm) - set(perm.cpu().detach().numpy())
        minus = torch.from_numpy(np.array(list(minus))).to(perm.device)

        tmp[perm] = x2
        # norm = self.norm(edge_index1, x1.size(0), minus, edge_weight,)
        tmp_minus = self.propagate(edge_index1, x=tmp)
        if minus.shape[0] != 0:
            tmp[minus] = tmp_minus[minus]

        return tmp, edge_index1

class v3_MeanunPooling(MessagePassing):
    def __init__(self):
        super(v3_MeanunPooling, self).__init__()

    @staticmethod
    def norm(edge_index, num_nodes, minus, edge_weight=None, improved=False,
             dtype=None):

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


    def forward(self, x1, x2, edge_index1, edge_index2, perm, edge_weight=None):
        tmp = torch.zeros(x1.shape, dtype=torch.float32, device=x1.device)
        tmpperm = np.arange(x1.size(0),)
        minus = set(tmpperm) - set(perm.cpu().detach().numpy())
        minus = torch.from_numpy(np.array(list(minus))).to(perm.device)

        tmp[perm] = x2
        norm = self.norm(edge_index1, x1.size(0),  edge_weight,)
        tmp_minus = self.propagate(edge_index1, x=tmp, norm=norm)
        if minus.shape[0] != 0:
            tmp[minus] = tmp_minus[minus]

        return tmp, edge_index1