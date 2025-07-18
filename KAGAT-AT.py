import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import KANLayer
from torch_geometric.utils import softmax

grid_size=5     # default 5
spline_order=3   # default 3

# KAGAT-AT Layer (Sparse Graph)
class KANGraphAttentionLayerARC2(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super(KANGraphAttentionLayerARC2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = 0.2
        self.KAN = KANLayer(in_features=in_features, out_features=out_features)
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    def forward(self, h, edge_index):
        HW = h @ self.W
        HW_KAN = self.KAN(h)
        row, col = edge_index  # 提取边的起点和终点
        HW_i = HW[row]  # 起点特征
        HW_j = HW[col]  # 终点特征
        HW_KAN_j = HW_KAN[col]
        a_input = torch.cat([HW_i, HW_j], dim=1)  # Shape: [E, 2 * out_features]
        e = self.leakyrelu(a_input @ self.a).squeeze(-1)  # Shape: [E]
        attention = softmax(e, index=row)  # 基于起点 row 对 e 进行 softmax
        attention = F.dropout(attention, p=0.6, training=self.training)  # 应用 Dropout
        h = torch.zeros_like(HW).scatter_add_(0, row.unsqueeze(-1).expand(-1, self.out_features), attention.unsqueeze(-1) * HW_KAN_j)
        return h

# KAGAT-NA2 2-Layer Model (Sparse Graph)
class KAGATa2(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nheads):
        super(KAGATa2, self).__init__()
        self.dropout = dropout
        self.attentions = nn.ModuleList(
            [KANGraphAttentionLayerARC2(nfeat, nhid, dropout=dropout) for _ in range(nheads)]
        )
        self.out_att = KANGraphAttentionLayerARC2(nhid * nheads, nclass, dropout=dropout)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = torch.cat([att(x, edge_index) for att in self.attentions], dim=1)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.out_att(x, edge_index)
        return x # 配合crossentropy_loss (隐含softmax实现）
      
# Dense
'''
class KANGraphAttentionLayerARC2(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super(KANGraphAttentionLayerARC2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = 0.2
        self.KAN = KANLayer(in_features=in_features, out_features=out_features)
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    def forward(self, h, adj):
        HW = h @ self.W
        HWa1T = torch.matmul(HW, self.a[:self.out_features, :])
        HWa2T = torch.matmul(HW, self.a[self.out_features:, :])
        e = HWa1T + HWa2T.T  
        e = self.leakyrelu(e)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, p=0.6, training=self.training)
        h = attention @ self.KAN(h)
        return h

class KAGATa2(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nheads):
        super(KAGATa2, self).__init__()
        self.dropout = dropout
        self.attentions = nn.ModuleList(
            [KANGraphAttentionLayerARC2(nfeat, nhid, dropout=dropout) for _ in range(nheads)]
        )
        self.out_att = KANGraphAttentionLayerARC2(nhid * nheads, nclass, dropout=dropout)
    def forward(self, x, adj):
        x = F.dropout(x, p=0.6, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.out_att(x, adj)
        return x # 配合crossentropy_loss (隐含softmax实现）
'''
