#-*- coding:utf-8 -*-
import torch
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from torch_geometric.nn import GraphConv,TopKPooling, ResGatedGraphConv, EdgePooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from typing import Optional, Callable, List
from torch_geometric.typing import Adj
import copy
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList, Sequential, Linear, BatchNorm1d, ReLU
from torch_geometric.nn.conv import (GCNConv, SAGEConv, GINConv, GATConv,PNAConv)
from torch_geometric.nn.models.jumping_knowledge import JumpingKnowledge
import os

feature_list = []
feature_num = 31  
df_path = './data'
dataset_path = './dataset/'
df_train = pd.read_excel(os.path.join(df_path,'data_train.xlsx'))
df_test = pd.read_excel(os.path.join(df_path,'data_test.xlsx'))
df_val = pd.read_excel(os.path.join(df_path,'data_val.xlsx'))

class NPC_Train_Dataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(NPC_Train_Dataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return [os.path.join(dataset_path,'dataset_train/','%d/','NPC.dataset')%feature_num]
    def download(self):
        pass
    def process(self):
        data_list = []
        grouped = df_train.groupby('session_id')
        for session_id, group in tqdm(grouped):
            sess_item_id = LabelEncoder().fit_transform(group.item_id)
            group = group.reset_index(drop=True)
            group['sess_item_id'] = sess_item_id
            length = len(group.item_id)
            node_features = group.loc[
                group.session_id == session_id, feature_list[:feature_num]].values
            node_features = torch.tensor(node_features, dtype=torch.float)
            b=[]
            c=[]
            d=[]
            for i in range(length):
                a=[i]*(length-1)
                b.extend(a)
            for j in range(length):
                c=[k for k in range(length)]
                c.remove(j)
                d.extend(c)
            target_nodes = b
            source_nodes = d
            edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
            x = node_features
            y = torch.tensor([group.DFS.values[0]], dtype=torch.float)
            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class NPC_Test_Dataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(NPC_Test_Dataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return [os.path.join(dataset_path,'dataset_test/','%d/','NPC.dataset')%feature_num]
    def download(self):
        pass
    def process(self):
        data_list = []
        # process by session_id
        grouped = df_test.groupby('session_id')
        for session_id, group in tqdm(grouped):
            sess_item_id = LabelEncoder().fit_transform(group.item_id)
            group = group.reset_index(drop=True)
            group['sess_item_id'] = sess_item_id
            length = len(group.item_id)
            node_features = group.loc[
                group.session_id == session_id, feature_list[:feature_num]].values
            node_features = torch.tensor(node_features, dtype=torch.float)
            b=[]
            c=[]
            d=[]
            for i in range(length):
                a=[i]*(length-1)
                b.extend(a)
            for j in range(length):
                c=[k for k in range(length)]
                c.remove(j)
                d.extend(c)
            target_nodes = b
            source_nodes = d
            edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
            x = node_features
            y = torch.tensor([group.DFS.values[0]], dtype=torch.float)
            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class NPC_val_Dataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(NPC_val_Dataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return [os.path.join(dataset_path,'dataset_val/','%d/','NPC.dataset')%feature_num]
    def download(self):
        pass
    def process(self):
        data_list = []
        # process by session_id
        grouped = df_val.groupby('session_id')
        for session_id, group in tqdm(grouped):
            sess_item_id = LabelEncoder().fit_transform(group.item_id)
            group = group.reset_index(drop=True)
            group['sess_item_id'] = sess_item_id
            length = len(group.item_id)
            node_features = group.loc[
                group.session_id == session_id, feature_list[:feature_num]].values
            node_features = torch.tensor(node_features, dtype=torch.float)
            b=[]
            c=[]
            d=[]
            for i in range(length):
                a=[i]*(length-1)
                b.extend(a)
            for j in range(length):
                c=[k for k in range(length)]
                c.remove(j)
                d.extend(c)
            target_nodes = b
            source_nodes = d
            edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
            x = node_features
            y = torch.tensor([group.DFS.values[0]], dtype=torch.float)
            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

num_fea = feature_num
class Net(torch.nn.Module):
    def __init__(self,channel):
        super(Net, self).__init__()
        self.conv01 = ResGatedGraphConv(num_fea, channel)
        self.bn01 = torch.nn.BatchNorm1d(channel)
        self.conv02 = ResGatedGraphConv(channel, channel)
        self.bn02 = torch.nn.BatchNorm1d(channel)
        self.pool1 = TopKPooling(channel, ratio=0.9)

        self.conv11 = ResGatedGraphConv(channel, channel*2)
        self.bn11 = torch.nn.BatchNorm1d(channel*2)
        self.conv12 = ResGatedGraphConv(channel*2, channel*2)
        self.bn12 = torch.nn.BatchNorm1d(channel*2)
        self.pool2 = TopKPooling(channel*2)

        self.conv21 = ResGatedGraphConv(channel*2, channel*4)
        self.bn21 = torch.nn.BatchNorm1d(channel*4)
        self.conv22 = ResGatedGraphConv(channel*4, channel*4)
        self.bn22 = torch.nn.BatchNorm1d(channel*4)
        self.pool3 = TopKPooling(channel*4)

        self.lin1 = torch.nn.Linear(channel * 8, channel * 4)
        self.lin2 = torch.nn.Linear(channel * 4, channel * 2)
        self.lin3 = torch.nn.Linear(channel * 2, 1)
        self.bn1 = torch.nn.BatchNorm1d(channel * 4)
        self.bn2 = torch.nn.BatchNorm1d(channel * 2)
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x1 = self.conv01(x, edge_index)
        x1 = self.bn01(x1)
        x1 = F.relu(x1)
        x1 = self.conv02(x, edge_index)
        x1 = self.bn02(x1)
        x1 = F.relu(x1)
        x1, edge_index, _, batch, _, _ = self.pool1(x1, edge_index, None, batch)
        out1 = torch.cat([gmp(x1, batch), gap(x1, batch)], dim=1)

        x2 = self.conv11(x1, edge_index)
        x2 = self.bn11(x2)
        x2 = F.relu(x2)
        x2 = self.conv12(x2, edge_index)
        x2 = self.bn12(x2)
        x2 = F.relu(x2)
        x2, edge_index, _, batch, _, _ = self.pool2(x2, edge_index, None, batch)
        out2 = torch.cat([gmp(x2, batch), gap(x2, batch)], dim=1)

        x3 = self.conv21(x2, edge_index)
        x3 = self.bn21(x3)
        x3 = F.relu(x3)
        x3 = self.conv22(x3, edge_index)
        x3 = self.bn22(x3)
        #x3 = x3+x2
        x3 = F.relu(x3)
        x3, edge_index, _, batch, _, _ = self.pool3(x3, edge_index, None, batch)
        out3 = torch.cat([gmp(x3, batch), gap(x3, batch)], dim=1)

        out = self.lin1(out3)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.lin2(out)
        out = self.bn2(out)
        out = self.act2(out)
        feature = out
        out = F.dropout(out, p=0.5)
        out = torch.sigmoid(self.lin3(out)).squeeze(1)
        return out,feature