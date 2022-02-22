#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import DataLoader
from torch_geometric.nn import GraphConv,TopKPooling, ResGatedGraphConv, EdgePooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import pickle
import os
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from model import NPC_Train_Dataset, NPC_Test_Dataset, NPC_val_Dataset, Net

feature_num = 31
path = './'
Clin_path = './data'
train_dataset = NPC_Train_Dataset(root=os.path.join(path,'dataset/','dataset_train/','%d/')%feature_num)
test_dataset = NPC_Test_Dataset(root=os.path.join(path,'dataset/','dataset_test/','%d/')%feature_num)
val_dataset = NPC_val_Dataset(root=os.path.join(path,'dataset/','dataset_val/','%d/')%feature_num)
train_loader = DataLoader(train_dataset, batch_size=60)
test_loader = DataLoader(test_dataset, batch_size=60)
train_loader_all = DataLoader(train_dataset, batch_size=len(train_dataset))
test_loader_all = DataLoader(test_dataset, batch_size=len(test_dataset))
val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))
channel=128
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = Net(channel).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
crit = torch.nn.BCELoss()

def get_train():
    model.eval()
    for data in train_loader_all:
        data = data.to(device)
        pred,out = model(data)
    return out

def get_test():
    model.eval()
    for data in test_loader_all:
        data = data.to(device)
        pred,out = model(data)
    return out

def get_val():
    model.eval()
    for data in val_loader:
        data = data.to(device)
        pred,out = model(data)
    return out

def get_train_feature(epoch):
    out= get_train()
    out=out.detach().cpu().numpy()
    out=pd.DataFrame(out)
    out.to_excel(os.path.join(path,'loss/','%d/','train_%d_%d.xlsx')%(feature_num,channel,epoch), index=False)

def get_test_feature(epoch):
    out= get_test()
    out=out.detach().cpu().numpy()
    out=pd.DataFrame(out)
    out.to_excel(os.path.join(path,'loss/','%d/','test_%d_%d.xlsx')%(feature_num,channel,epoch), index=False)

def get_val_feature(epoch):
    out= get_val_1()
    out=out.detach().cpu().numpy()
    out=pd.DataFrame(out)
    out.to_excel(os.path.join(path,'loss/','%d/','val_%d_%d.xlsx')%(feature_num,channel,epoch), index=False)

def train(epoch):
    model.train()
    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output,out = model(data)
        label = data.y.to(device)
        loss = crit(output, label)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)

def test(loader):
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred,out = model(data)
            pred = pred.detach().cpu().numpy()
            label = data.y.detach().cpu().numpy()
            predictions.append(pred)
            labels.append(label)
    predictions = np.hstack(predictions)
    labels = np.hstack(labels)
    auc = roc_auc_score(labels, predictions)
    return auc

def get_loss(epoch):
    x1 = range(1, epoch+1)
    x2 = range(1, epoch+1)
    y1 = train_Accuracy_list
    y2 = train_Loss_list
    y3 = test_Accuracy_list
    y4 = test_Loss_list
    plt.subplot(2, 1, 1)
    # plt.plot(x1, y1, 'o-',color='r')
    plt.plot(x1, y1, '.-',label="train_Auc")
    plt.plot(x1, y3, '.-',label="test_Auc")
    plt.title('Test auc vs. epoches')
    plt.ylabel('Test auc')
    plt.legend(loc='best')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, '.-',label="train_Loss")
    plt.plot(x2, y4, '.-',label="test_Loss")
    plt.xlabel('Test loss vs. epoches')
    plt.ylabel('Test loss')
    plt.legend(loc='best')
    plt.savefig(os.path.join(path,'loss/','%d/','auc_%d.png')%(feature_num,epoch))
    plt.clf()

def test_loss(epoch):
    model.eval()
    loss_all = 0
    #with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        output,out = model(data)
        label = data.y.to(device)
        loss = crit(output, label)
        loss_all += data.num_graphs * loss.item()
    return loss_all / len(test_dataset)

train_Loss_list = []
train_Accuracy_list = []
test_Loss_list = []
test_Accuracy_list = []

item = 101
for epoch in range(1, item):
    loss_train = train(epoch)
    loss_test = test_loss(epoch)
    train_auc = test(train_loader)
    test_auc = test(test_loader)
    print('Epoch: {:03d}, Loss_train: {:.5f}, Loss_test: {:.5f}, Train Auc: {:.5f}, Test Auc: {:.5f}'.format(epoch, loss_train, loss_test, train_auc, test_auc))
    train_Loss_list.append(loss_train)
    train_Accuracy_list.append(train_auc)
    test_Loss_list.append(loss_test)
    test_Accuracy_list.append(test_auc)
    if test_auc > 0.7:
        get_loss(epoch)
        torch.save(model.state_dict(),os.path.join(path,'model/','%d/','mrmr_%d.pth')%(feature_num,epoch))
        get_train_feature(epoch)
        get_test_feature(epoch)
        get_val_feature(epoch)