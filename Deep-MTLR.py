from math import sqrt, ceil
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, roc_auc_score, recall_score,roc_curve, auc
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from tqdm import tqdm, trange
import pickle
from torchmtlr import (MTLR, mtlr_neg_log_likelihood,mtlr_survival, mtlr_survival_at_times,mtlr_risk)
from torchmtlr.utils import (make_time_bins, encode_survival,make_optimizer)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sns.set(context="poster", style="white")
plt.rcParams["figure.figsize"] = (10, 7)#设置图的尺寸

def normalize(data, mean=None, std=None, skip_cols=[]):
    """将 Pandas DataFrame 的列归一化为零均值和单位标准差"""
    if mean is None:
        mean = data.mean(axis=0)
    if std is None:
        std = data.std(axis=0)
    if skip_cols is not None:
        mean[skip_cols] = 0
        std[skip_cols] = 1
    return (data - mean) / std, mean, std
def reset_parameters(model):
    """重置 PyTorch 模块及其子模块的参数"""
    for m in model.modules():
        try:
            m.reset_parameters()
        except AttributeError:
            continue
    return model
# training functions
def train_mtlr(model, data_train, time_bins,
               num_epochs=1000, lr=.01, weight_decay=0.,
               C1=1., batch_size=None,
               verbose=True, device="cpu"):
    """使用小批量梯度下降训练 MTLR 模型"""
    x = torch.tensor(data_train.drop(["time", "event"], axis=1).values, dtype=torch.float)
    y = encode_survival(data_train["time"].values, data_train["event"].values, time_bins)
    optimizer = make_optimizer(Adam, model, lr=lr, weight_decay=weight_decay)
    reset_parameters(model)
    model = model.to(device)
    model.train()
    train_loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)
    pbar =  trange(num_epochs, disable=not verbose)
    for i in pbar:
        for xi, yi in train_loader:
            xi, yi = xi.to(device), yi.to(device)
            y_pred = model(xi)
            loss = mtlr_neg_log_likelihood(y_pred, yi, model, C1=C1, average=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        pbar.set_description(f"[epoch {i+1: 4}/{num_epochs}]")
        pbar.set_postfix_str(f"loss = {loss.item():.4f}")
    model.eval()
    return model
def train_mtlr_cv(model, data_train, time_bins, cv=3, C1_vals=None, verbose=True, device="cpu", **kwargs):
    """使用小批量梯度下降训练 MTLR 模型，通过 K 折交叉验证确定最佳 L2 正则化强度"""
    if C1_vals is None:
        C1_vals = np.logspace(-2, 3, 6)
    kfold = KFold(n_splits=cv)
    nll_vals = defaultdict(list)
    for C1 in C1_vals:
        pbar = tqdm(kfold.split(data_train),
                    total=cv,
                    disable=not verbose,
                    desc=f"testing C1 = {C1:9}")
        for train_idx, val_idx in pbar:
            train_fold, val_fold = data_train.iloc[train_idx], data_train.iloc[val_idx]
            time_val, event_val = data_train.iloc[val_idx]["time"].values, data_train.iloc[val_idx]["event"].values
            x_val = torch.tensor(val_fold.drop(["time", "event"], axis=1).values,
                                 dtype=torch.float, device=device)
            y_val = encode_survival(time_val, event_val, time_bins).to(device)
            model = train_mtlr(model, train_fold, time_bins, C1=C1, device=device, verbose=False, **kwargs)
            with torch.no_grad():
                val_nll = mtlr_neg_log_likelihood(model(x_val), y_val, model, C1=C1)
                nll_vals[C1].append(val_nll.item())
            pbar.set_postfix_str(f"val nll = {val_nll.item():.2f}")
    # Choose regularization parameter with the lowest negative log-likelihood
    best_C1 = min(nll_vals, key=lambda k: sum(nll_vals[k]) / cv)
    if verbose:
        print(f"training with C1 = {best_C1}")
    model = train_mtlr(model, data_train, time_bins, C1=best_C1, device=device, verbose=verbose, **kwargs)
    return model
def plot_weights(weights, time_bins, feature_names):
    """Plot MTLR model parameters over time."""
    top_idxs = torch.argsort(weights.abs().sum(1), descending=True)[:10]
    fig, ax = plt.subplots()
    for i in top_idxs:
        ax.plot(np.pad(time_bins, (1, 0))[:-1], weights[i], label=feature_names[i])
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Weight")
    ax.axhline(0, c="k", linestyle="--")
    fig.legend()
    return ax
def compute_metric_at_times(metric, time_true, prob_pred, event_observed, score_times):
    """在给定时间点评估指标的辅助函数."""
    scores = []
    for time, pred in zip(score_times, prob_pred.T):
        target = time_true > time
        uncensored = target | event_observed.astype(bool)
        scores.append(metric(target[uncensored], pred[uncensored]))
    return scores
def brier_score_at_times(time_true, prob_pred, event_observed, score_times):
    scores = compute_metric_at_times(brier_score_loss,time_true,prob_pred,event_observed,score_times)
    return scores
def roc_auc_at_times(time_true, prob_pred, event_observed, score_times):
    scores = compute_metric_at_times(roc_auc_score,time_true,prob_pred, event_observed,score_times)
    return scores
def recall_at_times(time_true, prob_pred, event_observed, score_times):
    scores = compute_metric_at_times(recall_score,time_true,prob_pred, event_observed,score_times)
    return scores
path = './feature/'
data_train = pd.read_excel(os.path.join(path,'train.xlsx'))
data_test = pd.read_excel(os.path.join(path,'test.xlsx'))
data_val = pd.read_excel(os.path.join(path,'val.xlsx'))
eval_times = np.quantile(data_train.loc[data_train["event"] == 1, "time"], [.17, .48, .72, 0.84, .925]).astype(np.int)
metrics = []
data_train, mean_train, std_train = normalize(data_train, skip_cols=["time", "event"])
data_test, *_ = normalize(data_test, mean=mean_train, std=std_train, skip_cols=["time", "event"])
data_val, *_ = normalize(data_val, mean=mean_train, std=std_train, skip_cols=["time", "event"])
x_test = torch.tensor(data_test.drop(["time", "event"], axis=1).values, dtype=torch.float, device=device)
x_val = torch.tensor(data_val.drop(["time", "event"], axis=1).values, dtype=torch.float, device=device)
x_train = torch.tensor(data_train.drop(["time", "event"], axis=1).values, dtype=torch.float, device=device)
num_features = data_train.shape[1] - 2
time_bins = make_time_bins(data_train["time"].values, event=data_train["event"].values)
num_time_bins = len(time_bins)
'''fit Deep MTLR'''
deepmtlr = nn.Sequential(
    nn.Linear(num_features, 100),
    nn.ELU(),
    nn.Dropout(.4),
    nn.Linear(100, 32),
    nn.ELU(),
    nn.Dropout(.4),
    MTLR(in_features=32, num_time_bins=num_time_bins)
)
deepmtlr = train_mtlr_cv(deepmtlr, data_train, time_bins, num_epochs=20, lr=.01, batch_size=660, cv=5,weight_decay=1e-5, verbose=True, device=device)
torch.save(deepmtlr.state_dict(), os.path.join(path, 'model/', 'deep_mtlr.pth'))
with torch.no_grad():
    pred_deep = deepmtlr(torch.tensor(data_test.drop(["time", "event"], axis=1).values, dtype=torch.float))
    pred_survival = mtlr_survival_at_times(pred_deep, time_bins, eval_times)
    pred_risk = mtlr_risk(pred_deep).cpu().numpy()
    pred_deep_3 = deepmtlr(torch.tensor(data_train.drop(["time", "event"], axis=1).values, dtype=torch.float))
    pred_survival_3 = mtlr_survival_at_times(pred_deep_3, time_bins, eval_times)
    pred_risk_3 = mtlr_risk(pred_deep_3).cpu().numpy()
    pred_deep_val = deepmtlr(torch.tensor(data_val.drop(["time", "event"], axis=1).values, dtype=torch.float))
    pred_survival_val = mtlr_survival_at_times(pred_deep_val, time_bins, eval_times)
    pred_risk_val = mtlr_risk(pred_deep_val).cpu().numpy()
ci3 = concordance_index(data_test["time"], -pred_risk, event_observed=data_test["event"])
auc3 = roc_auc_at_times(data_test["time"], pred_survival, data_test["event"], eval_times)
fpr, tpr, threshold = roc_curve(data_test["event"], pred_risk)  ###计算真正率和假正率
roc_auc = auc(fpr, tpr)
ci3_val = concordance_index(data_val["time"], -pred_risk_val, event_observed=data_val["event"])
auc3_val = roc_auc_at_times(data_val["time"], pred_survival_val, data_val["event"], eval_times)
fpr, tpr, threshold = roc_curve(data_val["event"], pred_risk_val)  ###计算真正率和假正率
roc_auc_val = auc(fpr, tpr)
ci_3 = concordance_index(data_train["time"], -pred_risk_3, event_observed=data_train["event"])
auc_3 = roc_auc_at_times(data_train["time"], pred_survival_3, data_train["event"], eval_times)
fpr, tpr, threshold = roc_curve(data_train["event"], pred_risk_3)  ###计算真正率和假正率
roc_auc_3 = auc(fpr, tpr)
metrics.append({
    "model": "D-MTLR",
    #**{f"auc_{t}": auc3[i] for i, t in enumerate(eval_times)},
    **{f"auc": roc_auc},
    **{f"C-index": ci3},
    **{f"auc_val": roc_auc_val},
    **{f"C-index_val": ci3_val},
    #**{f"auc_train_{t}": auc_3[i] for i, t in enumerate(eval_times)},
    **{f"auc_train": roc_auc_3},
    **{f"C-index_train": ci_3}
})
'''Cox vs. Deep MTLR'''
a = pd.DataFrame(metrics).round(3)
print(a)
