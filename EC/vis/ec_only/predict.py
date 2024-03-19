from bertviz import head_view, model_view
from transformers import BertTokenizer, BertModel
from model_def import REG
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
from torch import nn
from scipy.stats import pearsonr, kendalltau
import argparse
from seq_dataloader import *
from seq_dataloader import _get_train_data_loader, _get_test_data_loader, freeze
import matplotlib.pyplot as plt
import math
from Bio.SeqUtils.ProtParam import ProteinAnalysis

def predict(model, data_path):
    batch_size = 500
    seq = pd.read_csv(data_path)
    test_loader = _get_test_data_loader(batch_size, data_path)
    test_predict_list, test_target_list = [],[]
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['targets'].to(device)
            predict_pMIC, _ = model(b_input_ids, attention_mask=b_input_mask)

            test_predict_list.extend(predict_pMIC.cpu().data.numpy())
            test_target_list.extend(b_labels.cpu())

    test_predict_list = [item for sublist in test_predict_list for item in sublist]
    seq["predicted_pMIC"] = test_predict_list
    test_mse = mean_squared_error(test_predict_list, test_target_list)
    test_r2 = r2_score(test_predict_list, test_target_list)
    test_pcc = pearsonr(test_predict_list, test_target_list)[0]
    test_ktc = kendalltau(test_predict_list, test_target_list)[0]

    print('test_mse', '{:.4f}'.format(test_mse), 'test_r2', '{:.4f}'.format(test_r2),
          "test_pcc: ", '{:.4f}'.format(test_pcc), "test_ktc: ", '{:.4f}'.format(test_ktc))
    seq.to_csv(data_path)

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "/home/jianxiu/Documents/EC/vis/kr12/prot_bert_finetune_reproduce.pkl"
    model = REG()
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    data_path = "/home/jianxiu/Documents/EC/vis/ec_only/ec_only.csv"
    predict(model, data_path)
    seq_data = pd.read_csv(data_path)
    train_pmic = seq_data["predicted_pMIC"]
    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Arial']
    p2_bins = np.arange(-4, 3, 0.5)
    plt.hist(train_pmic, color="lightblue", edgecolor="black")  # , weights=np.ones_like(train_pmic) / len(train_pmic)
    plt.xlabel("pMIC", size=15)  # , fontweight='bold'
    plt.ylabel("Frequency", size=15)
    plt.xticks(fontsize=12)  # , rotation=90)
    # ytick = np.arange(0, 1100, 100)
    # plt.yticks(ytick, fontsize=12)  # , rotation=90) color='w',
    plt.xlim([-4, 2.5])
    # plt.ylim([0, 1000])
    plt.savefig("/home/jianxiu/Documents/EC/vis/ec_only/ec_only.svg")
    plt.close()

