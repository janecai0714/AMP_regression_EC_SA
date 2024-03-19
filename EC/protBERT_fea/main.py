import torch
from torch.utils.data import DataLoader
from torch import autograd, nn

from transformers import BertModel, BertTokenizer
import re
import pandas as pd
from Bio import SeqIO
import numpy as np
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, kendalltau
from model import regNet, amp_dataset
import matplotlib.pyplot as plt

def gene_bert_fea(csv_path, tokenizer, train_fea_path):
    column = ["ID", "SEQUENCE_space", "EC_pMIC"]
    seq_data = pd.read_csv(csv_path, usecols=column)

    X = []
    print("Encoding training/testing sequences...")
    for i in range(len(seq_data["SEQUENCE_space"])):
        seq = re.sub(r"[UZOB]", "X", seq_data["SEQUENCE_space"][i])
        seq_encoded_input = tokenizer(seq, return_tensors='pt')
        seq_output = model(**seq_encoded_input)
        seq_fea = seq_output.pooler_output.detach().numpy()
        X.append(seq_fea)
    np.save(train_fea_path, X)
    return X

def ml_train_test(x_train, y_train, x_test, y_test, regressor_name):
    if regressor_name == "SVM":
        regressor = SVR()
    elif regressor_name == "RF":
        regressor = RandomForestRegressor()
    elif regressor_name == "DT":
        regressor = DecisionTreeRegressor()
    elif regressor_name == "KNN":
        regressor = KNeighborsRegressor()
    regressor.fit(x_train, y_train)
    preds = regressor.predict(x_test)
    reg_mse = mean_squared_error(y_test, preds)
    reg_pcc = pearsonr(y_test, preds)[0]
    reg_ktc = kendalltau(y_test, preds)[0]
    print("smart")
    return reg_mse, reg_pcc, reg_ktc

if  __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    model = BertModel.from_pretrained("Rostlab/prot_bert")
    max_num_split = 5
    regressor_name_list = ["SVM", "RF", "DT", "KNN"]
    data_list, regressor_list, mse_list, pcc_list, ktc_list = [], [], [], [], []
    for i in range(max_num_split):
        train_csv = "/home/jianxiu/Documents/EC/data/" + str(i) + "/train-EC.csv"
        train_fasta = "/home/jianxiu/Documents/EC/data/" + str(i) + "/train-EC.fasta"
        test_csv = "/home/jianxiu/Documents/EC/data/" + str(i) + "/test-EC.csv"
        test_fasta = "/home/jianxiu/Documents/EC/data/" + str(i) + "/test-EC.fasta"

        train_df = pd.read_csv(train_csv)
        train_fea_path = "/home/jianxiu/Documents/EC/data/" + str(i) + "/train-EC_bert.npy"
        train_ft = gene_bert_fea(train_csv, tokenizer, train_fea_path)
        train_ft = np.load(train_fea_path, allow_pickle=True)
        train_ft = np.squeeze(train_ft, axis=1)
        y_train = train_df["EC_pMIC"].values

        test_df = pd.read_csv(test_csv)
        y_test = test_df["EC_pMIC"].values
        test_fea_path = "/home/jianxiu/Documents/EC/data/" + str(i) + "/test-EC_bert.npy"
        test_ft = gene_bert_fea(test_csv, tokenizer, test_fea_path)
        test_ft = np.load(test_fea_path, allow_pickle=True)
        test_ft = np.squeeze(test_ft, axis=1)
        y_test = test_df["EC_pMIC"].values
        for regressor_name in regressor_name_list:
            reg_mse, reg_pcc, reg_ktc = ml_train_test(train_ft, y_train, test_ft, y_test, regressor_name)
            data_list.append(str(i))
            regressor_list.append(regressor_name)
            mse_list.append(reg_mse)
            pcc_list.append(reg_pcc)
            ktc_list.append(reg_ktc)
        print("smart")
    ml_path = "/home/jianxiu/Documents/EC/bert_fea/bert_ml.csv"
    ml_result_dict = {'data': data_list, "regressor": regressor_list,
                      "mse": mse_list, "pcc": pcc_list, "ktc": ktc_list}
    ml_result_df = pd.DataFrame(ml_result_dict)
    ml_result_df.to_csv(ml_path)
    print("smart")
