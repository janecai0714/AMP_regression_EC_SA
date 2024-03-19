import torch
from torch.utils.data import DataLoader
from torch import autograd, nn

from transformers import BertModel, BertTokenizer
import re
import pandas as pd
from Bio import SeqIO
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.spatial.distance import pdist

import matplotlib.pyplot as plt
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )
model = BertModel.from_pretrained("Rostlab/prot_bert")
def gene_bert_fea(seq_data, tmp_fea_path):
    X = []
    print("Encoding training/testing sequences...")
    for i in range(len(seq_data["SEQUENCE_space"])):
        seq = re.sub(r"[UZOB]", "X", seq_data["SEQUENCE_space"][i])
        seq_encoded_input = tokenizer(seq, return_tensors='pt')
        seq_output = model(**seq_encoded_input)
        seq_fea = seq_output.pooler_output.detach().numpy()
        X.append(seq_fea)
    np.save(tmp_fea_path, X)
    return X

def gene_ad(seq_fea, ad_path):
    flat_fea = []
    for i in range(seq_fea.shape[0]):
        flat_fea.append(seq_fea[i, :, :].flatten())
    flat_fea = np.array(flat_fea)
    flat_fea_mean = np.mean(flat_fea, axis=0)
    centroid = np.reshape(flat_fea_mean, (1, flat_fea_mean.shape[0]))

    sim_score = []
    for i in range(seq_fea.shape[0]):
        sim_score.extend(pdist(np.vstack([seq_fea[i].flatten(), centroid]))/1000)

    ad_dict = {
        "ad_centroid": centroid,
        "ad_mean": np.mean(sim_score),
        "ad_std": np.std(sim_score),
        "ad_ratio": 1.0
    }
    np.save(ad_path, ad_dict)
    return ad_dict





if __name__ == "__main__":
    max_num_split = 5
    for i in range(0, 2):
        column = ["ID", "SEQUENCE", "SEQUENCE_space", "EC_pMIC_scale"]

        train_csv_path = "/home/jianxiu/Documents/EC/data/" + str(i) + "/train-EC.csv"
        train_csv_ad_path = "/home/jianxiu/Documents/EC/data/" + str(i) + "/ad/train-EC_ad.csv"
        train_fea_path = "/home/jianxiu/Documents/EC/data/" + str(i) + "/ad/train-EC_tmp_fea.npy"
        train_ad_path = "/home/jianxiu/Documents/EC/data/" + str(i) + "/ad/train-EC_ad.npy"
        train_csv_df = pd.read_csv(train_csv_path)

        test_csv_path = "/home/jianxiu/Documents/EC/data/" + str(i) + "/test-EC.csv"
        test_csv_ad_path = "/home/jianxiu/Documents/EC/data/" + str(i) + "/ad/test-EC_ad.csv"
        test_fea_path = "/home/jianxiu/Documents/EC/data/" + str(i) + "/ad/test-EC_tmp_fea.npy"
        test_csv_df = pd.read_csv(test_csv_path)

        x_fea_train = gene_bert_fea(train_csv_df, train_fea_path)
        x_fea_train_np = np.load(train_fea_path, allow_pickle=True)
        ad_dict_train = gene_ad(x_fea_train_np, train_ad_path)
        ad_param = np.load(train_ad_path, allow_pickle=True).item()
        sim_score_train = []
        for i in range(x_fea_train_np.shape[0]):
            sim_score_train.extend(pdist(np.vstack([x_fea_train_np[i].flatten(), ad_param['ad_centroid']])) / 1000)
        ad_filter = (sim_score_train <= ad_param['ad_mean'] + ad_param['ad_ratio'] * ad_param['ad_std'])

        x_fea_test = gene_bert_fea(test_csv_df, test_fea_path)
        x_fea_test_np = np.load(test_fea_path, allow_pickle=True)
        sim_score_test = []  # check if test in the application domain of train, no need to cal ad_dict of test.
        for i in range(x_fea_test_np.shape[0]):
            sim_score_test.extend(pdist(np.vstack([x_fea_test_np[i].flatten(), ad_param['ad_centroid']])) / 1000)
        ad_filter_test = (sim_score_test <= ad_param['ad_mean'] + ad_param['ad_ratio'] * ad_param['ad_std'])

        train_csv_ad_df = train_csv_df[ad_filter]
        train_csv_ad_df.to_csv(train_csv_ad_path)

        test_csv_ad_df = test_csv_df[ad_filter_test]
        test_csv_ad_df.to_csv(test_csv_ad_path)
print("smart")