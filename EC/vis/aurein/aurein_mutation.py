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

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "/home/jianxiu/Documents/EC/vis/aurein/prot_bert_finetune_reproduce.pkl"
model = REG()
model.load_state_dict(torch.load(model_path))

# sequence = "G L F D I I K K I A E S F"
# inputs = tokenizer.encode_plus(sequence_mutated, return_tensors='pt', add_special_tokens=True)
# pred, _ = model(inputs['input_ids'], attention_mask = inputs['attention_mask'])
# mass = ProteinAnalysis("G L F D I I K K I A E S F").molecular_weight()
# pred_mic_ug = pow(10, pred) *
data_path = "/home/jianxiu/Documents/EC/vis/aurein/aurein_mutation.csv"
result_path = "/home/jianxiu/Documents/EC/vis/aurein/aurein_mutation_result.csv"
aurein = pd.read_csv(data_path)
# aurein["EC_MIC"] = aurein["ec(μg/mL)"]*1000/aurein["mass"]
# aurein["EC_pMIC"] = -np.log10(aurein["EC_MIC"])

# test_loader = _get_test_data_loader(500, data_path)
# test_predict_list = []
# test_target_list = []
# model.eval()
# with torch.no_grad():
#     for batch in test_loader:
#         b_input_ids = batch['input_ids']
#         b_input_mask = batch['attention_mask']
#         b_labels = batch['targets']
#         predict_MIC,_ = model(b_input_ids, attention_mask=b_input_mask)
#
#         test_predict_list.extend(predict_MIC.data.numpy())
#         test_target_list.extend(b_labels.data.numpy())
#
# test_predict_list = [item for sublist in test_predict_list for item in sublist]
# test_mse = mean_squared_error(test_predict_list, test_target_list)
# test_r2 = r2_score(test_predict_list, test_target_list)
# test_pcc = pearsonr(test_predict_list, test_target_list)[0]
# test_ktc = kendalltau(test_predict_list, test_target_list)[0]
#
# print('test_mse', '{:.4f}'.format(test_mse), 'test_r2', '{:.4f}'.format(test_r2),
#           "test_pcc: ", '{:.4f}'.format(test_pcc))
#
# result_dict = {"test_mse":test_mse, "test_pcc":test_pcc, "test_r2":test_r2}
# result_df = pd.DataFrame(result_dict)
# result_df.to_csv(result_path)

# aurein = pd.read_csv(data_path)
# aurein["predicted_pMIC"] = test_predict_list
# aurein["predicted_MIC"] = 10**(-aurein["predicted_pMIC"])

# set position of bar on x axis
residual = pd.read_csv("residual.csv")
# bar_width = 0.25
# br1 = np.arange(len(residue["ori_token"]))
# br2 = [x + bar_width for x in br1]
# # makre bar plot
# plt.bar(br1, residue["EC_pMIC_residues"], width = bar_width, color = "paleturquoise",
#         label = "Experimental pMIC value residues (-log μM)")
# plt.bar(br2, residue["predicted_pMIC_residues"], width = bar_width, color = "pink",
#         label = "Predicted pMIC value residues (-log μM)")
# plt.xticks([r + bar_width for r in range(len(residue["ori_token"]))],
#         residue["ori_token"])
# plt.ylim([-1.0, 1.0])
#
# plt.ylabel("Residues",  size=12, fontweight='bold')
# plt.xlabel("Original tokens", size=12, fontweight='bold')
# plt.legend()
# plt.show()

# log2_fold_change = []
# ori_MIC = aurein["EC_MIC"][12]
# for indx, value in aurein["EC_MIC"].items():
#     log2_fold_change.append(math.log2(value) - math.log2(ori_MIC))
# aurein["log2_fold_change"] = log2_fold_change

# pre_log2_fold_change = []
# ori_pre_MIC = aurein["predicted_MIC"][12]
# for indx, value in aurein["predicted_MIC"].items():
#     pre_log2_fold_change.append(math.log2(value) - math.log2(ori_pre_MIC))
# aurein["pre_log2_fold_change"] = pre_log2_fold_change


# my_color = ["deepskyblue", "pink"]
# residual.plot(x = 'ori_token', y = ["pMIC_log2", "pre_pMIC_log2"], kind="bar", color=my_color)
# plt.ylim([-2.0, 1.5])
# plt.ylabel("Log2 fold change of pMIC",  size=12)
# plt.xlabel("Original token", size=12)
# plt.legend(["Experimental pMIC log2 fold change (-log μM)", "Predicted pMIC log2 fold change (-log μM)"])
# plt.savefig("pMIC_log2_fold_change_barplot.png")
# plt.close()
print("smart")

change = pd.read_csv("aurein_ala_ec_log2.csv")
plt.rcParams['font.family'] = 'sans Serif'
plt.rcParams['font.serif'] = ['Arial']
my_color = ["deepskyblue", "tomato"]
change.plot(x = 'ori_token', y = ["log2_fold_change", "pre_log2_fold_change"], kind="bar", color=my_color)
plt.ylim([-5.0, 5.0])
plt.ylabel("MIC log2 fold change (μM)",  size=15)
plt.xlabel("Aurein 1.2 alanine variants", size=15)
plt.xticks(fontsize=10, rotation=0)
plt.yticks(fontsize=10)
plt.legend(["Experimental MIC log2 fold change (μM)", "Predicted MIC log2 fold change (μM)"],prop={'size':12})
plt.savefig("/home/jianxiu/Documents/EC/pics/svg/aurein/aurein_ala_ec_log2.svg")
plt.close()
