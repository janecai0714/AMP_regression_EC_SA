import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from collections import Counter
import matplotlib.patches as patches

train_path = "/home/jianxiu/Documents/SA/data/0/train-SA.csv"
test_path  = "/home/jianxiu/Documents/SA/data/0/test-SA.csv"

train_color='lightgreen'
test_color = 'green'
# train draw length density
train = pd.read_csv(train_path)
train_seqs = train["SEQUENCE"]
train_seqs_len = train_seqs.str.len()
plt.rcParams['font.family'] = 'sans Serif'
plt.rcParams['font.serif'] = ['Arial']
p1_bins = np.arange(0, 65, 5)
plt.hist(train_seqs_len, color=train_color, edgecolor="black", bins=p1_bins)
xtick = np.arange(0, 65)
plt.xlabel("Length", size=15)
plt.ylabel("Frequency",  size=15)
plt.xticks(fontsize=12)
ytick = np.arange(0, 1100, 100)
plt.yticks(ytick, fontsize=12)
plt.xlim([0, 65])
plt.ylim([0, 1000])

train_patch = patches.Patch(color=train_color, label='S. aureus train set')
plt.rcParams["legend.fontsize"] = 15
plt.legend(handles=[train_patch])
plt.savefig("/home/jianxiu/Documents/EC/pics/svg/data_ana/SA_train_len_hist.svg")
plt.close()

train_pmic = train["SA_pMIC"]
plt.rcParams['font.family'] = 'sans Serif'
plt.rcParams['font.serif'] = ['Arial']
p2_bins = np.arange(-4, 3, 0.5)
plt.hist(train_pmic, color=train_color, edgecolor="black", bins=p2_bins) # aquamarine
plt.xlabel("pMIC", size=15) #, fontweight='bold'
plt.ylabel("Frequency",  size=15)
plt.xticks(fontsize=12) # , rotation=90)
ytick = np.arange(0, 1100, 100)
plt.yticks(ytick, fontsize=12) #, rotation=90) color='w',
plt.xlim([-4, 2.5])
plt.ylim([0, 1000])
train_patch = patches.Patch(color=train_color, label='S. aureus train set')
plt.rcParams["legend.fontsize"] = 15
plt.legend(handles=[train_patch])
plt.savefig("/home/jianxiu/Documents/EC/pics/svg/data_ana/SA_train_pMIC_hist.svg")
plt.close()

train_amino_acids =" ".join(train["SEQUENCE_space"]).split()
train_aa = pd.Series(train_amino_acids).value_counts(sort=False).reset_index()
train_aa = train_aa.rename(columns={"index" : "aa", 0 : "count"})
train_aa = train_aa.sort_values("aa")
plt.rcParams['font.family'] = 'sans Serif'
plt.rcParams['font.serif'] = ['Arial']
plt.bar(train_aa["aa"], train_aa["count"] , color=train_color, edgecolor="black")
plt.xlabel("Amino acids", size=15) #, fontweight='bold'
plt.ylabel("Frequency",  size=15)
plt.xticks(fontsize=10) # , rotation=90)
ytick = np.arange(0, 16000, 1000)
plt.yticks(ytick, fontsize=10)
# plt.ylim([0, 1500])
train_patch = patches.Patch(color=train_color, label='S. aureus train set')
plt.rcParams["legend.fontsize"] = 15
plt.legend(handles=[train_patch])
plt.savefig("/home/jianxiu/Documents/EC/pics/svg/data_ana/SA_train_aa_hist.svg")
plt.close()

# test set
test = pd.read_csv(test_path)
test_seqs = test["SEQUENCE"]
test_seqs_len = test_seqs.str.len()
plt.hist(test_seqs_len, color=test_color, edgecolor="black", bins=p1_bins)
plt.rcParams['font.family'] = 'sans Serif'
plt.rcParams['font.serif'] = ['Arial']
xtick = np.arange(0, 65)
plt.xlabel("Length", size=15)
plt.ylabel("Frequency",  size=15)
plt.xticks(fontsize=12)
ytick = np.arange(0, 220, 20)
plt.yticks(ytick, fontsize=12)
plt.xlim([0, 65])
plt.ylim([0, 200])
test_patch = patches.Patch(color=test_color, label='S. aureus test set')
plt.rcParams["legend.fontsize"] = 15
plt.legend(handles=[test_patch])
plt.savefig("/home/jianxiu/Documents/EC/pics/svg/data_ana/SA_test_len_hist.svg")
plt.close()

test_pmic = test["SA_pMIC"]
plt.rcParams['font.family'] = 'sans Serif'
plt.rcParams['font.serif'] = ['Arial']
p2_bins = np.arange(-4, 3, 0.5)
plt.hist(test_pmic, color=test_color, edgecolor="black", bins=p2_bins) # aquamarine
plt.xlabel("pMIC", size=15) #, fontweight='bold'
plt.ylabel("Frequency",  size=15)
plt.xticks(fontsize=12) # , rotation=90)
ytick = np.arange(0, 220, 20)
plt.yticks(ytick, fontsize=12) #, rotation=90) color='w',
plt.xlim([-4, 2.5])
plt.ylim([0, 200])
test_patch = patches.Patch(color=test_color, label='S. aureus test set')
plt.rcParams["legend.fontsize"] = 15
plt.legend(handles=[test_patch])
plt.savefig("/home/jianxiu/Documents/EC/pics/svg/data_ana/SA_test_pMIC_hist.svg")
plt.close()

test_amino_acids =" ".join(test["SEQUENCE_space"]).split()
test_aa = pd.Series(test_amino_acids).value_counts(sort=False).reset_index()
test_aa = test_aa.rename(columns={"index" : "aa", 0 : "count"})
test_aa = test_aa.sort_values("aa")
plt.rcParams['font.family'] = 'sans Serif'
plt.rcParams['font.serif'] = ['Arial']
plt.bar(test_aa["aa"], test_aa["count"] , color=test_color, edgecolor="black")
plt.xlabel("Amino acids", size=15) #, fontweight='bold'
plt.ylabel("Frequency",  size=15)
plt.xticks(fontsize=10) # , rotation=90)
ytick = np.arange(0, 1600, 100)
plt.yticks(ytick, fontsize=10)
#plt.ylim([0, 1500])
test_patch = patches.Patch(color=test_color, label='S. aureus test set')
plt.rcParams["legend.fontsize"] = 15
plt.legend(handles=[test_patch])
plt.savefig("/home/jianxiu/Documents/EC/pics/svg/data_ana/SA_test_aa_hist.svg")
plt.close()


print("smart")
