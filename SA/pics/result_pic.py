import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import torch

# -3.9995, 2.44775
# -4, 2.5

path = "/home/jianxiu/Documents/SA/data/SA.csv"
# draw length density
ec = pd.read_csv(path)
seqs = ec["SEQUENCE"]
seqs_len = seqs.str.len()
bins = np.arange(5, 65, 5)
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
seqs_len.hist(color="aquamarine", edgecolor="black", bins=bins)
ytick = np.arange(0, 1100, 100)
plt.ylabel("Frequency",  size=12, fontweight='bold')
plt.xlabel("Length", size=12, fontweight='bold')
plt.xticks(fontsize=10) #, rotation=90)
plt.yticks(ytick, fontsize=10) #, rotation=90)
# plt.title("$\it{Escherichia}$" + " " + "$\it{coli}$", size=12, fontweight='bold')
plt.xlim([4, 61])
plt.ylim([0, 1000])
plt.grid(False)
plt.savefig("/home/jianxiu/Documents/SA/pics/SA_len_hist.png", dpi=100, format="png")
plt.close()
# draw pMIC density
ec = pd.read_csv(path)
pmic = ec["SA_pMIC"]
bins = np.arange(-4, 3, 0.5)

pmic.hist(color="aquamarine", edgecolor="black", bins=bins) # aquamarine
# plt.suptitle(title, fontsize=font + 6, fontweight='bold')
plt.ylabel("Frequency",  size=12, fontweight='bold')
plt.xlabel("pMIC", size=12, fontweight='bold')
plt.xticks(fontsize=10) # , rotation=90)
plt.yticks(ytick, fontsize=10) #, rotation=90) color='w',
# plt.yticks([])
plt.xlim([-4, 2.5])
plt.ylim([0, 1000])
plt.grid(False)
plt.savefig("/home/jianxiu/Documents/SA/pics/SA_pMIC_hist.png", dpi=100, format="png")
plt.close()

# draw target_predict scatter
ec_predit_path = "/home/jianxiu/Documents/SA/mse_loss_func/target_predict.csv"
ec_predit = pd.read_csv(ec_predit_path)
plt.scatter(ec_predit["target_value"], ec_predit["predict_value"], s=10, color="aquamarine", edgecolors="black")
plt.ylabel("Predicted pMIC values (-log μM/L)",  size=12, fontweight='bold')
plt.xlabel("Experimental pMIC values (-log μM/L)",  size=12, fontweight='bold')
# plt.xticks(fontsize=10) # , rotation=90)
# plt.yticks(ytick, fontsize=10) #, rotation=90) color='w',
plt.xlim([-4, 2.5])
plt.ylim([-4, 2.5])
#plt.grid(False)

x_points = [-4, 2.5]
y_points= [-4, 2.5]
plt.plot(x_points, y_points, linestyle='dashed')
plt.savefig("/home/jianxiu/Documents/SA/pics/SA_target_predict.png", dpi=100, format="png")
plt.close()
print("smart")