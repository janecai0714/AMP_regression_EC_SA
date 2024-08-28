import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import torch

path = "D:/EC/EC.csv"
# draw length density
ec = pd.read_csv(path)
seqs = ec["SEQUENCE"]
seqs_len = seqs.str.len()
bins = np.arange(5, 65, 5)
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
seqs_len.hist(color="lightblue", edgecolor="black", bins=bins)
ytick = np.arange(0, 1100, 100)
plt.ylabel("Frequency",  size=12) #, fontweight='bold'
plt.xlabel("Length", size=12) # , fontweight='bold'
plt.xticks(fontsize=10) #, rotation=90)
plt.yticks(ytick, fontsize=10) #, rotation=90)
# plt.title("$\it{Escherichia}$" + " " + "$\it{coli}$", size=12, fontweight='bold')
plt.xlim([4, 61])
plt.ylim([0, 1000])
plt.grid(False)
plt.savefig("D:/EC/pics/EC_len_hist.png", dpi=100, format="png")
plt.close()

# draw pMIC density
ec = pd.read_csv(path)
pmic = ec["EC_pMIC"]
bins = np.arange(-4, 3, 0.5)
pmic.hist(color="lightblue", edgecolor="black", bins=bins) # aquamarine
# plt.suptitle(title, fontsize=font + 6, fontweight='bold')
plt.ylabel("Frequency",  size=12)#, fontweight='bold'
plt.xlabel("pMIC", size=12)  #, fontweight='bold'
plt.xticks(fontsize=10) # , rotation=90)
plt.yticks(ytick, fontsize=10) #, rotation=90) color='w',
# plt.yticks([])
plt.xlim([-4, 2.5])
plt.ylim([0, 1000])
plt.grid(False)
plt.savefig("D:/EC/pics/EC_pMIC_hist.png", dpi=100, format="png")
plt.close()

# draw target_predict scatter
ec_predit_path = "D:/EC/target_predict_ec_276353.csv"
ec_predit = pd.read_csv(ec_predit_path)
plt.scatter(ec_predit["target_value"], ec_predit["predict_value"], s=10, color="lightblue", edgecolors="black")
plt.ylabel("Predicted pMIC values (-log μM)",  size=12) # , fontweight='bold'
plt.xlabel("Experimental pMIC values (-log μM)",  size=12) # , fontweight='bold'
# plt.xticks(fontsize=10) # , rotation=90)
# plt.yticks(ytick, fontsize=10) #, rotation=90) color='w',
plt.xlim([-4, 2.5])
plt.ylim([-4, 2.5])
plt.grid(False)

x_points = [-4, 3]
y_points= [-4, 3]
plt.plot(x_points, y_points, linestyle='dashed', color="blue")
plt.savefig("D:/EC/pics/EC_target_predict.png", dpi=100, format="png")
plt.close()
print("smart")

# draw train_frac
ec_train_path = "D:/EC/train_frac_result_ec.csv"
ec_train = pd.read_csv(ec_train_path)
plt.scatter(ec_train["frac"], ec_train["mse"], s=10, color="lightblue", edgecolors="black")
plt.plot(ec_train["frac"], ec_train["mse"])
plt.ylabel("MSE)",  size=12) # , fontweight='bold'
plt.xlabel("Coverage of training data",  size=12) # , fontweight='bold'
# plt.xlim([-4, 2.5])
# plt.ylim([-4, 2.5])
plt.grid(False)
plt.savefig("D:/EC/pics/EC_train_frac.png", dpi=100, format="png")
plt.close()
print("smart")