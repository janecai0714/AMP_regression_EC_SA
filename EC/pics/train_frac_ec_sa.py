import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ec_train = pd.read_csv("/home/jianxiu/Documents/EC/pics/ec_train_frac.csv")
sa_train = pd.read_csv("/home/jianxiu/Documents/EC/pics/sa_train_frac.csv")
plt.rcParams['font.family'] = 'sans Serif'
plt.rcParams['font.serif'] = ['Arial']
plt.plot(ec_train["coverage"], ec_train["mse"], color="dodgerblue", label="E. coli", zorder=1)
plt.plot(sa_train["coverage"], sa_train["mse"], color="mediumseagreen", label="S. aureus", zorder=1)

plt.errorbar(ec_train["coverage"], ec_train["mse"], yerr=ec_train["mse_std"],
             capsize=2, elinewidth=0.5, markeredgewidth=1, color="deepskyblue", zorder=1)
plt.errorbar(sa_train["coverage"], sa_train["mse"], yerr=sa_train["mse_std"],
             capsize=2, elinewidth=0.5, markeredgewidth=1, color="mediumseagreen", zorder=1)

plt.scatter(ec_train["coverage"], ec_train["mse"], color="blue", s=15, zorder=2)
plt.scatter(sa_train["coverage"], sa_train["mse"], color="green", s=15, zorder=2)

plt.legend()
plt.ylabel("MSE",  size=15) # , fontweight='bold'
plt.xlabel("Proportion of train data (%)",  size=15) # , fontweight='bold'
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.ylim([0, 0.5])
plt.grid(False)
plt.savefig("/home/jianxiu/Documents/EC/pics/svg/train_frac/train_frac_mse.svg")
plt.close()

# pcc
plt.rcParams['font.family'] = 'sans Serif'
plt.rcParams['font.serif'] = ['Arial']
plt.plot(ec_train["coverage"], ec_train["pcc"], color="dodgerblue", label="E. coli", zorder=1)
plt.plot(sa_train["coverage"], sa_train["pcc"], color="mediumseagreen", label="S. aureus", zorder=1)

plt.errorbar(ec_train["coverage"], ec_train["pcc"], yerr=ec_train["pcc_std"],
             capsize=2, elinewidth=0.5, markeredgewidth=2, color="deepskyblue", zorder=1)
plt.errorbar(sa_train["coverage"], sa_train["pcc"], yerr=sa_train["pcc_std"],
             capsize=2, elinewidth=0.5, markeredgewidth=2, color="mediumseagreen", zorder=1)

plt.scatter(ec_train["coverage"], ec_train["pcc"], color="blue", s=15, zorder=2)
plt.scatter(sa_train["coverage"], sa_train["pcc"], color="green", s=15, zorder=2)

plt.legend()
plt.ylabel("PCC",  size=15) # , fontweight='bold'
plt.xlabel("Proportion of train data (%)",  size=15) # , fontweight='bold'
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.ylim([0, 1])
plt.grid(False)
plt.savefig("/home/jianxiu/Documents/EC/pics/svg/train_frac/train_frac_pcc.svg")
plt.close()

# ktc
plt.rcParams['font.family'] = 'sans Serif'
plt.rcParams['font.serif'] = ['Arial']
plt.plot(ec_train["coverage"], ec_train["ktc"], color="dodgerblue", label="E. coli", zorder=1)
plt.errorbar(ec_train["coverage"], ec_train["ktc"], yerr=ec_train["ktc_std"],
             capsize=2, elinewidth=0.5, markeredgewidth=2, color="deepskyblue", zorder=1)
plt.scatter(ec_train["coverage"], ec_train["ktc"], color="blue", s=15, zorder=2)


plt.plot(sa_train["coverage"], sa_train["ktc"], color="mediumseagreen", label="S. aureus", zorder=1)
plt.errorbar(sa_train["coverage"], sa_train["ktc"], yerr=sa_train["ktc_std"],
             capsize=2, elinewidth=0.5, markeredgewidth=2, color="mediumseagreen", zorder=1)
plt.scatter(sa_train["coverage"], sa_train["ktc"], color="green", s=15, zorder=2)

plt.legend()
plt.ylabel("KTC",  size=15) # , fontweight='bold'
plt.xlabel("Proportion of train data (%)",  size=15) # , fontweight='bold'
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.ylim([0, 1])
plt.grid(False)
plt.savefig("/home/jianxiu/Documents/EC/pics/svg/train_frac/train_frac_ktc.svg")
plt.close()

print("smart")
