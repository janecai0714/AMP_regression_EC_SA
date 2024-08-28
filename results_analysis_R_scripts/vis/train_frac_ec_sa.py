import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ec_train = pd.read_csv("/home/jianxiu/Documents/iscb/vis/ec_train_frac.csv")
sa_train = pd.read_csv("/home/jianxiu/Documents/iscb/vis/sa_train_frac.csv")
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Arial']
plt.plot(ec_train["coverage"], ec_train["mse"], color="dodgerblue", label="E. coli")
plt.scatter(ec_train["coverage"], ec_train["mse"], color="blue", s=15)
plt.errorbar(ec_train["coverage"], ec_train["mse"], yerr=ec_train["std"],
             capsize=2, elinewidth=0.5, markeredgewidth=2, color="deepskyblue")

plt.plot(sa_train["coverage"], sa_train["mse"], color="mediumseagreen", label="S. aureus")
plt.scatter(sa_train["coverage"], sa_train["mse"], color="green", s=15)
plt.errorbar(sa_train["coverage"], sa_train["mse"], yerr=sa_train["std"],
             capsize=2, elinewidth=0.5, markeredgewidth=2, color="mediumseagreen")
plt.legend()
plt.ylabel("MSE",  size=15) # , fontweight='bold'
plt.xlabel("Coverage of training data",  size=15) # , fontweight='bold'
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.ylim([0, 0.5])
plt.grid(False)
plt.savefig("/home/jianxiu/Documents/EC/pics/svg/train_frac.svg")
plt.close()


print("smart")
