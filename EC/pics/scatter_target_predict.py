import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
import torch


# draw target_predict scatter
ec_predit_path = "/home/jianxiu/Documents/EC/bert_finetune/target_predict_276353.csv"
ec_predit = pd.read_csv(ec_predit_path)
plt.rcParams['font.family'] = 'sans Serif'
plt.rcParams['font.serif'] = ['Arial']
fig = plt.figure()
ax = fig.add_subplot()
plt.scatter(ec_predit["target_value"], ec_predit["predict_value"], s=10, color="lightblue", edgecolors="black")
plt.ylabel("Predicted pMIC values (-log μM)",  size=15)
plt.xlabel("Experimental pMIC values (-log μM)",  size=15)
plt.xticks(fontsize=12) # , rotation=90)
plt.yticks(fontsize=12) #, rotation=90) color='w',
plt.xlim([-4, 2.5])
plt.ylim([-4, 2.5])
plt.grid(False)

x_points = [-4, 3]
y_points= [-4, 3]
plt.plot(x_points, y_points, linestyle='dashed', color='blue')
ax.set_aspect('equal', adjustable='box')
plt.savefig("/home/jianxiu/Documents/EC/pics/svg/target_predict/EC_target_predict.svg")
plt.close()

# draw target_predict scatter
sa_predit_path = "/home/jianxiu/Documents/EC/bert_finetune/target_predict_SA_21143.csv"
sa_predit = pd.read_csv(sa_predit_path)
plt.rcParams['font.family'] = 'sans Serif'
plt.rcParams['font.serif'] = ['Arial']
fig = plt.figure()
ax = fig.add_subplot()
plt.scatter(sa_predit["target_value"], sa_predit["predict_value"], s=10, color="lightgreen", edgecolors="black")
plt.ylabel("Predicted pMIC values (-log μM)",  size=15)
plt.xlabel("Experimental pMIC values (-log μM)",  size=15)
plt.xticks(fontsize=12) # , rotation=90)
plt.yticks(fontsize=12) #, rotation=90) color='w',
plt.xlim([-4, 2.5])
plt.ylim([-4, 2.5])
plt.grid(False)

x_points = [-4, 3]
y_points= [-4, 3]
plt.plot(x_points, y_points, linestyle='dashed', color='green')
ax.set_aspect('equal', adjustable='box')
plt.savefig("/home/jianxiu/Documents/EC/pics/svg/target_predict/SA_target_predict.svg")
plt.close()
print("smart")