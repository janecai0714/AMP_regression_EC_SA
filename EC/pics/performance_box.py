import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import textwrap


perf = pd.read_csv("/home/jianxiu/Documents/EC/pics/perf_bert_ml.csv")
fig, ax = plt.subplots()
plt.rcParams['font.family'] = 'sans Serif'
plt.rcParams['font.serif'] = ['Arial']
hue_color={"E. coli": "lightblue", "S. aureus": "lightgreen"}
sns.boxplot(x = perf["models"], y = perf["mse"],
            hue = perf["species"], palette = hue_color, showfliers=0)
ax.set_xticklabels(["SVM+"+"\n"+"ProtBERT", "RF+"+"\n"+"ProtBERT", "DT+"+"\n"+"ProtBERT",
                    "KNN+"+"\n"+"ProtBERT", "Proposed" +"\n" +"model"])
plt.xlabel("Models", size=15) #, fontweight='bold'
plt.ylabel("MSE",  size=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig("/home/jianxiu/Documents/EC/pics/svg/perf_bert_ml/bert_ml_mse.svg", bbox_inches='tight')
plt.close()

# pcc
fig, ax = plt.subplots()
plt.rcParams['font.family'] = 'sans Serif'
plt.rcParams['font.serif'] = ['Arial']
hue_color={"E. coli": "lightblue", "S. aureus": "lightgreen"}
sns.boxplot(x = perf["models"], y = perf["pcc"],
            hue = perf["species"], palette = hue_color, showfliers=0)
ax.set_xticklabels(["SVM+"+"\n"+"ProtBERT", "RF+"+"\n"+"ProtBERT", "DT+"+"\n"+"ProtBERT",
                    "KNN+"+"\n"+"ProtBERT", "Proposed" +"\n" +"model"])
plt.xlabel("Models", size=15) #, fontweight='bold'
plt.ylabel("PCC",  size=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig("/home/jianxiu/Documents/EC/pics/svg/perf_bert_ml/bert_ml_pcc.svg", bbox_inches='tight')
plt.close()

fig, ax = plt.subplots()
plt.rcParams['font.family'] = 'sans Serif'
plt.rcParams['font.serif'] = ['Arial']
hue_color={"E. coli": "lightblue", "S. aureus": "lightgreen"}
sns.boxplot(x = perf["models"], y = perf["ktc"],
            hue = perf["species"], palette = hue_color, showfliers=0)
ax.set_xticklabels(["SVM+"+"\n"+"ProtBERT", "RF+"+"\n"+"ProtBERT", "DT+"+"\n"+"ProtBERT",
                    "KNN+"+"\n"+"ProtBERT", "Proposed" +"\n" +"model"])
plt.xlabel("Models", size=15) #, fontweight='bold'
plt.ylabel("KTC",  size=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig("/home/jianxiu/Documents/EC/pics/svg/perf_bert_ml/bert_ml_ktc.svg", bbox_inches='tight')
plt.close()

# machine learning
ml_df_perf = pd.read_csv("/home/jianxiu/Documents/EC/pics/perf_ml_dl.csv")
#ml_df_perf["M"]
fig, ax = plt.subplots()
plt.rcParams['font.family'] = 'sans Serif'
plt.rcParams['font.serif'] = ['Arial']
hue_color={"E. coli": "lightblue", "S. aureus": "lightgreen"}
sns.boxplot(x = ml_df_perf["models"], y = ml_df_perf["mse"],
            hue = ml_df_perf["species"], palette = hue_color, showfliers=0)
ax.set_xticklabels(["SVM", "RF", "DT", "KNN", "MBC", "MBC-"+"\n"+"Attention", "Proposed" +"\n" +"model"])
plt.xlabel("Models", size=15) #, fontweight='bold'
plt.ylabel("MSE",  size=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig("/home/jianxiu/Documents/EC/pics/svg/perf_ml_dl/ml_dl_mse.svg", bbox_inches='tight')
plt.close()

fig, ax = plt.subplots()
plt.rcParams['font.family'] = 'sans Serif'
plt.rcParams['font.serif'] = ['Arial']
hue_color={"E. coli": "lightblue", "S. aureus": "lightgreen"}
sns.boxplot(x = ml_df_perf["models"], y = ml_df_perf["pcc"],
            hue = ml_df_perf["species"], palette = hue_color, showfliers=0)
ax.set_xticklabels(["SVM", "RF", "DT", "KNN", "MBC", "MBC-"+"\n"+"Attention", "Proposed" +"\n" +"model"])
plt.xlabel("Models", size=15) #, fontweight='bold'
plt.ylabel("PCC",  size=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig("/home/jianxiu/Documents/EC/pics/svg/perf_ml_dl/ml_dl_pcc.svg", bbox_inches='tight')
plt.close()

fig, ax = plt.subplots()
plt.rcParams['font.family'] = 'sans Serif'
plt.rcParams['font.serif'] = ['Arial']
hue_color={"E. coli": "lightblue", "S. aureus": "lightgreen"}
sns.boxplot(x = ml_df_perf["models"], y = ml_df_perf["ktc"],
            hue = ml_df_perf["species"], palette = hue_color, showfliers=0)
ax.set_xticklabels(["SVM", "RF", "DT", "KNN", "MBC", "MBC-"+"\n"+"Attention", "Proposed" +"\n" +"model"])
plt.xlabel("Models", size=15) #, fontweight='bold'
plt.ylabel("KTC",  size=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig("/home/jianxiu/Documents/EC/pics/svg/perf_ml_dl/ml_dl_ktc.svg", bbox_inches='tight')
plt.close()

print("smart")

fig, ax = plt.subplots()
plt.rcParams['font.family'] = 'sans Serif'
plt.rcParams['font.serif'] = ['Arial']
hue_color={"E. coli": "lightblue", "S. aureus": "lightgreen"}
sns.boxplot(x = ml_df_perf["models"], y = ml_df_perf["ktc"],
            hue = ml_df_perf["species"], palette = hue_color, showfliers=0)
ax.set_xticklabels(["SVM", "RF", "DT", "KNN", "MBC", "MBC-"+"\n"+"Attention", "Proposed" +"\n" +"model"])
plt.xlabel("Models", size=15) #, fontweight='bold'
plt.ylabel("KTC",  size=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig("/home/jianxiu/Documents/EC/pics/svg/perf_ml_dl/ml_dl_pcc.svg", bbox_inches='tight')
plt.close()