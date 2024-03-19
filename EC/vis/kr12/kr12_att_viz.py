import torch
from transformers import BertTokenizer, BertModel
from bertviz import head_view
import re
from IPython.display import display
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# model = BertModel.from_pretrained("Rostlab/prot_bert", output_attentions=True)
self_bert = torch.load("/home/jianxiu/Documents/EC/vis/kr12/self_prot_bert.pth")
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = self_bert.to(device)
model = self_bert.eval()


sequence = "K R I V Q R I K D F L R"
sequence = re.sub(r"[UZOB]", "X", sequence)

inputs = tokenizer.encode_plus(sequence, return_tensors='pt', add_special_tokens=True)
input_ids = inputs['input_ids']
attention = self_bert(input_ids.to(device))[-1]
input_id_list = input_ids[0].tolist() # Batch index 0
tokens = tokenizer.convert_ids_to_tokens(input_id_list)

# attention_layer29 = attention[29].detach().cpu().numpy()
# attention_layer29 = attention_layer29.squeeze(axis=0)
# attention_layer29_head0 = attention_layer29[0]
# df = pd.DataFrame(attention_layer29_head0, columns = tokens)

def sum_token_att_layer_head(layer_num, head_num, token_pos):
    att_layerX = attention[layer_num].detach().cpu().numpy()
    att_layerX = att_layerX.squeeze(axis=0)
    att_layerX_headY = att_layerX[head_num]
    att_df = pd.DataFrame(att_layerX_headY)
    return att_df[token_pos].sum()

def sum_token_att_layer(layer_num, token_pos):
    att_all_head_list = []
    for head_index in range(16):
        att_one_head = sum_token_att_layer_head(layer_num, head_index, token_pos)
        att_all_head_list.append(att_one_head)
    return sum(att_all_head_list)

def sum_token_att(token_pos):
    att_all_layer_list = []
    for layer_index in range(30):
        att_one_layer = sum_token_att_layer(layer_index, token_pos)
        att_all_layer_list.append(att_one_layer)
    return sum(att_all_layer_list)

# attention for each token of a sequence on one layer
def sum_seq_att_layer(tokens, layer_num):
    seq_att_layer_list = []
    for token_index in range(len(tokens)):
        token_att = sum_token_att_layer(layer_num, token_index)
        seq_att_layer_list.append(token_att)
    return seq_att_layer_list

def sum_seq_att(tokens):
    seq_att_list = []
    for token_index in range(len(tokens)):
        token_att = sum_token_att(token_index)
        seq_att_list.append(token_att)
    return seq_att_list

# sum up attention for whole seqs for all layers all heads
# seq_att = sum_seq_att(tokens)
# seq_att_dict = {'token': tokens, 'att_all_layers': seq_att}
# seq_att_df = pd.DataFrame(seq_att_dict)
# seq_att_df_filtered = seq_att_df[(seq_att_df['token'] != '[CLS]') & (seq_att_df['token'] != '[SEP]')]
#
# # sum up attention for whole seqs for layer29's all heads
# seq_att_layer29 = sum_seq_att_layer(tokens, 29)
# seq_att_layer29_dict = {'token': tokens, 'att_layer29': seq_att_layer29}
# seq_att_layer29_df = pd.DataFrame(seq_att_layer29_dict )
# seq_att_layer29_df_filtered = seq_att_layer29_df[(seq_att_layer29_df['token'] != '[CLS]') & (seq_att_layer29_df['token'] != '[SEP]')]
#
# att = pd.read_csv("att.csv")
# plt.rcParams['font.family'] = 'DeJavu Serif'
# plt.rcParams['font.serif'] = ['Times New Roman']
# plt.scatter(att["token"], att["att_all_layers_ave"])
# plt.plot(att["token"], att["att_all_layers_ave"], color="skyblue")
# #plt.ylim([-5.0, 5.0])
# plt.ylabel("Average attention of all layers",  size=12)
# plt.xlabel("Original token", size=12)
# plt.savefig("att_all_scatter.png")
# plt.close()
#
att = pd.read_csv("att.csv")
plt.rcParams['font.family'] = 'sans Serif'
plt.rcParams['font.serif'] = ['Arial']
plt.scatter(att["token"], att["att_layer29_ave"], color="deeppink")
plt.plot(att["token"], att["att_layer29_ave"], color="pink")
plt.ylabel("Attention scores of the 29th layer",  size=15)
plt.ylim([0.0, 2.0])
plt.xlabel("Tokens", size=15)
plt.xticks(size=12)
plt.yticks(size=12)
plt.savefig("/home/jianxiu/Documents/EC/pics/svg/kr12/kr12_att_29_scatter.svg")
plt.close()

# attention for whole seqs for layers29 without sum up
tok_pos = []
for i in range(1, 13):
    tok_pos.append(tokens[i] + str(i))
att_layer29_f = []
for head_index in range(16):
    att_one_head = []
    for pos_index in range(1, 13):
        att = sum_token_att_layer_head(29, head_index, pos_index)
        att_one_head.append(att)
    att_layer29_f.append(att_one_head)
att_layer29_f = pd.DataFrame(att_layer29_f, columns = [tok_pos])
plt.rcParams['font.family'] = 'sans Serif'
plt.rcParams['font.serif'] = ['Arial']
colormap = sns.color_palette("coolwarm", 12)
heat_29_f = sns.heatmap(att_layer29_f, cmap = colormap, vmin=0, vmax=6)
heat_29_f.set_xlabel("Tokens", fontsize=15)
heat_29_f.set_ylabel("Heads of the 29th layer", fontsize=15)
fig_29 = heat_29_f.get_figure()
fig_29.savefig("/home/jianxiu/Documents/EC/pics/svg/kr12/kr12_heat_29.svg")
plt.close()


att_f = []
for layer_index in range(30):
    att_one_layer = []
    for pos_index in range(1, 13):
        att = sum_token_att_layer(layer_index, pos_index)
        att_one_layer.append(att)
    att_f.append(att_one_layer)
att_f = pd.DataFrame(att_f, columns = [tokens_f])
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
colormap = sns.color_palette("coolwarm", 12)
att_f_heatmap = sns.heatmap(att_f, cmap = colormap)
att_f_heatmap.set_xlabel("Original tokens", fontsize=12)
att_f_heatmap.set_ylabel("Layers", fontsize=12)
att_f_fig = att_f_heatmap.get_figure()
att_f_fig.savefig("att_ori")
plt.close()
print("smart")