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

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "/home/jianxiu/Documents/EC/mse_loss_func/prot_bert_finetune_amazon.pkl"
model = REG()
model.load_state_dict(torch.load(model_path))
#model = model.to(device)
test_loader = torch.load("/home/jianxiu/Documents/EC/mse_loss_func/test_loader.pkl")

test_predict_list = []
test_target_list = []
model.eval()
with torch.no_grad():
    for batch in test_loader:
        b_input_ids = batch['input_ids']
        b_input_mask = batch['attention_mask']
        b_labels = batch['targets']
        predict_MIC,_ = model(b_input_ids, attention_mask=b_input_mask)

        test_predict_list.extend(predict_MIC.data.numpy())
        test_target_list.extend(b_labels.data.numpy())

test_predict_list = [item for sublist in test_predict_list for item in sublist]
test_mse = mean_squared_error(test_predict_list, test_target_list)
test_r2 = r2_score(test_predict_list, test_target_list)
test_pcc = pearsonr(test_predict_list, test_target_list)[0]
test_ktc = kendalltau(test_predict_list, test_target_list)[0]

print('test_mse', '{:.4f}'.format(test_mse), 'test_r2', '{:.4f}'.format(test_r2),
          "test_pcc: ", '{:.4f}'.format(test_pcc))

# sequence = 'sequence = "C G K K P G G W K C K L"'
# inputs = tokenizer.encode_plus(sequence, return_tensors='pt', add_special_tokens=True)
# pred, _ = model(inputs['input_ids'], attention_mask = inputs['attention_mask']) # preds = -1.5203

# sequence_mutated = "C G K K W G W W K C K L"
# inputs_mutated = tokenizer.encode_plus(sequence_mutated, return_tensors='pt', add_special_tokens=True)
# pred_mutated, _ = model(inputs_mutated['input_ids'], attention_mask = inputs_mutated['attention_mask'])
print("smart")