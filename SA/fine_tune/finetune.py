import argparse

import torch
import os
from torch import nn
from transformers import BertTokenizer, get_linear_schedule_with_warmup
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, RandomSampler, TensorDataset
import torch.optim as optim
#import torch_optimizer as optim
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

from scipy.stats import pearsonr, kendalltau
from model_def import REG
import random

d_model = 1024
batch_size = 8
MAX_LEN = 77
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )

largest_MIC = 10000; def_bias = 4.0; def_scale = 1/6;
def descale(output, bias=def_bias, scale=def_scale):
    new_out = output / scale - bias
    return new_out
class Seq_Dataset(Dataset):
    def __init__(self, sequence, targets, tokenizer, max_len):
        self.sequence = sequence
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, item):
        sequence = str(self.sequence[item])
        target = self.targets[item]
        encoding = self.tokenizer.encode_plus(
            sequence,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
          'protein_sequence': sequence,
          'input_ids': encoding['input_ids'].flatten(),
          'attention_mask': encoding['attention_mask'].flatten(),
          'targets': torch.tensor(target, dtype=torch.float)
        }


def _get_train_data_loader(batch_size, train_dir):
    dataset = pd.read_csv(train_dir)
    train_data = Seq_Dataset(
        sequence=dataset.SEQUENCE_space.to_numpy(),
        targets=dataset.EC_pMIC.to_numpy(),
        tokenizer=tokenizer,
        max_len=MAX_LEN
  )

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return train_dataloader

def _get_test_data_loader(batch_size, test_dir):
    dataset = pd.read_csv(test_dir)
    test_data = Seq_Dataset(
        sequence=dataset.SEQUENCE_space.to_numpy(),
        targets=dataset.EC_pMIC.to_numpy(),
        tokenizer=tokenizer,
        max_len=MAX_LEN
  )

    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return test_dataloader




def freeze(model, frozen_layers):
    modules = [model.bert.encoder.layer[:frozen_layers]]
    for module in modules:
        for param in module.parameters():
            param.requires_grad = False

class ConcordanceCorCoeff(nn.Module):
    def __init__(self):
        super(ConcordanceCorCoeff, self).__init__()
        self.mean = torch.mean
        self.var = torch.var
        self.sum = torch.sum
        self.sqrt = torch.sqrt
        self.std = torch.std

    def forward(self, prediction, ground_truth):
        mean_gt = self.mean(ground_truth, 0)
        mean_pred = self.mean(prediction, 0)
        var_gt = self.var(ground_truth, 0)
        var_pred = self.var(prediction, 0)
        v_pred = prediction - mean_pred
        v_gt = ground_truth - mean_gt
        cor = self.sum(v_pred * v_gt) / (self.sqrt(self.sum(v_pred ** 2)) * self.sqrt(self.sum(v_gt ** 2)))
        sd_gt = self.std(ground_truth)
        sd_pred = self.std(prediction)
        numerator = 2 * cor * sd_gt * sd_pred
        denominator = var_gt + var_pred + (mean_gt - mean_pred) ** 2
        ccc = numerator / denominator
        return 1-ccc
def train(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    train_loader = _get_train_data_loader(args.batch_size, args.train_dir)
    test_loader = _get_train_data_loader(500, args.test_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # set the seed for generating random numbers


    model = REG()
    freeze(model, args.frozen_layers)
    model = model.to(device)

    optimizer = optim.AdamW(
        filter(lambda x: x.requires_grad is not False,
        model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay)


    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=0,
    #     num_training_steps=total_steps)

    #loss_fn = nn.HuberLoss().to(device)
    loss_fn = ConcordanceCorCoeff()
    #loss_fn = nn.MSELoss()
    mse_list = []
    r2_list = []
    pcc_list = []
    ktc_list = []

    test_mse_list = []
    test_r2_list = []
    test_pcc_list = []
    test_ktc_list = []
    min_mse = 0.5
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_predict_list = []
        train_target_list = []
        for batch in train_loader:
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['targets'].to(device)

            predict_MIC = model(b_input_ids, attention_mask=b_input_mask)
            loss = loss_fn(predict_MIC.view(-1), b_labels.float().view(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()
            optimizer.zero_grad()

            #scheduler.step()
            train_predict_list.append(predict_MIC.cpu().data.numpy())
            train_target_list.append(b_labels.cpu())
        train_predict_list = np.concatenate(train_predict_list).flatten()
        train_target_list = np.concatenate(train_target_list).flatten()

        mse_epoch = mean_squared_error(train_predict_list, train_target_list)
        r2_epoch = r2_score(train_predict_list, train_target_list)
        pcc_epoch = pearsonr(train_predict_list, train_target_list)[0]
        ktc_epoch = kendalltau(train_predict_list, train_target_list)[0]
        if mse_epoch < min_mse:
            min_mse = mse_epoch
            # torch.save(model.state_dict(), "prot_bert_finetune_amazon.pt")

        mse_list.append(mse_epoch)
        r2_list.append(r2_epoch)
        pcc_list.append(pcc_epoch)
        ktc_list.append(ktc_epoch)

        test_predict_list = []
        test_target_list = []
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                b_input_ids = batch['input_ids'].to(device)
                b_input_mask = batch['attention_mask'].to(device)
                b_labels = batch['targets'].to(device)
                predict_MIC = model(b_input_ids, attention_mask=b_input_mask)

                test_predict_list.append(predict_MIC.cpu().data.numpy())
                test_target_list.append(b_labels.cpu())
        test_predict_list = np.concatenate(test_predict_list).flatten()
        test_target_list = np.concatenate(test_target_list).flatten()
        test_mse = mean_squared_error(test_predict_list, test_target_list)
        test_r2 = r2_score(test_predict_list, test_target_list)
        test_pcc = pearsonr(test_predict_list, test_target_list)[0]
        test_ktc = kendalltau(test_predict_list, test_target_list)[0]
        test_mse_list.append(test_mse)
        test_r2_list.append(test_r2)
        test_pcc_list.append(test_pcc)
        test_ktc_list.append(test_ktc)
        print("epoch:", epoch, 'train_mse =', '{:.4f}'.format(mse_epoch), 'train_r2 =', '{:.4f}'.format(r2_epoch),
              "pcc: ", '{:.4f}'.format(pcc_epoch),
              'test_mse', '{:.4f}'.format(test_mse), 'test_r2', '{:.4f}'.format(test_r2),
              "test_pcc: ", '{:.4f}'.format(test_pcc))
        result_dict = {"seed": args.seed, "train_mse": mse_list, "pcc": r2_list, "test_mse": test_mse_list, "test_pcc": test_pcc_list,
                       "best_mse": min(test_mse_list), "best_pcc": max(test_pcc_list)}
        result_df = pd.DataFrame(result_dict)
        result_df.to_csv(args.result_dir)
    return min(test_mse_list), max(test_pcc_list)

if __name__ == "__main__":
    max_num_split = 5
    max_num_seed = 20
    for i in range(max_num_split):
        train_path = "/home/jianxiu/Documents/EC/data/" + str(i) + "/train-EC.csv"
        test_path = "/home/jianxiu/Documents/EC/data/" + str(i) + "/test-EC.csv"
        best_path = "/home/jianxiu/Documents/EC/data/" + str(i) + "/best_result-EC.csv"
        best_mse_list, best_pcc_list = [], []
        for j in range(max_num_seed):

            my_seed = random.randint(0, 1000000)
            result_path = "/home/jianxiu/Documents/EC/data/" + str(i) + "/result-EC-" + str(my_seed) + ".csv"

            parser = argparse.ArgumentParser()
            parser.add_argument("--seed", type=int, default=my_seed, metavar="S", help="random seed (default: 43)")
            parser.add_argument("--train_dir", type=str, default=train_path)
            parser.add_argument("--test_dir", type=str, default=test_path)
            parser.add_argument("--result_dir", type=str, default=result_path)
            parser.add_argument(
                "--batch-size", type=int, default=6, metavar="N", help="input batch size for training (default: 1)"
            )
            # parser.add_argument("--train_dir", type=str, default= "/home/jianxiu/Documents/EC/data/0/train-EC_ad.csv")
            # parser.add_argument("--test_dir", type=str, default="/home/jianxiu/Documents/EC/data/0/test-EC_ad.csv")
            parser.add_argument("--frozen_layers", type=int, default=0, metavar="NL",
                                help="number of frozen layers(default: 10)")
            parser.add_argument("--lr", type=float, default=1e-5, metavar="LR", help="learning rate (default: 0.3e-5)")
            parser.add_argument("--weight_decay", type=float, default=5e-3, metavar="M",
                                help="weight_decay (default: 0.01)")
            parser.add_argument("--epochs", type=int, default=100, metavar="N",
                                help="number of epochs to train (default: 100)")

            best_test_mse, best_test_pcc = train(parser.parse_args())
            best_mse_list.append(best_test_mse)
            best_pcc_list.append(best_test_pcc)

        best_result_dict = {"best_mse_repeat": min(best_mse_list), "best_pcc": max(best_pcc_list)}
        best_result_df = pd.DataFrame(best_result_dict)
        best_result_df.to_csv(best_path)
    print("smart")

