from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
from transformers import BertTokenizer, get_linear_schedule_with_warmup

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


def _get_train_data_loader(batch_size, train_dir, train_frac):
    dataset = pd.read_csv(train_dir)
    dataset = dataset.sample(frac = train_frac)
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