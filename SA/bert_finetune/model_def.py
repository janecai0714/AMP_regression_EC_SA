from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from transformers import BertForMaskedLM, BertTokenizer, pipeline
import torch
import torch.nn.functional as F
import torch.nn as nn

BertModel.from_pretrained("Rostlab/prot_bert")
class REG(nn.Module):
    def __init__(self):
        super(REG, self).__init__()
        self.bert = BertModel.from_pretrained("Rostlab/prot_bert")
        self.regressor= nn.Sequential(nn.LayerNorm(self.bert.config.hidden_size),
                                      #
                                      nn.Linear(self.bert.config.hidden_size, 512),
                                      nn.LeakyReLU(inplace=False),
                                      nn.Dropout(p=0.2),
                                      #
                                      nn.Linear(512, 128),
                                      nn.LeakyReLU(inplace=False),
                                      nn.Dropout(p=0.2),

                                      nn.Linear(128, 1))

    def forward(self, input_ids, attention_mask):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return self.regressor(output.pooler_output)