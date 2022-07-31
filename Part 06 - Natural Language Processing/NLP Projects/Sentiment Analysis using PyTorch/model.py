import config
import transformers
import torch.nn as nn


class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 512)
        self.out = nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        _, o2 = self.bert(
            ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False
        )
        x = self.fc1(o2)
        x = self.relu(x)
        x = self.dropout(o2)
        output = self.out(x)
        return output
