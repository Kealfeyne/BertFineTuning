import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertModel


class BertClassifier(torch.nn.Module):
    def __init__(self, num_labels: int, device: str):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('cointegrated/rubert-tiny2', output_attentions=False,
                                              output_hidden_states=False)
        self.fc1 = nn.Linear(312, 156)
        self.fc2 = nn.Linear(156, 64)
        self.fc3 = nn.Linear(64, num_labels)

        self.device = device

    def forward(self, x):
        outputs = self.bert(input_ids=x['input_ids'].to(self.device),
                            attention_mask=x['attention_mask'].to(self.device),
                            head_mask=None,
                            return_dict=True)

        hidden_state = outputs[0]

        x = hidden_state[:, 0]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return x
