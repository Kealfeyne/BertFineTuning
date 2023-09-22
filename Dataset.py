import torch
from torch.utils.data import Dataset


class Dataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer.batch_encode_plus(texts,
                                                    add_special_tokens=False,
                                                    return_attention_mask=True,
                                                    padding=True,
                                                    truncation=True,
                                                    # pad_to_max_length=True,
                                                    return_tensors='pt')
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: v[idx].clone().detach() for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
    def __len__(self):
        return len(self.encodings['input_ids'])