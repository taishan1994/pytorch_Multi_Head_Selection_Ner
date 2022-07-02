import torch
from torch.utils.data import Dataset, DataLoader
from preprocess_mhs import BertFeature
from utils import commonUtils


class NerDataset(Dataset):
    def __init__(self, features):
        # self.callback_info = callback_info
        self.nums = len(features)

        self.token_ids = [torch.tensor(example.token_ids).long() for example in features]
        self.attention_masks = [torch.tensor(example.attention_masks, dtype=torch.uint8) for example in features]
        self.token_type_ids = [torch.tensor(example.token_type_ids).long() for example in features]
        self.pair_labels = [torch.tensor(example.pair_labels).long() for example in features]
        self.entity_labels = [torch.tensor(example.entity_labels).float() for example in features]

    def __len__(self):
        return self.nums

    def __getitem__(self, index):
        data = {
            'token_ids': self.token_ids[index],
            'attention_masks': self.attention_masks[index],
            'token_type_ids': self.token_type_ids[index]
        }

        data['pair_labels'] = self.pair_labels[index]
        data['entity_labels'] = self.entity_labels[index]
        return data
