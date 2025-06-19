import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class LOGIDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=64):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item["input"]
        reasoning_text = item["reasoning"]
        answer_text = item["answer"]
        trigger = item["tool_trigger"]

        input_ids = self.tokenizer.encode(input_text, truncation=True, padding='max_length', max_length=self.max_length)
        reasoning_ids = self.tokenizer.encode(reasoning_text, truncation=True, padding='max_length', max_length=self.max_length)
        answer_ids = self.tokenizer.encode(answer_text, truncation=True, padding='max_length', max_length=self.max_length)

        return {
            'input_ids': torch.tensor(input_ids),
            'reasoning_ids': torch.tensor(reasoning_ids),
            'answer_ids': torch.tensor(answer_ids),
            'tool_trigger': torch.tensor(trigger)
        }
