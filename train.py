import json
from dataset import LOGIDataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

# Load data
with open("data/logi_synthetic_data.jsonl", 'r') as f:
    data = [json.loads(line) for line in f]

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
dataset = LOGIDataset(data, tokenizer)
loader = DataLoader(dataset, batch_size=2, shuffle=True)
