import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import requests
import tiktoken

DATA_PATH = "sales_textbook.txt"
DATA_SOURCE_URL = "https://huggingface.co/datasets/goendalf666/sales-textbook_for_convincing_and_selling/raw/main/sales_textbook.txt"

if not os.path.exists(DATA_PATH):
    r = requests.get(DATA_SOURCE_URL)
    with open(DATA_PATH, "w") as f:
        f.write(r.text)

with open(DATA_PATH, "r") as f:
    text = f.read()

encoding = tiktoken.get_encoding("cl100k_base")
tokenized_text = encoding.encode(text)

max_token_value = max(tokenized_text) + 1
tokenized_text = torch.tensor(tokenized_text)

# split the data into training and validation sets
train_size = int(0.9 * len(tokenized_text))
val_size = len(tokenized_text) - train_size

print(f"Training set size: {train_size}")
print(f"Validate set size: {val_size}")

train_data = tokenized_text[:train_size]
val_data = tokenized_text[train_size:]

# hyperparameters
context_length = 16
d_model = 64
n_heads = 4
batch_size = 4
learning_rate = 1e-3
dropout = 0.1
max_iters = 500
eval_interval = 50
eval_iters = 20
device = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_SPEED = 1337
torch.manual_seed(TORCH_SPEED)

class FeedforwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x)))

class MultiHeadAttention(nn.Module):

