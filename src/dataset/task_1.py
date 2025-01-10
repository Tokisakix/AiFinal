import os
import json
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

from src.setting import (
    TASK_1_DATA_ROOT,
    TASK_1_OUTPUT_ROOT,
)

def process_task_1_data():
    data_file = os.path.join(TASK_1_DATA_ROOT, "train.txt")
    with open(data_file, "r", encoding="utf-8") as f:
        raw_data = f.read()

    sentences = [line.strip() for line in tqdm(raw_data.split("\n")) if line.strip()]
    vocab = set("".join(sentences))

    token2idx = {
        "<pad>":0,
        "<unk>":1,
        "<sep>":2,
    }
    for token in vocab:
        token2idx[token] = len(token2idx)

    idx2token = {idx: token for token, idx in token2idx.items()}

    json.dump(
        token2idx, open(os.path.join(TASK_1_OUTPUT_ROOT, "token2idx.json"),
        "w", encoding="utf-8"), ensure_ascii=False,
    )
    json.dump(
        idx2token, open(os.path.join(TASK_1_OUTPUT_ROOT, "idx2token.json"),
        "w", encoding="utf-8"), ensure_ascii=False,
    )
    return

class DatasetTaskV1(Dataset):
    def __init__(self, file_path, vocab_path, max_length=512):
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

        with open(file_path, "r", encoding="utf-8") as f:
            self.data = f.read().split("\n")
            self.data = [x for x in self.data if x.strip() and len(x.strip()) <= 64]
            self.data = [x for x in self.data if len(x.strip()) >= 16]

        self.max_length = max_length
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        tokens = [self.vocab.get(char, self.vocab["<unk>"]) for char in text]

        if len(tokens) > self.max_length - 1:
            tokens = tokens[:self.max_length-1]

        tokens = tokens + [self.vocab["<pad>"]] * (self.max_length - len(tokens))

        x = torch.tensor(tokens[:-1])
        y = torch.tensor(tokens[1:])

        return x, y