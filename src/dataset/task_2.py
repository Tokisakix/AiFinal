import os
import json
import torch
import pandas as pd
from sklearn.utils import shuffle
from torch.utils.data import Dataset

from src.setting import (
    SEED,
    TASK_1_OUTPUT_ROOT,
    TASK_2_DATA_ROOT,
    TASK_2_OUTPUT_ROOT,
)

def process_task_2_data():
    with open(os.path.join(TASK_1_OUTPUT_ROOT, "token2idx.json"), "r", encoding="utf-8") as f:
        token2idx = json.load(f)

    data = pd.read_csv(os.path.join(TASK_2_DATA_ROOT, "ChnSentiCorp_htl_all.csv"))
    data = data.dropna(subset=["review", "label"]).reset_index(drop=True)
    data = shuffle(data, random_state=SEED).reset_index(drop=True)
    data_0 = data[data["label"] == 0].reset_index(drop=True)
    data_1 = data[data["label"] == 1].reset_index(drop=True)

    n_samples = min(len(data_0), len(data_1))
    balanced_data_0 = data_0.head(n_samples)
    balanced_data_1 = data_1.head(n_samples)
    balanced_data = pd.concat([balanced_data_0, balanced_data_1], ignore_index=True)

    test_data_0 = balanced_data_0.sample(n=250, random_state=SEED)
    test_data_1 = balanced_data_1.sample(n=250, random_state=SEED)
    test_data = pd.concat([test_data_0, test_data_1], ignore_index=True)

    train_data_0 = balanced_data_0.drop(test_data_0.index).reset_index(drop=True)
    train_data_1 = balanced_data_1.drop(test_data_1.index).reset_index(drop=True)
    train_data = pd.concat([train_data_0, train_data_1], ignore_index=True)

    train_data.to_csv(os.path.join(TASK_2_OUTPUT_ROOT, "emotion_train.csv"), index=False)
    test_data.to_csv(os.path.join(TASK_2_OUTPUT_ROOT, "emotion_test.csv"), index=False)
    all_text = "".join(map(str, train_data["review"]))
    vocab = set(all_text)

    for token in vocab:
        if token not in token2idx:
            token2idx[token] = len(token2idx)

    idx2token = {idx: token for token, idx in token2idx.items()}

    json.dump(
        token2idx, open(os.path.join(TASK_2_OUTPUT_ROOT, "token2idx.json"),
        "w", encoding="utf-8"), ensure_ascii=False,
    )
    json.dump(
        idx2token, open(os.path.join(TASK_2_OUTPUT_ROOT, "idx2token.json"),
        "w", encoding="utf-8"), ensure_ascii=False,
    )
    return

class DatasetTaskV2(Dataset):
    def __init__(self, csv_file, vocab_path, max_length=512):
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        self.data = pd.read_csv(csv_file, dtype={"review": str, "label": int})
        self.data["review"] = self.data["review"].fillna("")
        self.max_length = max_length
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.loc[idx, "review"]
        label = self.data.loc[idx, "label"]
        if not isinstance(text, str):
            text = str(text)
        tokens = [self.vocab.get(char, self.vocab["<unk>"]) for char in text]
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens += [self.vocab["<pad>"]] * (self.max_length - len(tokens))
        input_ids = torch.tensor(tokens)
        label = torch.tensor(label, dtype=torch.float)
        return input_ids, label