import os
import json
from tqdm import tqdm

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