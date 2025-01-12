import os
import json
import time
import torch
import torch.nn.functional as F
import torch.quantization
from torch.quantization import quantize_dynamic

from .sampler import Sampler
from src.model import GPTInferModel
from src.setting import (
    TASK_1_OUTPUT_ROOT,
    TASK_1_MODEL_PATH,
    INFER_TEMPERATURE,
    INFER_TOP_K,
    INFER_TOP_P,
    INFER_DEVICE,
)

def load_vocab(vocab_path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        token2idx = json.load(f)
    idx2token = {idx: token for token, idx in token2idx.items()}
    return token2idx, idx2token

def generate(model, token2idx, idx2token, input_text, sampler, device, max_length=64):
    model.eval()
    tokens = [token2idx.get(char, token2idx["<unk>"]) for char in input_text]
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)
    stop_token = [token2idx[token] for token in ["<sep>", "。", "？", "！"]]

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            next_token_logits = outputs[0, -1, :]
            probs = F.softmax(next_token_logits, dim=-1)
            # start_time = time.perf_counter()
            next_token_id = sampler.apply(probs)
            # end_time = time.perf_counter()
            # print(f"time: {end_time - start_time}")
            tokens.append(next_token_id)
            input_ids = torch.tensor([tokens], dtype=torch.long).to(device)
            if next_token_id in stop_token:
                break

    generated_tokens = input_ids[0].tolist()
    generated_text = "".join([idx2token.get(token_id, "") for token_id in generated_tokens])
    reply = generated_text[len(input_text):]
    return reply

def infer():
    token2idx, idx2token = load_vocab(os.path.join(TASK_1_OUTPUT_ROOT, "token2idx.json"))
    device = INFER_DEVICE

    model = GPTInferModel(len(token2idx)).to(device)
    model.load_state_dict(torch.load(TASK_1_MODEL_PATH, map_location=device, weights_only=True))
    if device == "cpu":
        model = quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        model.to(device)
    elif "cuda" in device:
        model = model.half()
        model = torch.compile(model)
    sampler = Sampler(INFER_TEMPERATURE, INFER_TOP_K, INFER_TOP_P)

    try:
        while True:
            user_input = input("[+] User: ")
            reply = generate(model, token2idx, idx2token, user_input, sampler, device)
            print(f"[+] Bot : {reply}")
    except KeyboardInterrupt:
        print("[+] Exit.")
    return