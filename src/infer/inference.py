import os
import json
import torch
import torch.nn.functional as F

from .sampler import Sampler
from src.model import GPTModel
from src.setting import (
    TASK_1_OUTPUT_ROOT,
    INFER_TEMPERATURE,
    INFER_TOP_K,
    INFER_TOP_P,
)

def load_vocab(vocab_path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        token2idx = json.load(f)
    idx2token = {idx: token for token, idx in token2idx.items()}
    return token2idx, idx2token

def generate(model, token2idx, idx2token, input_text, sampler, device, max_length=50):
    model.eval()
    tokens = [token2idx.get(char, token2idx["<unk>"]) for char in input_text]
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            next_token_logits = outputs[:, -1, :]
            probs = F.softmax(next_token_logits, dim=-1)
            next_token_id = sampler.apply(probs)
            input_ids = torch.cat([input_ids, next_token_id], dim=1)
            if next_token_id.item() == token2idx["<sep>"]:
                break

    generated_tokens = input_ids[0].tolist()
    generated_text = "".join([idx2token.get(token_id, "") for token_id in generated_tokens])
    reply = generated_text[len(input_text):]
    return reply

def infer():
    token2idx, idx2token = load_vocab(os.path.join(TASK_1_OUTPUT_ROOT, "token2idx.json"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GPTModel(len(token2idx)).to(device)
    checkpoint = torch.load(".checkpoints/best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    sampler = Sampler(INFER_TEMPERATURE, INFER_TOP_K, INFER_TOP_P)

    try:
        while True:
            user_input = input("[+] User: ")
            reply = generate(model, token2idx, idx2token, user_input, sampler, device)
            print(f"[+] Bot : {reply}")
    except KeyboardInterrupt:
        print("[+] Exit.")