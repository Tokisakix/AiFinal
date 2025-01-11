import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import process_task_1_data, DatasetTaskV1
from src.model import GPTModel
from src.setting import (
    TASK_1_DATA_ROOT,
    TASK_1_OUTPUT_ROOT,
    TASK_1_SAVE_ROOT,
    TASK_1_BATCH_SIZE,
    TASK_1_LEARNING_RATE,
    TASK_1_NUM_EPOCHS,
    D_MODEL,
    N_HEADS,
    N_LAYERS,
    D_FF,
    DROPOUT,
)

def train_model(model, train_loader, optimizer, criterion, device, num_epochs, save_dir, start_epoch):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    saved_checkpoints = []

    model.train()
    for epoch in range(start_epoch, num_epochs):
        total_loss = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output.view(-1, output.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch: {epoch}, Batch: {batch_idx} / {len(train_loader)}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")

        checkpoint_name = f"epoch_{epoch+1}-loss_{avg_loss:.4f}.pt"
        checkpoint_path = os.path.join(save_dir, checkpoint_name)
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
        }, checkpoint_path)

        saved_checkpoints.append(checkpoint_path)

        if len(saved_checkpoints) > 5:
            oldest_checkpoint = saved_checkpoints.pop(0)
            os.remove(oldest_checkpoint)
        return

def train_task_1():
    process_task_1_data()
    dataset = DatasetTaskV1(
        os.path.join(TASK_1_DATA_ROOT, "train.txt"),
        os.path.join(TASK_1_OUTPUT_ROOT, "token2idx.json"),
    )

    train_loader = DataLoader(dataset, batch_size=TASK_1_BATCH_SIZE, shuffle=True)
    token2idx = json.load(open(os.path.join(TASK_1_OUTPUT_ROOT, "token2idx.json"), "r", encoding="utf-8"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTModel(
        len(token2idx),
        D_MODEL,
        N_HEADS,
        N_LAYERS,
        D_FF,
        DROPOUT
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=TASK_1_LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=token2idx["<pad>"])

    if not os.path.exists(TASK_1_SAVE_ROOT):
        os.makedirs(TASK_1_SAVE_ROOT)

    checkpoint_files = [f for f in os.listdir(TASK_1_SAVE_ROOT) if f.endswith(".pt") and f.startswith("epoch_")]
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split("_")[1].split("-")[0]))
        checkpoint_path = os.path.join(TASK_1_SAVE_ROOT, latest_checkpoint)

        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming training from epoch {start_epoch}")
    else:
        start_epoch = 0

    train_model(model, train_loader, optimizer, criterion, device, TASK_1_NUM_EPOCHS, TASK_1_SAVE_ROOT, start_epoch)
    return