import os
import json
import torch
from torch import nn
from tqdm import tqdm
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.dataset import DatasetTaskV1
from src.model import GPTModel
from src.setting import (
    TASK_1_DATA_ROOT,
    TASK_1_OUTPUT_ROOT,
    TASK_1_SAVE_ROOT,
    TASK_1_BATCH_SIZE,
    TASK_1_LEARNING_RATE,
    TASK_1_NUM_EPOCHS,
    TASK_1_SAVE_MODEL_NUM,
    D_MODEL,
    N_HEADS,
    N_LAYERS,
    D_FF,
    DROPOUT,
)

def train_model(model, train_loader, optimizer, criterion, device, num_epochs, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_model_path_list = []

    model.train()
    for epoch in range(1, num_epochs + 1):
        total_loss = 0
        for x, y in tqdm(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output.view(-1, output.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[+] Epoch {epoch} completed. Average loss: {avg_loss:.4f}")

        save_model_path = os.path.join(save_dir, f"model_{epoch}.pth")
        torch.save(model.state_dict(), save_model_path)
        save_model_path_list.append(save_model_path)
        print(f"[+] Save model into {save_model_path}")

        if len(save_model_path_list) > TASK_1_SAVE_MODEL_NUM:
            remove_model_path = save_model_path_list.pop(0)
            os.remove(remove_model_path)
            print(f"[+] Remove {remove_model_path}")

    return

def run_train_task_1(rank, world_size, nodes, node_rank, master_addr, master_port):
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=world_size,
        rank=rank,
    )

    dataset = DatasetTaskV1(
        os.path.join(TASK_1_DATA_ROOT, "train.txt"),
        os.path.join(TASK_1_OUTPUT_ROOT, "token2idx.json"),
    )

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(dataset, batch_size=TASK_1_BATCH_SIZE, sampler=sampler)

    token2idx = json.load(open(os.path.join(TASK_1_OUTPUT_ROOT, "token2idx.json"), "r", encoding="utf-8"))

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    model = GPTModel(
        len(token2idx),
        D_MODEL,
        N_HEADS,
        N_LAYERS,
        D_FF,
        DROPOUT
    ).to(device)

    model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=TASK_1_LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=token2idx["<pad>"])

    if not os.path.exists(TASK_1_SAVE_ROOT):
        os.makedirs(TASK_1_SAVE_ROOT)

    train_model(model, train_loader, optimizer, criterion, device, TASK_1_NUM_EPOCHS, TASK_1_SAVE_ROOT)

    dist.destroy_process_group()
    return

def train_task_1(nproc, nodes, node_rank, master_addr, master_port):
    world_size = nproc * nodes
    mp.spawn(
        run_train_task_1,
        args=(world_size, nodes, node_rank, master_addr, master_port),
        nprocs=nproc,
        join=True,
    )
    return