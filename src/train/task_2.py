import os
import json
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.dataset import DatasetTaskV2
from src.model import GPTModel
from src.setting import (
    TASK_2_OUTPUT_ROOT,
    TASK_2_SAVE_ROOT,
    TASK_2_BATCH_SIZE,
    TASK_2_LEARNING_RATE,
    TASK_2_NUM_EPOCHS,
    TASK_2_FREEZE_TYPE,
    TASK_2_SAVE_MODEL_NUM,
    TASK_1_MODEL_PATH,
    D_MODEL,
    N_HEADS,
    N_LAYERS,
    D_FF,
    DROPOUT,
)

def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            pooled_output = torch.mean(outputs, dim=1)
            pred = model.classifier(pooled_output)
            predicted = (torch.sigmoid(pred) > 0.5).float()
            total += y.size(0)
            correct += (predicted.squeeze() == y).sum().item()
    accuracy = 100 * correct / total
    return accuracy

def freeze_parameters(model, freeze_type="all"):
    if freeze_type == "all":
        for param in model.parameters():
            param.requires_grad = True
    elif freeze_type == "last":
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if "layers.5" in name:
                param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True
    return

def train_model(rank, model, train_loader, test_loader, optimizer, criterion, device, num_epochs, save_dir):
    if rank == 0 and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_model_path_list = []
        
    model.train()
    for epoch in range(1, num_epochs + 1):
        train_loss = 0
        correct = 0
        total = 0
        for x, y in tqdm(enumerate(train_loader)):
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            pooled_output = torch.mean(outputs, dim=1)
            pred = model.classifier(pooled_output)

            optimizer.zero_grad()
            loss = criterion(pred.squeeze(), y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            total += y.size(0)
            predicted = (torch.sigmoid(pred) > 0.5).float()
            correct += (predicted.squeeze() == y).sum().item()
            
        if rank == 0:
            train_loss = train_loss / len(train_loader)
            train_acc = 100 * correct / total
            test_acc = evaluate_model(model, test_loader, device)
            print(f"[+] Epoch {epoch} loss: {train_loss:.4f} Train Acc {train_acc:.2f}% Test Acc: {test_acc:.2f}%")

            save_model_path = os.path.join(save_dir, f"model_{epoch}.pth")
            torch.save(model.state_dict(), save_model_path)
            save_model_path_list.append(save_model_path)
            print(f"[+] Save model into {save_model_path}")

            if len(save_model_path_list) > TASK_2_SAVE_MODEL_NUM:
                remove_model_path = save_model_path_list.pop(0)
                os.remove(remove_model_path)
                print(f"[+] Remove {remove_model_path}")

    return

def load_pretrained_model(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    pretrained_dict = checkpoint["model_state_dict"]
    model_dict = model.state_dict()
    pretrained_embed = pretrained_dict["embedding.weight"]
    current_embed = model_dict["embedding.weight"]
    current_embed[:pretrained_embed.size(0)] = pretrained_embed
    pretrained_out_w = pretrained_dict["out.weight"]
    pretrained_out_b = pretrained_dict["out.bias"]
    current_out_w = model_dict["out.weight"]
    current_out_b = model_dict["out.bias"]
    current_out_w[:pretrained_out_w.size(0)] = pretrained_out_w
    current_out_b[:pretrained_out_b.size(0)] = pretrained_out_b
    for name, param in pretrained_dict.items():
        if name not in ["embedding.weight", "out.weight", "out.bias"]:
            model_dict[name].copy_(param)
    model.load_state_dict(model_dict)
    return model

def run_train_task_2(rank, world_size, nodes, node_rank, master_addr, master_port):
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=world_size,
        rank=rank,
    )

    train_dataset = DatasetTaskV2(
        os.path.join(TASK_2_OUTPUT_ROOT, "emotion_train.csv"),
        os.path.join(TASK_2_OUTPUT_ROOT, "token2idx.json"),
    )
    test_dataset = DatasetTaskV2(
        os.path.join(TASK_2_OUTPUT_ROOT, "emotion_test.csv"),
        os.path.join(TASK_2_OUTPUT_ROOT, "token2idx.json"),
    )

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=TASK_2_BATCH_SIZE, sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=TASK_2_BATCH_SIZE, sampler=test_sampler)

    token2idx = json.load(open(os.path.join(TASK_2_OUTPUT_ROOT, "token2idx.json"), "r", encoding="utf-8"))

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    model = GPTModel(
        len(token2idx),
        D_MODEL,
        N_HEADS,
        N_LAYERS,
        D_FF,
        DROPOUT
    ).to(device)

    model = load_pretrained_model(model, TASK_1_MODEL_PATH, device)

    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.out.out_features, 1)
    ).to(device)

    freeze_parameters(model, freeze_type=TASK_2_FREEZE_TYPE)

    model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=TASK_2_LEARNING_RATE,
        weight_decay=1e-2
    )
    criterion = nn.BCEWithLogitsLoss()

    train_model(model, train_loader, test_loader, optimizer, criterion, device, TASK_2_NUM_EPOCHS, TASK_2_SAVE_ROOT)

    dist.destroy_process_group()
    return

def train_task_2(nproc, nodes, node_rank, master_addr, master_port):
    world_size = nproc * nodes
    mp.spawn(
        run_train_task_2,
        args=(world_size, nodes, node_rank, master_addr, master_port),
        nprocs=nproc,
        join=True
    )
    return