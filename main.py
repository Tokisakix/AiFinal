import argparse

from src.dataset import process_task_1_data, process_task_2_data
from src.train import train_task_1, train_task_2
from src.infer import infer

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--task", type=int, help="task number")
    args.add_argument("--mode", type=str, help="train or infer")
    args.add_argument("--nproc", type=int, default=1, help="number of processes")
    args.add_argument("--nodes", type=int, default=1, help="number of nodes")
    args.add_argument("--node_rank", type=int, default=0, help="node rank")
    args.add_argument("--master_addr", type=str, default="localhost", help="master address")
    args.add_argument("--master_port", type=int, default=12345, help="master port")
    args = args.parse_args()
    TASK = args.task
    MODE = args.mode

    if TASK == 1:
        process_task_1_data()
        print("[+] Process task 1 data done!")

        if MODE == "train":
            train_task_1(args.nproc, args.nodes, args.node_rank, args.master_addr, args.master_port)
            print("[+] Train task 1 done!")
        elif MODE == "infer":
            infer()
            print("[+] Infer task 1 done!")

    elif TASK == 2:
        process_task_2_data()
        print("[+] Process task 2 data done!")

        if MODE == "train":
            train_task_2(args.nproc, args.nodes, args.node_rank, args.master_addr, args.master_port)
            print("[+] Train task 2 done!")

    exit(0)