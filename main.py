import argparse

from src.dataset import process_task_1_data, process_task_2_data
from src.train import train_task_1, train_task_2
from src.infer import infer_task_1, infer_task_2

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--task", type=int)
    args.add_argument("--mode", type=str)
    args = args.parse_args()
    TASK = args.task
    MODE = args.mode

    if TASK == 1:
        process_task_1_data()
        print("[+] Process task 1 data done!")

        if MODE == "train":
            train_task_1()
            print("[+] Train task 1 done!")
        elif MODE == "infer":
            infer_task_1()
            print("[+] Infer task 1 done!")

    elif TASK == 2:
        process_task_2_data()
        print("[+] Process task 2 data done!")

        if MODE == "train":
            train_task_2()
            print("[+] Train task 1 done!")
        elif MODE == "infer":
            infer_task_2()
            print("[+] Infer task 1 done!")

    exit(0)