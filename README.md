# 中山大学 2024 年机器学习第二次大作业

## 环境配置

```bash
conda create -n <env_name> python=3.12
conda activate <env_name>
pip install -r requirements.txt
```

## 运行方式

执行 `python main.py -h` 查看运行方式

```bash
usage: main.py
    [-h]
    [--task TASK]
    [--mode MODE]
    [--nproc NPROC]
    [--nodes NODES]
    [--node_rank NODE_RANK]
    [--master_addr MASTER_ADDR]
    [--master_port MASTER_PORT]

options:
  -h, --help                    show this help message and exit
  --task TASK                   task number
  --mode MODE                   train or infer
  --nproc NPROC                 number of processes
  --nodes NODES                 number of nodes
  --node_rank NODE_RANK         node rank
  --master_addr MASTER_ADDR     master address
  --master_port MASTER_PORT     master port
```

## 任务 1

训练超参数可以在 `src/setting.py` 中直接修改 

**1. 单机单卡训练**

```bash
cd workfolder/
python main.py --task 1 --mode train
```

**2. 单机多卡训练**

```bash
cd workfolder/
python main.py --task 1 --mode train --nproc 6
```

**3. 多机多卡训练**

```bash
cd workfolder/

# in node 0
python main.py --task 1 --mode train --nproc 6 --nodes 2 --node_rank 0 --master_addr <master_addr> --master_port <master_port>

# in node 1
python main.py --task 1 --mode train --nproc 6 --nodes 2 --node_rank 1 --master_addr <master_addr> --master_port <master_port>
```

**4. 推理**

```bash
python main.py --task 1 --mode infer
```

## 任务 2

训练超参数可以在 `src/setting.py` 中直接修改 

**1. 单机单卡微调**

```bash
cd workfolder/
python main.py --task 2 --mode train
```

**2. 单机多卡微调**

```bash
cd workfolder/
python main.py --task 2 --mode train --nproc 6
```

**3. 多机多卡微调**

```bash
cd workfolder/

# in node 0
python main.py --task 2 --mode train --nproc 6 --nodes 2 --node_rank 0 --master_addr <master_addr> --master_port <master_port>

# in node 1
python main.py --task 2 --mode train --nproc 6 --nodes 2 --node_rank 1 --master_addr <master_addr> --master_port <master_port>
```