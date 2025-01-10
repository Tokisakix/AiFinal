import os

# Path setting
DATA_ROOT = "data/"
OUTPUT_ROOT = ".output/"

SEED = 22331109

# Task 1 setting
TASK_1_DATA_ROOT = os.path.join(DATA_ROOT, "task_1")
TASK_1_OUTPUT_ROOT = os.path.join(OUTPUT_ROOT, "task_1")

# Task 2 setting
TASK_2_DATA_ROOT = os.path.join(DATA_ROOT, "task_2")
TASK_2_OUTPUT_ROOT = os.path.join(OUTPUT_ROOT, "task_2")

# Train setting
BATCH_SIZE = 96
D_MODEL = 256
N_HEADS = 8
N_LAYERS = 6
D_FF = 1024
DROPOUT = 0.1
LEARNING_RATE = 0.0001
NUM_EPOCHS = 50