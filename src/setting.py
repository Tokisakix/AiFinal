import os

# Path setting
DATA_ROOT   = "data/"
OUTPUT_ROOT = ".output/"
SAVE_ROOT   = ".checkpoints/"

SEED = 22331109

# Task 1 setting
TASK_1_DATA_ROOT   = os.path.join(DATA_ROOT, "task_1")
TASK_1_OUTPUT_ROOT = os.path.join(OUTPUT_ROOT, "task_1")
TASK_1_SAVE_ROOT   = os.path.join(SAVE_ROOT, "task_1")

TASK_1_BATCH_SIZE    = 32
TASK_1_LEARNING_RATE = 1e-4
TASK_1_NUM_EPOCHS    = 50

# Task 2 setting
TASK_2_DATA_ROOT   = os.path.join(DATA_ROOT, "task_2")
TASK_2_OUTPUT_ROOT = os.path.join(OUTPUT_ROOT, "task_2")
TASK_2_SAVE_ROOT   = os.path.join(SAVE_ROOT, "task_2")

TASK_2_BATCH_SIZE    = 32
TASK_2_LEARNING_RATE = 1e-5
TASK_2_NUM_EPOCHS    = 20
TASK_2_FREEZE_TYPE   = "all" # "all" or "last"

# Model setting
D_MODEL  = 256
N_HEADS  = 8
N_LAYERS = 6
D_FF     = 1024
DROPOUT  = 0.1

# Inference setting
INFER_TEMPERATURE = 0.8
INFER_TOP_K = 40
INFER_TOP_P = 0.9