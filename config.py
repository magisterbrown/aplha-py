import torch
ROWS=3
COLS=4
INAROW=3
RAND=0.25
DTYPE=torch.float32

#Model
LR=2e-3
TRAIN_BATCH=256
EPOCHS=3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
