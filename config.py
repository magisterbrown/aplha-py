import torch
ROWS=4
COLS=5
INAROW=3
RAND=0.25
DTYPE=torch.float32

#Model
LR=2e-3
TRAIN_BATCH=512
EPOCHS=5
KL_TARG=0.02
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
