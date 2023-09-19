import torch
import random
import scipy
import numpy as np
from multiprocessing.context import SpawnContext
from multiprocessing import shared_memory
from collections import OrderedDict
from typing import Tuple
from typing import List
from config import ROWS, COLS, DTYPE

class Reporter:
    def __init__(self, context: SpawnContext, batch_size: int):
        self.lock=context.Lock()
        self.reads=context.Array('i', [0]*batch_size)
        self.fields=torch.empty(batch_size, 2, ROWS, COLS).share_memory_()
        self.probs=torch.empty(batch_size, COLS).share_memory_()
        self.values=torch.empty(batch_size, 1).share_memory_()
        self.filled = context.Value('i', 0)
        self.full = context.Event()

    def insert(self, fields: List[torch.Tensor], probs: List[torch.Tensor], values: List[int]):
        for field, prob, value in zip(fields, probs, values):
            with self.lock:
                if self.filled.value < self.fields.shape[0]:
                    self.fields[self.filled.value] = field
                    self.probs[self.filled.value] = prob 
                    self.values[self.filled.value] = value
                    self.filled.value+=1 
                    print(f'Loading {self.filled.value}/{self.fields.shape[0]}')
                    if self.filled.value>=self.fields.shape[0]:
                        self.full.set()
                else:
                    probs = scipy.special.softmax(np.frombuffer(self.reads.get_obj(), dtype=np.int32))
                    replace = np.random.choice(np.arange(len(probs)),p=probs)
                    #replace = random.randint(0, self.filled.value-1)
                    self.fields[replace] = field
                    self.probs[replace] = prob 
                    self.values[replace] = value
                    self.reads[replace] = 0

    def read(self) -> torch.Tensor:
        self.full.wait()
        with self.lock:
            self.reads[:] = [x+1 for x in self.reads]
            return tuple(map(torch.clone, (self.fields, self.probs, self.values)))

class SharedWeights:
    def __init__(self, manager):
        self.weights = manager.dict()
        self.version = manager.Value('i', 0)
        self.lock = manager.Lock()
    
    def save_weights(self, up_weights: OrderedDict):
        with self.lock:
            for k,v in up_weights.items():
                self.weights[k] = torch.clone(v).cpu()
            self.version.value+=1

    def get_weights(self) -> Tuple[dict, int] :
        with self.lock:
            return self.weights.copy(), self.version.value

