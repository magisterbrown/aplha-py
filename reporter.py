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
        self.batch_size=batch_size
        self.reads=context.Array('i', [0]*batch_size)
        fmoves=torch.empty(batch_size,1,dtype=torch.int).share_memory_()
        fields=torch.empty(batch_size, 2, ROWS, COLS).share_memory_()
        probs=torch.empty(batch_size, COLS).share_memory_()
        values=torch.empty(batch_size, 1).share_memory_()
        self.shared_tensors = (fmoves, fields, probs, values)
        self.filled = context.Value('i', 0)
        self.full = context.Event()

    def insert(self, first_moves: List[bool], fields: List[torch.Tensor], probs: List[torch.Tensor], values: List[int]):
        for reported_tensors in zip(first_moves, fields, probs, values):
            with self.lock:
                if self.filled.value < self.batch_size:
                    for shared, reported in zip(self.shared_tensors, reported_tensors):
                        shared[self.filled.value] = reported
                    self.filled.value+=1 
                    print(f'Loading {self.filled.value}/{self.batch_size}')
                    if self.filled.value>=self.batch_size:
                        self.full.set()
                else:
                    probs = scipy.special.softmax(np.frombuffer(self.reads.get_obj(), dtype=np.int32))
                    replace = np.random.choice(np.arange(len(probs)),p=probs)
                    for shared, reported in zip(self.shared_tensors, reported_tensors):
                        shared[replace] = reported
                    self.reads[replace] = 0

    def read(self) -> torch.Tensor:
        self.full.wait()
        with self.lock:
            self.reads[:] = [x+1 for x in self.reads]
            return tuple(map(torch.clone, self.shared_tensors))

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

