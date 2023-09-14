import torch
import random
from multiprocessing.context import SpawnContext
from multiprocessing import shared_memory
from typing import List
from config import ROWS, COLS, DTYPE

class Reporter:
    def __init__(self, context: SpawnContext, batch_size: int):
        self.lock=context.Lock()
        self.fields=torch.empty(batch_size, 2, ROWS, COLS).share_memory_()
        self.probs=torch.empty(batch_size, COLS).share_memory_()
        self.values=torch.empty(batch_size, 1).share_memory_()
        self.filled = context.Value('i', 0)
        self.full = context.Event()

    def insert(self, fields: List[torch.Tensor], probs: List[torch.Tensor], values: List[int]):
        for field, prob, value in zip(fields, probs, values):
            self.lock.acquire()
            if self.filled.value < self.fields.shape[0]:
                self.fields[self.filled.value] = field
                self.probs[self.filled.value] = prob 
                self.values[self.filled.value] = value
                self.filled.value+=1 
                if self.filled.value>=self.fields.shape[0]:
                    self.full.set()
            else:
                #TODO: use probailities to replace older
                replace = random.randint(0, self.filled.value-1)
                self.fields[replace] = field
                self.probs[replace] = prob 
                self.values[replace] = value
            self.lock.release()

    def read(self) -> torch.Tensor:
        self.full.wait()
        return tuple(map(torch.clone, (self.fields, self.probs, self.values)))


