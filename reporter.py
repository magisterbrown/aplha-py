import torch
from multiprocessing.context import SpawnContext
from multiprocessing import shared_memory
from typing import List
class Reporter:
    def __init__(self, context: SpawnContext, batch_size: int):
        self.lock=context.Lock()
        self.data=shared_memory.ShareableList()
        self.batch_size = batch_size

    def insert(self, fields: List[torch.Tensor], probs: List[torch.Tensor], winner: int):
        import madbg;madbg.set_trace()
        pass

    def read(self) -> torch.Tensor:
        pass


