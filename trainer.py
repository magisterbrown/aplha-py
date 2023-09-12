from multiprocessing.queues import Queue
from multiprocessing.connection import Connection
from typing import Tuple
import torch
import time


batch_size = 16

def feedback(q: Queue, resps: Tuple[Connection]):
    while True:
        batch = [q.get()]
        while len(batch)<batch_size and not q.empty():
            batch.append(q.get())
        process = torch.stack([el.field for el in batch])
        
        # TODO: RUN nn

        for waiter in batch:
            print(resps)
            print(waiter.player)
            resps[waiter.player].send(8)
        time.sleep(0.3)

    pass
