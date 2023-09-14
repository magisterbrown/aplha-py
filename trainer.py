from multiprocessing.queues import Queue
from multiprocessing.connection import Connection
from typing import Tuple
from model import ConnNet
from reporter import Reporter
import torch
import time


batch_size = 16

def feedback(q: Queue, resps: Tuple[Connection]):
    model = ConnNet()
    while True:
        batch = [q.get()]
        while len(batch)<batch_size and not q.empty():
            batch.append(q.get())
        with torch.no_grad():
            model.eval()
            results=model(torch.stack([el.field for el in batch]*2))
        for (policy, value), info in zip(zip(*results),batch*2):
            resps[info.player].send((policy.numpy(), value.numpy()))

def train(reporter: Reporter):
    model = ConnNet()
    while True:
        batch = reporter.read()
        #import madbg; madbg.set_trace()
        print('aa')
        break
        
    pass
