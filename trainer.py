from multiprocessing.queues import Queue
from multiprocessing.connection import Connection
from typing import Tuple
from model import ConnNet
from reporter import Reporter
from config import LR, EPOCHS, DEVICE
import torch
import torch.nn.functional as F
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
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    while True:
        fields, probs, values = reporter.read()
        for i in range(EPOCHS):
            import madbg; madbg.set_trace()
            optimizer.zero_grad()
            pred_probs, pred_values = model(fields.to(DEVICE))
            loss = F.cross_entropy(pred_probs, probs.to(DEVICE))+F.mse_loss(pred_values, values.to(DEVICE))
            loss.backward()
            optimizer.step()

            pass
        #import madbg; madbg.set_trace()
        print('aa')
        break
        
    pass
