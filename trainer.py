from multiprocessing.queues import Queue
from multiprocessing.connection import Connection
from typing import Tuple
from model import ConnNet
from reporter import Reporter, SharedWeights
from config import LR, EPOCHS, DEVICE, KL_TARG
import torch
import torch.nn.functional as F
import time



ten_num = lambda x:x.cpu().detach().item()
batch_size = 16

def feedback(q: Queue, resps: Tuple[Connection], sharew: SharedWeights):
    model = ConnNet()
    weights_version = 0
    while True:
        batch = [q.get()]
        while len(batch)<batch_size and not q.empty():
            batch.append(q.get())
        #import madbg; madbg.set_trace()
        if weights_version < sharew.version.value:
            weights, weights_version = sharew.get_weights()
            model.load_state_dict(weights)
        with torch.no_grad():
            model.eval()
            results=model(torch.stack([el.field for el in batch]))
        for (policy, value), info in zip(zip(*results),batch):
            resps[info.player].send((policy.numpy(), value.numpy()))

def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(reporter: Reporter, sharew: SharedWeights):
    model = ConnNet()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    lr_mult = 1
    while True:
        fields, probs, values = reporter.read()
        kl=KL_TARG
        for i in range(EPOCHS):
            optimizer.zero_grad()
            set_learning_rate(optimizer, LR*lr_mult)
            pred_probs, pred_values = model(fields.to(DEVICE))
            loss = F.cross_entropy(pred_probs, probs.to(DEVICE))+F.mse_loss(pred_values, values.to(DEVICE))
            loss.backward()
            optimizer.step()
            try:
                with torch.no_grad():
                    kl = torch.mean(torch.sum(old_probs*(torch.log(old_probs+1e-10)-torch.log(pred_probs+1e-10)),axis=1))
            except NameError:
                old_probs, old_values = pred_probs, pred_values 
            if kl > KL_TARG*4:
                break
        del old_probs, old_values
        kl = ten_num(kl)
        lr_mult *= 0.66 if kl>KL_TARG*2 else 1.5 if kl<KL_TARG/2 else 1
        lr_mult = max(min(lr_mult, 10), 0.1)
        sharew.save_weights(model.state_dict())
        
