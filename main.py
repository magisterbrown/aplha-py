import torch
from torch.multiprocessing import Queue, Pipe, Process, spawn, get_context
from trainer import feedback, train
from agent import play
from reporter import Reporter, SharedWeights
from config import TRAIN_BATCH 

smp = get_context('spawn')
train_fields = Reporter(smp, TRAIN_BATCH)
agents = 8
resps, waits = list(zip(*[Pipe() for p in range(agents)]))
if __name__ == '__main__':
    manager = smp.Manager()
    sharew = SharedWeights(manager)
    to_analyze = smp.Queue()
    predictor = smp.Process(target=feedback, args=(to_analyze, resps, sharew))
    trainer = smp.Process(target=train, args=(train_fields, sharew))

    predictor.start()
    trainer.start()
    spawn(play, args=(waits, to_analyze, train_fields), nprocs=agents)
    trainer.join()
    predictor.join()
