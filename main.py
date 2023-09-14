import torch
from torch.multiprocessing import Queue, Pipe, Process, spawn, get_context
from trainer import feedback, train
from agent import play
from reporter import Reporter

smp = get_context('spawn')
train_fields = Reporter(smp, 4)
to_analyze = smp.Queue()
agents = 1
resps, waits = list(zip(*[Pipe() for p in range(agents)]))
if __name__ == '__main__':
    predictor = smp.Process(target=feedback, args=(to_analyze, resps))
    trainer = smp.Process(target=train, args=(train_fields, ))

    predictor.start()
    trainer.start()
    spawn(play, args=(waits, to_analyze, train_fields), nprocs=agents)
    trainer.join()
    predictor.join()
