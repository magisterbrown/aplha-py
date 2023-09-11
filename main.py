import torch
from torch.multiprocessing import Queue, Pipe, Process, spawn, get_context
from trainer import feedback
from agent import play

smp = get_context('spawn')
to_analyze = smp.Queue()
agents = 6
resps, waits = list(zip(*[Pipe() for p in range(agents)]))
if __name__ == '__main__':
    trainer = smp.Process(target=feedback, args=(to_analyze, resps))
    trainer.start()
    spawn(play, args=(waits, to_analyze), nprocs=agents)
    trainer.join()
