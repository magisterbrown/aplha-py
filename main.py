import torch
from torch.multiprocessing import Queue, Pipe, Process, spawn
from trainer import train
from agent import play

to_analyze = Queue()
agents = 6
resps, waits = list(zip(*[Pipe() for p in range(agents)]))
if __name__ == '__main__':
    trainer = Process(target=train, args=(to_analyze,))
    trainer.start()
    spawn(play, args=(waits, to_analyze), nprocs=agents)
    trainer.join()
