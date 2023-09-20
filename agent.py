from typing import Tuple
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.queues import Queue
from numpy.typing import NDArray
from model import field_to_tensor
from mcts import TreeNode
from kaggle_environments.utils import structify
from kaggle_environments import make
from config import ROWS, COLS, INAROW, RAND, DTYPE
from reporter import Reporter 
import time
import torch
import numpy as np

from game import playout, visits_to_probs

def play(i: int, pipe: Tuple[Connection], submit: Queue, reporter: Reporter):
    my_conn = pipe[i]
    while True:
        play_record(i, my_conn, submit, reporter)

@dataclass
class Analyze:
    player: int
    first_move: torch.Tensor
    field: torch.Tensor

def play_record(idx: int, pipe: Connection, submit: Queue, reporter: Reporter):
    figures = [1,2]
    def pipe_prediction(field: NDArray[int], leaf: TreeNode) -> Tuple[NDArray[float], float]:
        submit.put(Analyze(idx, torch.tensor([not leaf.player], dtype=torch.int), field_to_tensor(field,figures[leaf.player],figures[not leaf.player])))
        revced = pipe.recv()
        return  revced
    config = structify({'rows':ROWS,'columns': COLS,'inarow':INAROW})
    env = make("connectx", debug=False, configuration=config)
    root = TreeNode()
    fields = list()
    probs = list()
    while not env.done:
        board = env.state[0]['observation']['board']
        for i in range(200):
            playout(board, config, root, pipe_prediction)
        visit_probs = visits_to_probs([v._visit_count for v in root.children.values()])
        valid_moves = list(root.children.keys())
        tensor_probs = torch.zeros(config.columns)
        tensor_probs[valid_moves] = torch.from_numpy(visit_probs).to(DTYPE)
        probs.append(tensor_probs)
        fields.append(field_to_tensor(np.array(board), figures[root.player], figures[not root.player]))

        visit_probs = visit_probs*(1-RAND)+np.random.dirichlet(np.ones(visit_probs.shape),size=1)[0]*RAND
        step = int(np.random.choice(valid_moves, p=visit_probs))
        env.step([step]*2)
        root = root.children[step]
    values = ([-1,1]*(1+len(probs)//2))[-len(probs):] if env.state[0]['reward'] != 0 else [0]*len(probs)
    first_move = [i%2==0 for i in range(len(probs))]
    reporter.insert(first_move, fields, probs, values)
