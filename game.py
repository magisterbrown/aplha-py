from typing import List, Callable, Tuple
from numpy.typing import NDArray
from kaggle_environments.utils import Struct, structify
from kaggle_environments.envs.connectx.connectx import play, is_win
from mcts import TreeNode, find_best_playout
import numpy as np
import scipy

def visits_to_probs(probs: List[int], temp: float=1e-3) -> NDArray[float]:
    return scipy.special.softmax(1.0/temp * np.log(np.array(probs) + 1e-10))

def playout(board: List[int], 
        conf: Struct,
        root: TreeNode, 
        get_prediction: Callable[[NDArray[int], TreeNode], Tuple[NDArray[float], float]]):
    board = np.array(board)
    figures = [1,2]
    curr = root.player
    steps, leaf = find_best_playout(root)
    for step in steps:
        play(board, step, figures[curr], conf)
        curr = not curr
    
    if np.all(board[:conf.columns]!=0):
        leaf.update(0)
    elif len(steps) and is_win(board, steps[-1], figures[not leaf.player], conf, has_played=True):
        leaf.update(1)
    else:  
        predicted_probs, value = get_prediction(board, leaf)
        ilegal_moves = np.nonzero(board[:conf.columns])[0]
        predicted_probs[ilegal_moves] = 0
        leaf.expand(predicted_probs)
        leaf.update(value)

