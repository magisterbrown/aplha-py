from typing import Tuple
from multiprocessing.connection import Connection
from multiprocessing.queues import Queue

def play(i: int, pipe: Tuple[Connection], submit: Queue):
    my_conn = pipe[i]
    print(type(submit))
