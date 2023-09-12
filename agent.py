from typing import Tuple
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.queues import Queue
import time
import torch

def play(i: int, pipe: Tuple[Connection], submit: Queue):
    my_conn = pipe[i]
    while True:
        play_record(i, my_conn, submit)

@dataclass
class Analyze:
    player: int
    field: torch.Tensor
    my_fig: int
    enemy_fig: int


def play_record(idx: int, pipe: Connection, submit: Queue):
    #submit.put(torch.tensor(3))
    for i in range(3):
        time.sleep(0.1)

        #Playgame
        submit.put(Analyze(idx, torch.tensor([3])))
        resp = pipe.recv()
        
    #config = structify({'rows':rows,'columns': columns,'inarow':win})
    #env = make("connectx", debug=False, configuration=config)
    #root = TreeNode()
    #steps = list()
    #while not env.done:
    #    board = env.state[0]['observation']['board']
    #    for i in range(200):
    #        playout(board, config, root, get_server_pred)
    #    #open('tree.dot', 'w').write(tree_to_dot(root))
    #    visit_probs = visits_to_probs([v._visit_count for v in root.children.values()])
    #    prob_pairs = {k:v for k,v in zip(root.children.keys(), visit_probs)}
    #    # TODO: submit visits as probailities not as counts
    #    steps.append({'field': board, 'probs': prob_pairs, 'player_fig': figures[root.player], 'enemy_fig': figures[not root.player]})
    #    visit_probs = visit_probs*(1-rand)+np.random.dirichlet(np.ones(visit_probs.shape),size=1)[0]*rand
    #    
    #    step = np.random.choice(list(prob_pairs.keys()), p=visit_probs)
    #    env.step([step]*2)
    #    root = root.children[step]
    #value = 0 if env.state[0]['reward']==0 else figures[not root.player]
    #data = {'steps':steps, 'winner': value}
    #resp = http.request('POST', server, body=json.dumps(data))
    #resp = json.loads(resp.data.decode('utf-8'))


