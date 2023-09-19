import torch
from model import ConnNet, field_to_tensor
from game import playout
from mcts import TreeNode

model = ConnNet()
model.load_state_dict(torch.load('points/last.pth'))
model.eval()
import numpy as np

def get_pred(field, leaf):
    figures = [1,2]
    with torch.no_grad():
        res = model(field_to_tensor(field,figures[leaf.player],figures[not leaf.player])[None])
    return res[0][0].numpy(), res[1].item()

def alpha_agent(obs, config):
    root = TreeNode(player=obs.mark==2)
    for i in range(20):
        playout(obs.board, config, root, get_pred)
    visit_counts = { k:x._visit_count for k,x in root.children.items()}
    print(np.reshape(np.array(obs.board),(4,5)))
    print(visit_counts)
    return max(visit_counts, key=visit_counts.get)
