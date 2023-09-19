from kaggle_environments import evaluate, make, utils
from kaggle_environments.envs.connectx.connectx import negamax_agent, random_agent
from config import ROWS, COLS, INAROW
from kaggle_environments.utils import structify
from xbot import alpha_agent

if __name__=='__main__':
    config = structify({'rows':ROWS,'columns': COLS,'inarow':INAROW})
    env = make("connectx",configuration=config, debug=True)
    results = {-1:0,0:0,1:0}
    for i in range(1):
        env.run([alpha_agent, negamax_agent])
        results[env.state[0].reward]+=1
        print(i)
    print(f'WIns {results[1]} Losses: {results[-1]} Draws: {results[0]}')
