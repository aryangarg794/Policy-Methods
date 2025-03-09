from policy_methods.reinforce import PolicyGradient
from policy_methods.actor_critic import QActorCritic

import os
import argparse
import gymnasium as gym
import numpy as np
from tqdm import tqdm


agents = {
    'reinforce' : 'REINFORCE',
    'ac' : 'Actor-Critic',
}

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--agent', type=str, default='reinforce', 
                    help='select agent: reinforce -> REINFORCE, ac -> Actor-Critic')

parser.add_argument('-e', '--numsteps', type=int, default=1000, help='number of steps/episodes')
parser.add_argument('-lc', '--lrc', type=float, default=0.001, help='learning rate critic')
parser.add_argument('-la', '--lra', type=float, default=0.001, help='learning rate actor')
parser.add_argument('-i', '--input', type=float, default=4, help='input shape')
parser.add_argument('-y', '--discount', type=float, default=0.99, help='discount factor')
parser.add_argument('-d', '--device', type=str, default='cpu', help='device')
parser.add_argument('-g', '--game', type=str, default='CartPole-v1', help='select game to run on')
parser.add_argument('-s', '--save', type=bool, default=True, help='to save or not')
args = parser.parse_args()
args_dict = vars(args)

agent_type = args.agent
game = args.game

env = gym.make(game)

if agent_type == 'reinforce':
    agent = PolicyGradient(environ=env,
                           input_shape=args.input,
                           gamma=args.discount,
                           learning_rate=args.lrc,
                           game_name=args.game,
                           device=args.device
                           )
elif agent_type == 'ac':
    agent = QActorCritic(environ=env,
                         input_shape=args.input,
                         gamma=args.discount,
                         lr_critic=args.lrc,
                         lr_actor=args.lra,
                         game_name=args.game,
                         device=args.device) 
    
    
if __name__ == "__main__":
    text = f'Training with Agent: {agents[agent_type]}'
    terminal_width = os.get_terminal_size().columns
    padding = (terminal_width - len(text)) // 2
    print('=' * terminal_width)
    print(' ' * padding + text)
    print('=' * terminal_width + '\n')
    
    for setting, value in args_dict.items():
        setting_text = f'{setting}: {value}'
        padding_item = (terminal_width - len(setting_text)) // 2
        print(' ' * padding_item + setting_text) 

    
    agent.train(num_steps=args.numsteps)