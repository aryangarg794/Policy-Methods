import torch.nn as nn
import numpy as np
import gymnasium as gym
import torch

from datetime import datetime
from policy_methods.reinforce import PolicyGradient
from policy_methods.utils import RollingAverage


class Critic(nn.Module):
    
    def __init__(
        self, 
        environ: gym.Env, 
        input_shape: int, 
        hidden_layers: list = [64],
        gamma: float = 0.99, 
        device: str = 'cpu',
        learning_rate: float = 0.01, 
        max_grad: float = 1.0, 
        *args, 
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        
        self.device = device
        self.input_shape = input_shape
        self.env = environ
        self.gamma = gamma
        self.max_grad = max_grad
        
        self.layers = nn.Sequential(
            nn.Linear(input_shape, hidden_layers[0]), 
            nn.ReLU()
        )
        
        for i in range(1, len(hidden_layers)):
            self.layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
            self.layers.append(nn.ReLU())
            
        self.layers.append(nn.Linear(hidden_layers[-1], environ.action_space.n))
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
    
    def forward(
        self,
        x: torch.tensor
    ) -> torch.tensor:
        x = x.flatten(start_dim=1) if len(x.shape) > 1 else x.flatten()
        return self.layers(x)
        
    def update_step(
        self, 
        state: torch.tensor, 
        next_state: torch.tensor, 
        action: torch.tensor, 
        next_action: torch.tensor,
        reward: torch.tensor,
        done: torch.tensor
    ) -> torch.tensor:
        value = self(state).gather(1, action)
        with torch.no_grad():
            next_value = self(next_state).gather(1, next_action)
        td_target = reward + self.gamma * next_value * (1 - done)
        
        self.optimizer.zero_grad()
        loss = torch.functional.F.mse_loss(value, td_target, reduction='none')
        loss.sum().backward()
        self.optimizer.step()
        
        return td_target - value

class QActorCritic:
    
    def __init__(
        self, 
        environ: gym.Env, 
        input_shape: int, 
        game_name: str, 
        h_layers_actor: list = [64],
        h_layers_critic: list = [64],
        gamma: float = 0.99,
        device: str = 'cpu',
        lr_critic: float = 0.01,
        lr_actor: float = 0.01, 
        max_grad: float = 1.5,
        *args, 
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        
        self._agent_name = 'ActorCritic'
        self.game_name = game_name
        self.device = device
        self.env = environ
        self.gamma = gamma
        
        self.actor = PolicyGradient(
            environ, 
            input_shape, 
            game_name,
            h_layers_actor,
            gamma, 
            device, 
            lr_actor,
            max_grad 
        ).to(device)
        
        self.critic = Critic(
            environ, 
            input_shape, 
            h_layers_critic,
            gamma, 
            device, 
            lr_critic,
            max_grad
        ).to(device)
        
        
    def train(
        self, 
        num_steps: int = 1000,
        save: bool = False
    ) -> list:
        
        avg = RollingAverage(20)

        buffer = {'states': [], 'next_states' : [], 'actions': [], 'next_actions' : [], 'rewards' : [], 'dones' : []}
        
        def update_buffer(state, next_state, action, next_action, reward, done):
            buffer['states'].append(state)
            buffer['next_states'].append(next_state)
            buffer['actions'].append(action)
            buffer['next_actions'].append(next_action)
            buffer['rewards'].append(reward)
            buffer['dones'].append(done)
        
        def reset_buffer():
            return {'states': [], 'next_states' : [], 'actions': [], 'next_actions' : [], 'rewards' : [], 'dones' : []}
        
        def get_batch():
            return buffer['states'], buffer['next_states'], buffer['actions'], buffer['next_actions'], buffer['rewards'], buffer['dones']
        
        for step in range(1, num_steps):
            obs, _ = self.env.reset()
            action = self.actor.select_action(obs)
            done = False
            ep_reward = 0 
            while not done:
                obs_prime, reward, terminated, truncated, _ = self.env.step(action)
                next_action = self.actor.select_action(obs_prime)     
                                
                ep_reward += reward
                done = terminated or truncated
                update_buffer(obs, obs_prime, action, next_action, reward, done)
                
                obs = obs_prime
                action = next_action
            
            batch_obs, batch_obs_prime, batch_actions, batch_actions_prime, batch_rewards, batch_dones = get_batch()
            
            targets = self.critic.update_step(
                torch.as_tensor(np.array(batch_obs), dtype=torch.float32, device=self.device).view(-1, 4), # placeholder 4
                torch.as_tensor(np.array(batch_obs_prime), dtype=torch.float32, device=self.device).view(-1, 4),
                torch.as_tensor(np.array(batch_actions), dtype=torch.int64, device=self.device).view(-1, 1),
                torch.as_tensor(np.array(batch_actions_prime), dtype=torch.int64, device=self.device).view(-1, 1),
                torch.as_tensor(np.array(batch_rewards), dtype=torch.float32, device=self.device).view(-1, 1),
                torch.as_tensor(np.array(batch_dones), dtype=torch.int64, device=self.device).view(-1, 1)
            )
            
            self.actor.update_step(
                buffer['states'], 
                buffer['actions'],
                targets.detach()
            )
        
            avg.update(ep_reward)
            print(f'Step: {step+1} | Avg Reward: {ep_reward:.1f}', end='\r')
            ep_reward = 0
            buffer = reset_buffer()

        if save:
            torch.save(self.actor.state_dict(), f'models/{self._agent_name}_{self.game_name.upper()}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}')

        return avg.averages