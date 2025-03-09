import torch.nn as nn
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
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=0.001)
        
        self.apply(self._init_weights)
    
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
        done: bool
    ) -> torch.tensor:
        value = self(state)[:, action]
        with torch.no_grad():
            next_value = self(next_state)[:, next_action]
        td_target = reward + self.gamma * next_value * (1 - done)
        
        self.optimizer.zero_grad()
        loss = torch.functional.F.mse_loss(value, td_target, reduction='none').sum()
        loss.backward()
        self.optimizer.step()
        
        return td_target - value
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.orthogonal_(m.weight, gain=1.0)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

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
        
        obs, _ = self.env.reset()
        action = self.actor.select_action(obs)
        ep_reward = 0
        for step in range(1, num_steps):
            obs_prime, reward, terminated, truncated, _ = self.env.step(action)
            
            ep_reward += reward
            next_action = self.actor.select_action(obs_prime)
            
            target = self.critic.update_step(
                torch.as_tensor(obs, device=self.device).unsqueeze(dim=0),
                torch.as_tensor(obs_prime, device=self.device).unsqueeze(dim=0),
                torch.as_tensor(action, device=self.device).unsqueeze(dim=0),
                torch.as_tensor(next_action, device=self.device).unsqueeze(dim=0),
                torch.as_tensor(reward, device=self.device).unsqueeze(dim=0),
                terminated or truncated
            )
                
            
            self.actor.update_step(
                [obs], 
                [action],
                target.detach()
            )
            
            obs = obs_prime
            action = next_action
            if terminated or truncated: 
                obs, _ = self.env.reset()
                action = self.actor.select_action(obs)
                avg.update(ep_reward)
                ep_reward = 0
                
                print(f'Step: {step+1} | Avg Reward: {avg.get_average:.1f}', end='\r')
            
        if save:
            torch.save(self.actor.state_dict(), f'../models/{self._agent_name}_{self.game_name.upper()}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}')

        return avg.averages