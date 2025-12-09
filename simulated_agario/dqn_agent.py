"""
Simple DQN Agent for Agar.io - Cross-Compatible Version
Uses PyTorch for neural network
EXACTLY matches the original dqn_agent.py hyperparameters and structure
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


# ============================================================================
# NEURAL NETWORK (matches original exactly)
# ============================================================================

class QNetwork(nn.Module):
    """Simple MLP Q-Network"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)


# ============================================================================
# REPLAY BUFFER (matches original exactly)
# ============================================================================

class ReplayBuffer:
    """Simple experience replay buffer"""
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


# ============================================================================
# DQN AGENT (matches original hyperparameters exactly)
# ============================================================================

class DQNAgent:
    """Deep Q-Network Agent - Cross-compatible with live game"""
    
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.3,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=32,
        target_update_freq=100
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Epsilon for exploration (matches original)
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Q-Networks (matches original architecture)
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size)
        
        # Step counter for target updates
        self.steps = 0
        
        # Training stats
        self.episodes = 0
        self.episode_rewards = []
        self.losses = []
        self.wins = []  # Track win/loss per episode (1 = win, 0 = loss)
        self.kills = []  # Track kills per episode
        self.total_wins = 0
    
    def select_action(self, state, training=True):
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.buffer.push(state, action, reward, next_state, done)
    
    def learn(self):
        """Sample batch and update Q-network (matches original exactly)"""
        if len(self.buffer) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q values (standard DQN, matches original)
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # Loss (MSE, matches original)
        loss = nn.MSELoss()(current_q, target_q)
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network periodically
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        loss_val = loss.item()
        self.losses.append(loss_val)
        return loss_val
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def end_episode(self, episode_reward, won=False, kills=0):
        """Called at end of episode"""
        self.episodes += 1
        self.episode_rewards.append(episode_reward)
        self.wins.append(1 if won else 0)
        self.kills.append(kills)
        if won:
            self.total_wins += 1
        self.decay_epsilon()
    
    def get_stats(self):
        """Get training statistics"""
        # Calculate win rate over last 100 episodes
        recent_wins = self.wins[-100:] if self.wins else []
        win_rate = sum(recent_wins) / len(recent_wins) * 100 if recent_wins else 0
        
        return {
            'episodes': self.episodes,
            'steps': self.steps,
            'epsilon': self.epsilon,
            'avg_reward_10': np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0,
            'avg_reward_100': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0,
            'buffer_size': len(self.buffer),
            'last_reward': self.episode_rewards[-1] if self.episode_rewards else 0,
            'total_wins': self.total_wins,
            'win_rate': win_rate,
            'last_won': self.wins[-1] if self.wins else 0
        }
    
    def save(self, path="dqn_model.pth"):
        """Save model weights and training history (compatible with original format)"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'episodes': self.episodes,
            'episode_rewards': self.episode_rewards,
            'wins': self.wins,
            'kills': self.kills,
            'total_wins': self.total_wins
        }, path)
        
        # Also save training history as CSV for easy graphing
        import csv
        history_path = path.replace('.pth', '_history.csv')
        with open(history_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'reward', 'won', 'kills'])
            for i in range(len(self.episode_rewards)):
                reward = self.episode_rewards[i]
                won = self.wins[i] if i < len(self.wins) else 0
                kills = self.kills[i] if i < len(self.kills) else 0
                writer.writerow([i + 1, reward, won, kills])
        print(f"Training history saved to {history_path}")
    
    def load(self, path="dqn_model.pth"):
        """Load model weights"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']


if __name__ == "__main__":
    # Quick test
    from game_env import AgarioSimEnv
    
    env = AgarioSimEnv()
    agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_dim)
    
    print(f"State dim: {env.state_dim}")
    print(f"Action dim: {env.action_dim}")
    print(f"Device: {agent.device}")
    print("Agent initialized successfully!")
