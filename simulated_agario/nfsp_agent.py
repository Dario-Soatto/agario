"""
Neural Fictitious Self-Play (NFSP) Agent for Agar.io
Combines Reinforcement Learning and Supervised Learning for self-play training.

NFSP Components:
1. Q-Network (RL) - learns best response via DQN
2. Policy Network (SL) - learns average strategy from past actions
3. RL Buffer - standard experience replay for Q-learning
4. SL Buffer - reservoir sampling of (state, action) pairs
5. Anticipatory parameter η - probability of using best response vs average strategy
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque


# ============================================================================
# NETWORKS
# ============================================================================

class QNetwork(nn.Module):
    """Q-Network for reinforcement learning (best response)"""
    
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


class PolicyNetwork(nn.Module):
    """Policy Network for supervised learning (average strategy)"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return F.softmax(self.network(x), dim=-1)
    
    def get_action_probs(self, state):
        """Get action probabilities"""
        with torch.no_grad():
            probs = self.forward(state)
        return probs


# ============================================================================
# BUFFERS
# ============================================================================

class RLBuffer:
    """Experience replay buffer for Q-learning"""
    
    def __init__(self, capacity=200000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
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


class SLBuffer:
    """
    Reservoir sampling buffer for supervised learning
    Stores (state, action) pairs from best response actions
    """
    
    def __init__(self, capacity=2000000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.total_seen = 0
    
    def push(self, state, action):
        """Add (state, action) pair using reservoir sampling"""
        self.total_seen += 1
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action))
        else:
            # Reservoir sampling
            idx = random.randint(0, self.total_seen - 1)
            if idx < self.capacity:
                self.buffer[idx] = (state, action)
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions = zip(*batch)
        return np.array(states), np.array(actions)
    
    def __len__(self):
        return len(self.buffer)


# ============================================================================
# NFSP AGENT
# ============================================================================

class NFSPAgent:
    """
    Neural Fictitious Self-Play Agent
    
    Uses anticipatory dynamics:
    - With probability η: use best response (Q-network)
    - With probability 1-η: use average strategy (Policy network)
    """
    
    def __init__(
        self,
        state_dim,
        action_dim,
        # RL parameters
        rl_lr=0.001,
        gamma=0.99,
        epsilon_start=0.1,  # Lower than DQN since we also explore via η
        epsilon_end=0.01,
        epsilon_decay=0.9999,
        # SL parameters
        sl_lr=0.001,
        # NFSP parameters
        eta=0.1,  # Anticipatory parameter (prob of best response)
        rl_buffer_size=200000,
        sl_buffer_size=2000000,
        batch_size=128,
        target_update_freq=300,
        # Training modes
        min_rl_buffer=1000,
        min_sl_buffer=1000,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.eta = eta
        self.min_rl_buffer = min_rl_buffer
        self.min_sl_buffer = min_sl_buffer
        
        # Epsilon for RL exploration
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Q-Networks (for best response)
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=rl_lr)
        
        # Policy Network (for average strategy)
        self.policy_network = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=sl_lr)
        
        # Buffers
        self.rl_buffer = RLBuffer(rl_buffer_size)
        self.sl_buffer = SLBuffer(sl_buffer_size)
        
        # Counters
        self.steps = 0
        self.episodes = 0
        self.episode_rewards = []
        self.rl_losses = []
        self.sl_losses = []
        self.wins = []  # Track win/loss per episode (1 = win, 0 = loss)
        self.kills = []  # Track kills per episode
        self.total_wins = 0
        
        # Current policy mode for this episode
        self.use_best_response = True
    
    def begin_episode(self):
        """Called at start of episode to set policy mode"""
        # With probability η, use best response; otherwise use average strategy
        self.use_best_response = random.random() < self.eta
    
    def select_action(self, state, training=True):
        """
        Select action based on current policy mode
        
        During training:
        - If using best response: ε-greedy over Q-values
        - If using average strategy: sample from policy network
        
        During evaluation:
        - Always use average strategy (the equilibrium approximation)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if not training:
            # Evaluation: use average strategy
            with torch.no_grad():
                probs = self.policy_network(state_tensor).cpu().numpy()[0]
            return np.random.choice(self.action_dim, p=probs)
        
        if self.use_best_response:
            # Best response with ε-greedy exploration
            if random.random() < self.epsilon:
                return random.randint(0, self.action_dim - 1)
            
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
        else:
            # Average strategy: sample from policy
            with torch.no_grad():
                probs = self.policy_network(state_tensor).cpu().numpy()[0]
            return np.random.choice(self.action_dim, p=probs)
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store experience in appropriate buffers"""
        # Always store in RL buffer
        self.rl_buffer.push(state, action, reward, next_state, done)
        
        # Only store in SL buffer if using best response
        # (we learn average strategy from best response actions)
        if self.use_best_response:
            self.sl_buffer.push(state, action)
    
    def learn_rl(self):
        """Update Q-network (reinforcement learning)"""
        if len(self.rl_buffer) < self.min_rl_buffer:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.rl_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # Loss and update
        loss = nn.MSELoss()(current_q, target_q)
        
        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        loss_val = loss.item()
        self.rl_losses.append(loss_val)
        return loss_val
    
    def learn_sl(self):
        """Update policy network (supervised learning on past actions)"""
        if len(self.sl_buffer) < self.min_sl_buffer:
            return None
        
        # Sample batch
        states, actions = self.sl_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        
        # Get action probabilities
        probs = self.policy_network(states)
        
        # Cross-entropy loss (negative log likelihood)
        loss = F.cross_entropy(probs.log(), actions)
        
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()
        
        loss_val = loss.item()
        self.sl_losses.append(loss_val)
        return loss_val
    
    def learn(self):
        """Combined learning step"""
        rl_loss = self.learn_rl()
        sl_loss = self.learn_sl()
        return rl_loss, sl_loss
    
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
            'eta': self.eta,
            'avg_reward_10': np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0,
            'avg_reward_100': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            'avg_rl_loss': np.mean(self.rl_losses[-100:]) if self.rl_losses else 0,
            'avg_sl_loss': np.mean(self.sl_losses[-100:]) if self.sl_losses else 0,
            'rl_buffer_size': len(self.rl_buffer),
            'sl_buffer_size': len(self.sl_buffer),
            'last_reward': self.episode_rewards[-1] if self.episode_rewards else 0,
            'using_best_response': self.use_best_response,
            'total_wins': self.total_wins,
            'win_rate': win_rate,
            'last_won': self.wins[-1] if self.wins else 0
        }
    
    def save(self, path="nfsp_model.pth"):
        """Save model weights and training history"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'policy_network': self.policy_network.state_dict(),
            'q_optimizer': self.q_optimizer.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
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
    
    def load(self, path="nfsp_model.pth"):
        """Load model weights"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.policy_network.load_state_dict(checkpoint['policy_network'])
        self.q_optimizer.load_state_dict(checkpoint['q_optimizer'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.episodes = checkpoint.get('episodes', 0)
    
    def export_for_live_game(self, path="live_model.pth"):
        """
        Export model in format compatible with original dqn_agent.py
        Uses the Q-network for greedy action selection in live play
        """
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.q_optimizer.state_dict(),
            'epsilon': 0.05,  # Low epsilon for live play
            'steps': self.steps
        }, path)
        print(f"Model exported for live game: {path}")


if __name__ == "__main__":
    from game_env import AgarioSimEnv
    
    env = AgarioSimEnv()
    agent = NFSPAgent(state_dim=env.state_dim, action_dim=env.action_dim)
    
    print(f"NFSP Agent initialized")
    print(f"State dim: {env.state_dim}")
    print(f"Action dim: {env.action_dim}")
    print(f"Device: {agent.device}")
    print(f"Eta (best response prob): {agent.eta}")

