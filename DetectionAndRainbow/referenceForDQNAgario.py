"""
Simple DQN Agent for Agar.io
Uses PyTorch for neural network
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import time
import os
from pynput import keyboard

from rl_environment import AgarioEnv

# Global flag for Q press
stop_training = False
def on_press(key):
    global stop_training
    try:
        if key.char == 'q':
            stop_training = True
    except AttributeError:
        pass
key_listener = keyboard.Listener(on_press=on_press)
key_listener.start()


# ============================================================================
# NEURAL NETWORK
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
# REPLAY BUFFER
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
# DQN AGENT
# ============================================================================

class DQNAgent:
    """Deep Q-Network Agent"""
    
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
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
        
        # Epsilon for exploration
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Q-Networks
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size)
        
        # Step counter for target updates
        self.steps = 0
    
    def select_action(self, state):
        """Select action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.buffer.push(state, action, reward, next_state, done)
    
    def learn(self):
        """Sample batch and update Q-network"""
        if len(self.buffer) < self.batch_size:
            return None  # Not enough samples
        
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
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # Loss
        loss = nn.MSELoss()(current_q, target_q)
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network periodically
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, path="checkpoints/dqn_model.pth"):
        """Save model weights"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path="checkpoints/dqn_model.pth"):
        """Load model weights"""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        print(f"Model loaded from {path}")


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train(num_episodes=100, max_steps_per_episode=500, save_freq=10, monitor_y_offset=0, monitor_dims=None):
    """Train the DQN agent"""
    
    print("="*60)
    print("DQN TRAINING FOR AGAR.IO")
    print("="*60)
    print("Make sure Agar.io is open and ready!")
    print(f"Monitor Y-offset: {monitor_y_offset}")
    print("Press Q anytime to stop")
    print("Starting in 5 seconds...")
    time.sleep(5)
    
    # Create environment and agent
    env = AgarioEnv(step_delay=0.1, monitor_y_offset=monitor_y_offset, monitor_dims=monitor_dims)
    print(f"Capture region: {env.monitor}")
    print(f"State dim: {env.state_dim}, Action dim: {env.action_dim}")
    print("="*60)
    agent = DQNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim
    )
    
    # Training stats
    episode_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        losses = []
        
        action_names = ['up', 'down', 'left', 'right', 'up-right', 'down-right', 'up-left', 'down-left']
        for step in range(max_steps_per_episode):
            # Select and execute action
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            # Periodic debug print every 50 steps
            if step % 50 == 0:
                print(f"  [Step {step}] Score: {info['score']} | Action: {action_names[action]} | Reward: {reward:.1f}")
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Learn
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)
            
            episode_reward += reward
            state = next_state
            
            if done or stop_training:
                if stop_training:
                    print("\n[Q pressed - stopping training]")
                    return agent, episode_rewards
                break
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Log progress
        avg_loss = np.mean(losses) if losses else 0
        episode_rewards.append(episode_reward)
        avg_reward_last_10 = np.mean(episode_rewards[-10:])
        
        print(f"Episode {episode+1}/{num_episodes} | "
              f"Reward: {episode_reward:.1f} | "
              f"Avg(10): {avg_reward_last_10:.1f} | "
              f"Epsilon: {agent.epsilon:.3f} | "
              f"Loss: {avg_loss:.4f} | "
              f"Steps: {step+1}")
        
        # Save periodically
        if (episode + 1) % save_freq == 0:
            agent.save(f"checkpoints/dqn_model_ep{episode+1}.pth")
    
    # Final save
    agent.save("checkpoints/dqn_model_final.pth")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print(f"Final avg reward (last 10): {np.mean(episode_rewards[-10:]):.1f}")
    print("="*60)
    
    return agent, episode_rewards


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--external', action='store_true', help='Use external monitor above')
    args = parser.parse_args()
    
    # Auto-detect offset if external flag is set
    monitor_y_offset = 0
    monitor_dims = None  # (width, height, x_offset)
    if args.external:
        import mss
        with mss.mss() as sct:
            if len(sct.monitors) > 2:  # monitor[0] is "all", [1] is primary, [2]+ are external
                external = sct.monitors[2]
                monitor_y_offset = external['top']
                monitor_dims = (external['width'], external['height'], external['left'])
                print(f"\n{'='*60}")
                print(f"EXTERNAL MONITOR DETECTED")
                print(f"{'='*60}")
                print(f"Monitor index: 2")
                print(f"Dimensions: {external['width']}x{external['height']}")
                print(f"Position: top={external['top']}, left={external['left']}")
                print(f"Y-offset applied: {monitor_y_offset}")
                print(f"{'='*60}\n")
    
    agent, rewards = train(
        num_episodes=50,
        max_steps_per_episode=300,
        save_freq=10,
        monitor_y_offset=monitor_y_offset,
        monitor_dims=monitor_dims
    )