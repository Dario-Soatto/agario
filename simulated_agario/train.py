"""
Training Script for Agar.io Bot - OPTIMIZED FOR SPEED
Phase 1: DQN training against smart bots - ULTRA FAST
Phase 2: NFSP self-play training

The final model is cross-compatible with the live Agar.io game.
"""

import numpy as np
import time
import argparse
import os

from game_env import AgarioSimEnv, SelfPlayEnv
from dqn_agent import DQNAgent
from nfsp_agent import NFSPAgent


def train_dqn_vs_bots(num_episodes=5000, save_freq=500):
    """
    Phase 1: Train DQN agent against varied bot opponents
    OPTIMIZED FOR MAXIMUM SPEED
    
    Args:
        num_episodes: Number of training episodes (default: 5000)
        save_freq: How often to save checkpoints
    """
    print("=" * 70)
    print("PHASE 1: DQN TRAINING VS SMART BOTS [ULTRA FAST MODE]")
    print("=" * 70)
    print(f"Episodes: {num_episodes}")
    print(f"Optimizations: No delays, minimal logging, fast stepping")
    print("=" * 70)
    
    # Create environment and agent
    env = AgarioSimEnv()
    agent = DQNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        lr=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.3,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=32,
        target_update_freq=100
    )
    
    print(f"State dim: {env.state_dim}, Action dim: {env.action_dim}, Device: {agent.device}")
    print("=" * 70)
    
    # Training stats
    start_time = time.time()
    episode_rewards = []
    episode_lengths = []
    best_avg_reward = float('-inf')
    
    # Tracking for speed display
    last_print_time = start_time
    last_print_episode = 0
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        step = 0
        
        # Run until player dies (no step limit, no delays)
        while True:
            step += 1
            
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            agent.store_transition(state, action, reward, next_state, done)
            agent.learn()
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        won = info.get('won', False)
        kills = info.get('episode_kills', 0)
        agent.end_episode(episode_reward, won=won, kills=kills)
        episode_rewards.append(episode_reward)
        episode_lengths.append(step)
        
        # Logging - much less frequent for speed
        if (episode + 1) % 50 == 0:
            current_time = time.time()
            elapsed = current_time - start_time
            recent_time = current_time - last_print_time
            episodes_since_print = episode + 1 - last_print_episode
            
            avg_10 = np.mean(episode_rewards[-10:])
            avg_100 = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else avg_10
            avg_len = np.mean(episode_lengths[-50:])
            
            # Calculate speeds
            overall_eps_per_sec = (episode + 1) / elapsed
            recent_eps_per_sec = episodes_since_print / recent_time if recent_time > 0 else 0
            
            print(f"Ep {episode+1:5d}/{num_episodes} | "
                  f"Rew: {episode_reward:6.1f} | "
                  f"Avg100: {avg_100:6.1f} | "
                  f"Eps: {agent.epsilon:.3f} | "
                  f"Len: {avg_len:.0f} | "
                  f"Speed: {recent_eps_per_sec:.1f} ep/s | "
                  f"K/D: {env.total_kills}/{env.total_deaths}")
            
            last_print_time = current_time
            last_print_episode = episode + 1
            
            # Save best model
            if avg_100 > best_avg_reward and len(episode_rewards) >= 100:
                best_avg_reward = avg_100
                agent.save("dqn_best.pth")
        
        # Periodic save
        if (episode + 1) % save_freq == 0:
            agent.save(f"dqn_ep{episode+1}.pth")
            print(f"[CHECKPOINT] Saved dqn_ep{episode+1}.pth")
    
    # Final save
    agent.save("dqn_final.pth")
    
    elapsed = time.time() - start_time
    total_steps = sum(episode_lengths)
    
    print("\n" + "=" * 70)
    print(f"PHASE 1 COMPLETE")
    print(f"Time: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)")
    print(f"Speed: {num_episodes/elapsed:.2f} episodes/sec")
    print(f"Steps: {total_steps} total ({total_steps/elapsed:.1f} steps/sec)")
    print(f"Final avg reward (last 100): {np.mean(episode_rewards[-100:]):.1f}")
    print(f"Best avg reward: {best_avg_reward:.1f}")
    print(f"Avg episode length: {np.mean(episode_lengths):.0f} steps")
    print(f"Total kills: {env.total_kills}, Total deaths: {env.total_deaths}")
    print("=" * 70)
    
    return agent


def train_nfsp_selfplay(num_episodes=5000, save_freq=500, 
                        pretrained_path=None, eta=0.1):
    """
    Phase 2: NFSP self-play training - OPTIMIZED FOR SPEED
    
    Args:
        num_episodes: Number of training episodes
        save_freq: How often to save checkpoints
        pretrained_path: Path to pretrained DQN model to initialize from
        eta: Anticipatory parameter (prob of using best response)
    """
    print("\n" + "=" * 70)
    print("PHASE 2: NFSP SELF-PLAY TRAINING [ULTRA FAST MODE]")
    print("=" * 70)
    print(f"Episodes: {num_episodes}, Eta: {eta}")
    print("=" * 70)
    
    env = AgarioSimEnv()
    
    # Create NFSP agents
    agent1 = NFSPAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        eta=eta,
        rl_lr=0.001,
        sl_lr=0.001,
        gamma=0.99,
        epsilon_start=0.1,
        epsilon_end=0.01,
        epsilon_decay=0.9999,
        rl_buffer_size=200000,
        sl_buffer_size=2000000,
        batch_size=128,
        target_update_freq=300
    )
    
    agent2 = NFSPAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        eta=eta,
        rl_lr=0.001,
        sl_lr=0.001,
        gamma=0.99,
        epsilon_start=0.1,
        epsilon_end=0.01,
        epsilon_decay=0.9999,
        rl_buffer_size=200000,
        sl_buffer_size=2000000,
        batch_size=128,
        target_update_freq=300
    )
    
    # Load pretrained weights if available
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading pretrained model from {pretrained_path}")
        checkpoint = __import__('torch').load(pretrained_path, map_location=agent1.device)
        agent1.q_network.load_state_dict(checkpoint['q_network'])
        agent1.target_network.load_state_dict(checkpoint['target_network'])
        agent2.q_network.load_state_dict(checkpoint['q_network'])
        agent2.target_network.load_state_dict(checkpoint['target_network'])
    
    print(f"Device: {agent1.device}")
    print("=" * 70)
    
    selfplay_env = SelfPlayEnv(opponent_model=agent2)
    
    start_time = time.time()
    episode_rewards = []
    episode_lengths = []
    best_avg_reward = float('-inf')
    
    last_print_time = start_time
    last_print_episode = 0
    
    for episode in range(num_episodes):
        agent1.begin_episode()
        agent2.begin_episode()
        
        state = selfplay_env.reset()
        episode_reward = 0
        step = 0
        
        while True:
            step += 1
            
            action = agent1.select_action(state)
            next_state, reward, done, info = selfplay_env.step(action)
            
            agent1.store_transition(state, action, reward, next_state, done)
            
            if selfplay_env.opponent is not None:
                opp_state = selfplay_env._get_opponent_state()
                agent2.store_transition(opp_state, 0, -reward * 0.5, opp_state, done)
            
            agent1.learn()
            agent2.learn()
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        won = info.get('won', False)
        kills = info.get('episode_kills', 0)
        agent1.end_episode(episode_reward, won=won, kills=kills)
        agent2.end_episode(-episode_reward, won=not won, kills=0)
        episode_rewards.append(episode_reward)
        episode_lengths.append(step)
        
        # Less frequent logging
        if (episode + 1) % 50 == 0:
            current_time = time.time()
            elapsed = current_time - start_time
            recent_time = current_time - last_print_time
            episodes_since_print = episode + 1 - last_print_episode
            
            avg_10 = np.mean(episode_rewards[-10:])
            avg_100 = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else avg_10
            stats1 = agent1.get_stats()
            recent_eps_per_sec = episodes_since_print / recent_time if recent_time > 0 else 0
            
            print(f"Ep {episode+1:5d}/{num_episodes} | "
                  f"Rew: {episode_reward:6.1f} | "
                  f"Avg100: {avg_100:6.1f} | "
                  f"RL_Loss: {stats1['avg_rl_loss']:.4f} | "
                  f"SL_Loss: {stats1['avg_sl_loss']:.4f} | "
                  f"Speed: {recent_eps_per_sec:.1f} ep/s")
            
            last_print_time = current_time
            last_print_episode = episode + 1
            
            if avg_100 > best_avg_reward and len(episode_rewards) >= 100:
                best_avg_reward = avg_100
                agent1.save("nfsp_best.pth")
        
        if (episode + 1) % save_freq == 0:
            agent1.save(f"nfsp_ep{episode+1}.pth")
            print(f"[CHECKPOINT] Saved nfsp_ep{episode+1}.pth")
            if (episode + 1) % (save_freq * 2) == 0:
                agent2.q_network.load_state_dict(agent1.q_network.state_dict())
                agent2.policy_network.load_state_dict(agent1.policy_network.state_dict())
    
    agent1.save("nfsp_final.pth")
    agent1.export_for_live_game("live_model.pth")
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"PHASE 2 COMPLETE")
    print(f"Time: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)")
    print(f"Speed: {num_episodes/elapsed:.2f} episodes/sec")
    print(f"Final avg reward (last 100): {np.mean(episode_rewards[-100:]):.1f}")
    print(f"Best avg reward: {best_avg_reward:.1f}")
    print("=" * 70)
    
    return agent1


def full_training_pipeline(dqn_episodes=5000, nfsp_episodes=5000):
    """Full training pipeline - ULTRA FAST"""
    print("\n")
    print("█" * 70)
    print("█  AGAR.IO BOT TRAINING PIPELINE [ULTRA FAST MODE]")
    print("█  Phase 1: DQN vs Smart Bots")
    print("█  Phase 2: NFSP Self-Play")
    print("█" * 70)
    print("\n")
    
    overall_start = time.time()
    
    dqn_agent = train_dqn_vs_bots(
        num_episodes=dqn_episodes,
        save_freq=500
    )
    
    nfsp_agent = train_nfsp_selfplay(
        num_episodes=nfsp_episodes,
        save_freq=500,
        pretrained_path="dqn_final.pth",
        eta=0.1
    )
    
    total_time = time.time() - overall_start
    
    print("\n")
    print("█" * 70)
    print("█  TRAINING COMPLETE!")
    print(f"█  Total time: {total_time/60:.2f} minutes")
    print(f"█  Total episodes: {dqn_episodes + nfsp_episodes}")
    print(f"█  Average speed: {(dqn_episodes + nfsp_episodes)/total_time:.2f} ep/s")
    print("█  Models: dqn_final.pth, nfsp_final.pth, live_model.pth")
    print("█" * 70)
    
    return nfsp_agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Agar.io Bot - ULTRA FAST')
    parser.add_argument('--phase', type=str, default='full', 
                       choices=['dqn', 'nfsp', 'full'],
                       help='Training phase to run (dqn, nfsp, or full)')
    parser.add_argument('--dqn-episodes', type=int, default=5000,
                       help='Number of DQN training episodes')
    parser.add_argument('--nfsp-episodes', type=int, default=5000,
                       help='Number of NFSP training episodes')
    parser.add_argument('--pretrained', type=str, default=None,
                       help='Path to pretrained model for NFSP')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("OPTIMIZATION MODE: MAXIMUM SPEED")
    print("- No delays or sleeps")
    print("- Minimal logging (every 50 episodes)")
    print("- Fast environment stepping")
    print("- Optimized for throughput")
    print("=" * 70)
    print()
    
    if args.phase == 'dqn':
        train_dqn_vs_bots(num_episodes=args.dqn_episodes)
    elif args.phase == 'nfsp':
        train_nfsp_selfplay(
            num_episodes=args.nfsp_episodes,
            pretrained_path=args.pretrained or "dqn_final.pth"
        )
    else:
        full_training_pipeline(
            dqn_episodes=args.dqn_episodes,
            nfsp_episodes=args.nfsp_episodes
        )
