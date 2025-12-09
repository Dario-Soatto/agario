"""
Training History Visualization Script
Reads the CSV training history and generates informative graphs.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving files

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def load_data(csv_path):
    """Load training history from CSV"""
    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} episodes from {csv_path}")
    return df

def rolling_average(data, window=100):
    """Calculate rolling average"""
    return pd.Series(data).rolling(window=window, min_periods=1).mean()

def create_plots(df, output_prefix="training"):
    """Create and save training visualization plots"""
    
    # Set up the style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax1 = plt.subplots(figsize=(14, 8))
    fig.suptitle('Agar.io Training Progress: Win Rate & Kills Over Time', fontsize=16, fontweight='bold')
    
    episodes = df['episode']
    wins = df['won']
    
    # Check if kills column exists (for older CSVs)
    has_kills = 'kills' in df.columns
    if has_kills:
        kills = df['kills']
    
    # Primary y-axis: Win Rate
    win_rate_100 = rolling_average(wins, 100) * 100
    win_rate_500 = rolling_average(wins, 500) * 100
    
    line1 = ax1.plot(episodes, win_rate_100, color='#27ae60', label='Win Rate (100-ep avg)', linewidth=2.5)
    line2 = ax1.plot(episodes, win_rate_500, color='#8e44ad', label='Win Rate (500-ep avg)', linewidth=2.5, linestyle='--')
    
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Win Rate (%)', fontsize=12, color='#27ae60')
    ax1.set_ylim(0, 100)
    ax1.tick_params(axis='y', labelcolor='#27ae60')
    ax1.grid(True, alpha=0.3)
    
    # Secondary y-axis: Kills
    if has_kills:
        ax2 = ax1.twinx()
        
        kills_100 = rolling_average(kills, 100)
        kills_500 = rolling_average(kills, 500)
        
        line3 = ax2.plot(episodes, kills_100, color='#e74c3c', label='Kills per Episode (100-ep avg)', linewidth=2.5, alpha=0.8)
        line4 = ax2.plot(episodes, kills_500, color='#c0392b', label='Kills per Episode (500-ep avg)', linewidth=2.5, linestyle='--', alpha=0.8)
        
        ax2.set_ylabel('Kills per Episode', fontsize=12, color='#e74c3c')
        ax2.tick_params(axis='y', labelcolor='#e74c3c')
        
        # Set reasonable y-limit for kills (0 to max + 20%)
        max_kills = kills_100.max()
        if not np.isnan(max_kills):
            ax2.set_ylim(0, max_kills * 1.2)
        
        # Combine legends
        lines = line1 + line2 + line3 + line4
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', fontsize=10)
    else:
        # No kills data, just show win rate legend
        ax1.legend(loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = f"{output_prefix}_plots.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved plot to: {output_path}")
    
    # Try to show the plot (works in GUI environments)
    try:
        plt.show()
    except:
        pass  # Skip if no display available
    
    plt.close()
    return output_path

def print_summary(df):
    """Print training summary statistics"""
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    
    total_episodes = len(df)
    total_wins = df['won'].sum()
    win_rate = (total_wins / total_episodes) * 100 if total_episodes > 0 else 0
    
    print(f"Total Episodes:     {total_episodes:,}")
    print(f"Total Wins:         {int(total_wins):,}")
    print(f"Overall Win Rate:   {win_rate:.2f}%")
    print()
    
    # Kills statistics if available
    if 'kills' in df.columns:
        total_kills = df['kills'].sum()
        avg_kills = df['kills'].mean()
        max_kills = df['kills'].max()
        print(f"Kill Statistics:")
        print(f"  Total Kills:      {int(total_kills):,}")
        print(f"  Avg per Episode:  {avg_kills:.2f}")
        print(f"  Max in Episode:   {int(max_kills)}")
        print()
    
    print(f"Reward Statistics:")
    print(f"  Mean:             {df['reward'].mean():.2f}")
    print(f"  Std Dev:          {df['reward'].std():.2f}")
    print(f"  Min:              {df['reward'].min():.2f}")
    print(f"  Max:              {df['reward'].max():.2f}")
    print()
    
    # First vs last 500 episodes comparison
    if total_episodes >= 1000:
        first_500 = df.head(500)
        last_500 = df.tail(500)
        
        print("Progress Comparison (First 500 vs Last 500 episodes):")
        print(f"  Avg Reward:       {first_500['reward'].mean():.2f} → {last_500['reward'].mean():.2f}")
        print(f"  Win Rate:         {first_500['won'].mean()*100:.2f}% → {last_500['won'].mean()*100:.2f}%")
        
        if 'kills' in df.columns:
            print(f"  Avg Kills:        {first_500['kills'].mean():.2f} → {last_500['kills'].mean():.2f}")
        
        improvement = (last_500['reward'].mean() - first_500['reward'].mean())
        win_improvement = (last_500['won'].mean() - first_500['won'].mean())*100
        print(f"  Improvement:      {improvement:.2f} reward, {win_improvement:.2f}% win rate")
    
    print("="*50)

def main():
    # Default CSV path
    default_path = "dqn_model_history.csv"
    
    # Check command line argument
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = default_path
    
    # Look for CSV in current directory or simulated_agario folder
    if not os.path.exists(csv_path):
        alt_path = os.path.join(os.path.dirname(__file__), csv_path)
        if os.path.exists(alt_path):
            csv_path = alt_path
    
    print("="*50)
    print("AGAR.IO TRAINING HISTORY ANALYZER")
    print("="*50)
    
    # Load and process data
    df = load_data(csv_path)
    
    # Print summary
    print_summary(df)
    
    # Create and save plots
    output_prefix = os.path.splitext(csv_path)[0]
    create_plots(df, output_prefix)

if __name__ == "__main__":
    main()

