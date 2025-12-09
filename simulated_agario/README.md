# Agar.io Trainer - DQN/NFSP - ULTRA FAST MODE

Cross-compatible reinforcement learning bot for Agar.io with optimized training.

## 🚀 Quick Start - Ultra Fast Training

For maximum training speed (thousands of episodes):

```bash
cd simulated_agario

# Train DQN only (5000 episodes - should complete in ~5-15 minutes)
python train.py --phase dqn --dqn-episodes 5000

# Train full pipeline (10000 total episodes)
python train.py --dqn-episodes 5000 --nfsp-episodes 5000

# Custom episode counts
python train.py --phase dqn --dqn-episodes 10000
```

## ⚡ Speed Optimizations

The command-line trainer is heavily optimized for speed:

### Training Script Optimizations:
- ❌ No delays or sleep statements
- 📊 Minimal logging (every 50 episodes instead of 10)
- ⚡ Real-time speed tracking (episodes/second)
- 💾 Efficient checkpointing

### Environment Optimizations:
- 🔢 Squared distance calculations (avoids sqrt)
- 📦 `__slots__` for all classes (reduced memory)
- 🎯 Inline collision detection
- 🔄 Pre-computed constants
- 🚀 Optimized loops and indexing
- 📈 Cached score calculations

Expected speeds on modern hardware:
- **2-10 episodes/second** during early training (short episodes)
- **0.5-2 episodes/second** later (longer survival times)
- **5000 episodes in ~10-20 minutes** (varies by hardware)

## 🎮 Web Visualization (Slower, for Monitoring)

For watching the agent train in real-time:

```bash
python app.py
# Open http://localhost:3000
# Use speed slider (up to 100x)
```

Note: Web visualization is slower due to rendering overhead. Use command-line training for maximum speed.

## 🎯 Game Configuration

Current settings optimized for fast learning:

| Setting | Value | Reason |
|---------|-------|--------|
| Map Size | 3000 x 2000 | Large exploration space |
| Player Start | 15 radius | Smallest (challenging) |
| Bots | 5 total | Varied sizes & behaviors |
| Food | 150 particles | Fast growth |
| Move Speed | 10 | Quick navigation |
| Food Growth | 3x radius² | Rapid size increases |

## 📊 Output Files

After training:
- `dqn_final.pth` - DQN model after phase 1
- `dqn_best.pth` - Best performing DQN model (auto-saved when avg reward improves)
- `nfsp_final.pth` - NFSP model after phase 2
- `nfsp_best.pth` - Best performing NFSP model
- `live_model.pth` - **Ready for live game!** (exported from NFSP)

## 🔄 Using Trained Model in Live Game

```bash
# Copy the model to main directory
cp simulated_agario/live_model.pth ./

# In your live game script:
agent.load('live_model.pth')
```

The model is **cross-compatible** - trained in simulation, works in live game!

## 🧠 State Space (24-dim, cross-compatible)

```
[0-2]   Self: x, y, radius
[3]     Score (current size)
[4-12]  3 nearest viruses (always -1 in sim)
[13-21] 3 nearest players/bots
[22]    Food count (always 0 - invisible to agent)
[23]    Game ended flag
```

## 🤖 Bot Behaviors

The agent trains against 5 bots with different AI:
1. **Tiny** (8-12) - flee, zigzag
2. **Small** (12-18) - flee, patrol
3. **Medium** (20-35) - smart, chase
4. **Large** (40-60) - chase, smart
5. **Boss** (70-90) - patrol, smart

## 📈 Training Tips

1. **For speed**: Use command-line training
2. **For monitoring**: Use web UI at lower speed
3. **Checkpoints**: Models saved every 500 episodes
4. **Best models**: Automatically saved when avg reward improves

## 🔧 Requirements

```bash
pip install torch numpy flask
```

## 📝 Example Training Session

```bash
$ python train.py --phase dqn --dqn-episodes 10000

======================================================================
OPTIMIZATION MODE: MAXIMUM SPEED
- No delays or sleeps
- Minimal logging (every 50 episodes)
- Fast environment stepping
- Optimized for throughput
======================================================================

======================================================================
PHASE 1: DQN TRAINING VS SMART BOTS [ULTRA FAST MODE]
======================================================================
Episodes: 10000
Optimizations: No delays, minimal logging, fast stepping
======================================================================
State dim: 24, Action dim: 8, Device: cpu
======================================================================

Ep    50/10000 | Rew:   85.3 | Avg100:   67.2 | Eps: 0.951 | Len: 245 | Speed: 3.2 ep/s | K/D: 12/8
Ep   100/10000 | Rew:  102.7 | Avg100:   78.1 | Eps: 0.904 | Len: 312 | Speed: 2.8 ep/s | K/D: 28/15
...
```

---

**Pro Tip**: Training is CPU-bound. Close other applications for maximum speed!

