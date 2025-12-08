"""
Test Dueling DQN for Agar.io (from dqnAgario.py)
=================================================
Loads checkpoint and runs 5 test episodes with NO exploration.
Logs detailed CSV per step, tracks avg score & time alive.

Architecture: Dueling DQN (26-dim state, 5 semantic actions)
Press Q to quit
"""

import torch
import torch.nn as nn
import numpy as np
import time
import os
import csv
import glob
import re
from datetime import datetime
from pynput import keyboard
import cv2
import mss
import pyautogui

from detectStart import mssStartButton
from detectScore import mssBottomLeftCorner
from detectContinue import mssMiddleRectangle

# ============== Q TO QUIT ==============
stop_test = False
def on_press(key):
    global stop_test
    try:
        if key.char == 'q':
            stop_test = True
            print("\n[Q pressed] Stopping test...")
    except:
        pass
keyboard.Listener(on_press=on_press).start()


# ============== CSV LOGGER ==============
class TestCSVLogger:
    def __init__(self, checkpoint_name):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"dqn_test_{checkpoint_name}_{timestamp}.csv"
        self.file = open(self.filename, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            'timestamp', 'episode', 'step',
            'action', 'action_name',
            'q_eat', 'q_hunt', 'q_flee', 'q_tovirus', 'q_fromvirus', 'q_chosen',
            'self_radius', 'food_count', 'player_count',
            'nearest_food_dx', 'nearest_food_dy',
            'nearest_player_dx', 'nearest_player_dy', 'nearest_player_r',
            'has_virus', 'steps_alive', 'time_alive_sec', 'score'
        ])
        self.file.flush()
    
    def log(self, ep, step, action, action_name, q_values, env, time_alive, score):
        # Extract state info from env
        food_dx = env.last_food[0][0] if env.last_food else 0
        food_dy = env.last_food[0][1] if env.last_food else 0
        player_dx = env.last_players[0][0] if env.last_players else 0
        player_dy = env.last_players[0][1] if env.last_players else 0
        player_r = env.last_players[0][2] if env.last_players else 0
        
        self.writer.writerow([
            datetime.now().isoformat(), ep, step,
            action, action_name,
            f"{q_values[0]:.4f}", f"{q_values[1]:.4f}", f"{q_values[2]:.4f}",
            f"{q_values[3]:.4f}", f"{q_values[4]:.4f}", f"{q_values[action]:.4f}",
            env.self_radius, env.food_count, env.player_count,
            f"{food_dx:.1f}", f"{food_dy:.1f}",
            f"{player_dx:.1f}", f"{player_dy:.1f}", f"{player_r:.1f}",
            1 if env.last_virus else 0, step, f"{time_alive:.2f}", score
        ])
        self.file.flush()
    
    def close(self):
        self.file.close()


# ============== UNIFIED DETECTOR (from dqnAgario.py) ==============
class UnifiedDetector:
    VIRUS_BGR = (72, 244, 18)
    VIRUS_TOL = 40
    
    def __init__(self):
        self.sct = mss.mss()
        mon = self.sct.monitors[2] if len(self.sct.monitors) > 2 else self.sct.monitors[1]
        pad = int(mon['width'] * 0.05)
        self.region = {
            "top": mon['top'] + pad, "left": mon['left'] + pad,
            "width": mon['width'] - 2*pad, "height": mon['height'] - 2*pad
        }
        self.w = self.region['width']
        self.h = self.region['height']
        self.cx = self.w // 2
        self.cy = self.h // 2
    
    def detect_all(self):
        raw = np.array(self.sct.grab(self.region), dtype=np.uint8)
        img = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 30, 30]), np.array([180, 255, 255]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        self_blob = None
        players = []
        food = []
        viruses = []
        
        for c in contours:
            area = cv2.contourArea(c)
            if area < 10:
                continue
            (x, y), r = cv2.minEnclosingCircle(c)
            x, y, r = int(x), int(y), int(r)
            
            is_virus = False
            if 0 <= y < self.h and 0 <= x < self.w:
                px = img[y, x]
                if (abs(int(px[0]) - self.VIRUS_BGR[0]) < self.VIRUS_TOL and
                    abs(int(px[1]) - self.VIRUS_BGR[1]) < self.VIRUS_TOL and
                    abs(int(px[2]) - self.VIRUS_BGR[2]) < self.VIRUS_TOL):
                    is_virus = True
                    if r > 30:
                        dist = ((x - self.cx)**2 + (y - self.cy)**2)**0.5
                        viruses.append((x - self.cx, y - self.cy, dist))
            
            if is_virus:
                continue
            
            dist = ((x - self.cx)**2 + (y - self.cy)**2)**0.5
            
            if r >= 10 and dist < min(self.w, self.h) * 0.15:
                if self_blob is None or dist < self_blob[3]:
                    self_blob = (x, y, r, dist)
            elif r >= 30:
                players.append((x - self.cx, y - self.cy, r, dist))
            elif 2 <= r < 30:
                food.append((x - self.cx, y - self.cy, dist))
        
        players.sort(key=lambda p: p[3])
        food.sort(key=lambda f: f[2])
        viruses.sort(key=lambda v: v[2])
        
        self_radius = self_blob[2] if self_blob else None
        players_out = [(p[0], p[1], p[2]) for p in players[:3]]
        food_out = [(f[0], f[1]) for f in food[:7]]
        virus_out = (viruses[0][0], viruses[0][1]) if viruses else None
        
        return self_radius, players_out, food_out, virus_out


# ============== DUELING DQN NETWORK (EXACT from dqnAgario.py) ==============
class QNet(nn.Module):
    """Dueling DQN - MUST match dqnAgario.py exactly for checkpoint loading"""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        
        self.value_stream = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )
        
        self.scale = 1.0 / (2 ** 0.5)
    
    def forward(self, x):
        features = self.feature(x)
        features = features * self.scale
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q


# ============== TEST ENVIRONMENT ==============
class TestEnv:
    CONTINUE_CLICK = (759, -566)
    START_CLICK = (754, -655)
    ACTION_DURATION = 0.25
    ACTION_REFRESH_RATE = 0.05
    SMALLER_THRESHOLD = 0.7
    LARGER_THRESHOLD = 1.0
    
    def __init__(self):
        self.detector = UnifiedDetector()
        self.start_det = mssStartButton()
        self.score_det = mssBottomLeftCorner()
        self.cont_det = mssMiddleRectangle()
        
        self.w = self.detector.w
        self.h = self.detector.h
        
        with mss.mss() as sct:
            mon = sct.monitors[2] if len(sct.monitors) > 2 else sct.monitors[1]
            self.mon_w = mon['width']
            self.mon_h = mon['height']
            self.mon_left = mon['left']
            self.mon_top = mon['top']
        
        self.center_x = self.mon_left + self.mon_w / 2
        self.center_y = self.mon_top + self.mon_h / 2
        
        self.last_self_radius = None
        self.last_players = []
        self.last_food = []
        self.last_virus = None
        
        self.actions = [
            self._act_toward_nearest_food,
            self._act_toward_smaller_player,
            self._act_away_from_larger_player,
            self._act_toward_virus,
            self._act_away_from_virus,
        ]
        
        self.action_names = ["EAT_FOOD", "HUNT_PREY", "FLEE_THREAT", "TO_VIRUS", "FROM_VIRUS"]
        self.state_dim = 26
        self.action_dim = 5
        self.self_radius = 60
        self.food_count = 0
        self.player_count = 0
    
    def _move_to_relative(self, dx, dy, scale=1.0):
        sx = dx * scale
        sy = dy * scale
        target_x = self.center_x + sx
        target_y = self.center_y + sy
        target_x = max(self.mon_left, min(self.mon_left + self.mon_w - 1, target_x))
        target_y = max(self.mon_top, min(self.mon_top + self.mon_h - 1, target_y))
        pyautogui.moveTo(target_x, target_y)
    
    def _move_toward(self, dx, dy):
        self._move_to_relative(dx, dy, scale=1.0)
    
    def _move_away(self, dx, dy):
        self._move_to_relative(-dx, -dy, scale=1.0)
    
    def _act_toward_nearest_food(self):
        if self.last_food:
            dx, dy = self.last_food[0]
            self._move_toward(dx, dy)
    
    def _act_toward_smaller_player(self):
        if not self.last_players:
            return self._act_toward_nearest_food()
        target = None
        for (dx, dy, r) in self.last_players:
            if self.self_radius and r < self.self_radius * self.SMALLER_THRESHOLD:
                target = (dx, dy)
                break
        if target is None:
            return self._act_toward_nearest_food()
        dx, dy = target
        self._move_toward(dx, dy)
    
    def _act_away_from_larger_player(self):
        if not self.last_players:
            return self._act_toward_nearest_food()
        threats = []
        for (dx, dy, r) in self.last_players:
            if self.self_radius and r >= self.self_radius * self.LARGER_THRESHOLD:
                dist = (dx**2 + dy**2)**0.5
                threats.append((dx, dy, r, dist))
        if threats:
            threats.sort(key=lambda t: t[3])
            dx, dy = threats[0][0], threats[0][1]
            self._move_away(dx, dy)
        else:
            dx, dy, _ = self.last_players[0]
            self._move_away(dx, dy)
    
    def _act_toward_virus(self):
        if self.last_virus is None:
            return self._act_toward_nearest_food()
        dx, dy = self.last_virus
        self._move_toward(dx, dy)
    
    def _act_away_from_virus(self):
        if self.last_virus is None:
            return self._act_toward_nearest_food()
        dx, dy = self.last_virus
        self._move_away(dx, dy)
    
    def _build_state(self, self_r, players, food, virus):
        state = np.zeros(26, dtype=np.float32)
        max_dist = self.w
        max_radius = 300
        
        for i in range(3):
            if i < len(players):
                dx, dy, r = players[i]
                state[i*3] = np.clip(dx / max_dist, -1, 1)
                state[i*3 + 1] = np.clip(dy / max_dist, -1, 1)
                state[i*3 + 2] = min(r / max_radius, 1.0)
        
        if virus:
            state[9] = np.clip(virus[0] / max_dist, -1, 1)
            state[10] = np.clip(virus[1] / max_dist, -1, 1)
        
        for i in range(7):
            if i < len(food):
                dx, dy = food[i]
                state[11 + i*2] = np.clip(dx / max_dist, -1, 1)
                state[11 + i*2 + 1] = np.clip(dy / max_dist, -1, 1)
        
        state[25] = min((self_r or 60) / max_radius, 1.0)
        return state
    
    def get_state(self):
        self_r, players, food, virus = self.detector.detect_all()
        if self_r:
            self.self_radius = self_r
        
        self.last_self_radius = self_r
        self.last_players = players
        self.last_food = food
        self.last_virus = virus
        
        self.food_count = len(food)
        self.player_count = len(players)
        
        return self._build_state(self_r, players, food, virus)
    
    def step(self, action):
        start_time = time.time()
        while time.time() - start_time < self.ACTION_DURATION:
            self.get_state()
            self.actions[action]()
            time.sleep(self.ACTION_REFRESH_RATE)
        return self.get_state()
    
    def check_death(self):
        return self.cont_det.hasContinue() or self.start_det.hasStart()
    
    def handle_death(self):
        if self.cont_det.hasContinue():
            pyautogui.click(*self.CONTINUE_CLICK)
            time.sleep(0.3)
        if self.start_det.hasStart():
            pyautogui.click(*self.START_CLICK)
            time.sleep(0.3)
    
    def reset(self):
        for _ in range(3):
            if self.cont_det.hasContinue():
                pyautogui.click(*self.CONTINUE_CLICK)
                time.sleep(0.5)
            if self.start_det.hasStart():
                pyautogui.click(*self.START_CLICK)
                time.sleep(0.5)
            time.sleep(0.3)
        self.self_radius = 60
        return self.get_state()
    
    def get_score(self):
        return self.score_det.getScore()


# ============== TEST AGENT (NO EXPLORATION) ==============
class TestAgent:
    def __init__(self, state_dim=26, action_dim=5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q = QNet(state_dim, action_dim).to(self.device)
        self.q.eval()  # Evaluation mode
    
    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if 'q' in ckpt:
            self.q.load_state_dict(ckpt['q'])
        elif 'q_net' in ckpt:
            self.q.load_state_dict(ckpt['q_net'])
        else:
            self.q.load_state_dict(ckpt)
        self.q.eval()
        print(f"✓ Loaded: {path}")
    
    def act(self, state):
        """Pure greedy action - NO exploration"""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q(state_t)
            action = q_values.argmax(1).item()
            return action, q_values.cpu().numpy()[0]


# ============== FIND CHECKPOINT ==============
def find_dqn_checkpoint(target_ep=None):
    """Find DQN checkpoint. If target_ep specified, find that one."""
    ckpts = []
    for f in glob.glob("checkpoints/dqn_ep*.pth") + glob.glob("checkpoints/*dqn_ep*.pth"):
        m = re.search(r'ep(\d+)', f)
        if m:
            ckpts.append((int(m.group(1)), f))
    
    if not ckpts:
        return None
    
    ckpts.sort(key=lambda x: x[0])
    
    if target_ep:
        for ep, path in ckpts:
            if ep == target_ep:
                return path
    
    # Return latest
    return ckpts[-1][1]


# ============== MAIN TEST ==============
def test(checkpoint_path=None, num_episodes=5, max_steps=600):
    global stop_test
    
    print("=" * 70)
    print("🧪 DUELING DQN TEST MODE (from dqnAgario.py)")
    print("=" * 70)
    print("Architecture: Dueling DQN")
    print("  State: 26 dims (3 players + 1 virus + 7 food + self)")
    print("  Actions: 5 semantic (EAT, HUNT, FLEE, TO_VIRUS, FROM_VIRUS)")
    print()
    print("Mode: PURE GREEDY (ε = 0, no exploration)")
    print(f"Running {num_episodes} test episodes")
    print("Press Q to stop")
    print("=" * 70)
    
    # Find checkpoint
    if checkpoint_path is None:
        # Default: try ep80 (right before the 82202 score)
        checkpoint_path = find_dqn_checkpoint(80)
        if not checkpoint_path:
            checkpoint_path = find_dqn_checkpoint()
    
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found!")
        return
    
    print(f"\nLoading: {checkpoint_path}")
    
    env = TestEnv()
    agent = TestAgent()
    agent.load(checkpoint_path)
    
    # Extract checkpoint name for CSV
    ckpt_name = os.path.basename(checkpoint_path).replace('.pth', '')
    logger = TestCSVLogger(ckpt_name)
    print(f"Logging to: {logger.filename}")
    
    print("\nStarting test in 3 seconds...")
    print("Make sure Agar.io is visible!")
    time.sleep(3)
    
    # Stats tracking
    episode_stats = []
    
    for ep in range(num_episodes):
        if stop_test:
            break
        
        print(f"\n{'='*50}")
        print(f"EPISODE {ep + 1}/{num_episodes}")
        print(f"{'='*50}")
        
        state = env.reset()
        ep_start = time.time()
        score = 0
        last_score_check = time.time()
        
        for step in range(max_steps):
            if stop_test:
                break
            
            action, q_values = agent.act(state)
            state = env.step(action)
            
            time_alive = time.time() - ep_start
            
            # Check score periodically
            if time.time() - last_score_check > 2.0:
                last_score_check = time.time()
                new_score = env.get_score()
                if new_score > score:
                    score = new_score
            
            # Log
            logger.log(ep + 1, step, action, env.action_names[action], q_values, env, time_alive, score)
            
            # Print progress
            if step % 10 == 0:
                q_str = f"[{q_values[0]:.1f},{q_values[1]:.1f},{q_values[2]:.1f},{q_values[3]:.1f},{q_values[4]:.1f}]"
                print(f"\rEp{ep+1} Step{step:4d} | {env.action_names[action]:11s} | "
                      f"R={env.self_radius:3d} F={env.food_count:2d} P={env.player_count} | "
                      f"Q={q_str} | Score={score} Time={time_alive:.1f}s", end="", flush=True)
            
            # Check for death
            if env.check_death():
                print(f"\n💀 Death at step {step}!")
                final_score = env.get_score()
                if final_score > score:
                    score = final_score
                env.handle_death()
                break
        
        if stop_test:
            break
        
        # Episode stats
        time_alive = time.time() - ep_start
        episode_stats.append({
            'episode': ep + 1,
            'steps': step,
            'time_alive': time_alive,
            'score': score
        })
        
        print(f"\n📊 Episode {ep+1} Results:")
        print(f"   Steps: {step}, Time: {time_alive:.1f}s, Score: {score}")
    
    logger.close()
    
    # Print summary
    print("\n" + "=" * 70)
    print("📈 TEST SUMMARY")
    print("=" * 70)
    
    if episode_stats:
        avg_steps = np.mean([e['steps'] for e in episode_stats])
        avg_time = np.mean([e['time_alive'] for e in episode_stats])
        avg_score = np.mean([e['score'] for e in episode_stats])
        
        print(f"\nCheckpoint: {checkpoint_path}")
        print(f"Episodes completed: {len(episode_stats)}")
        print(f"\n  ⏱️  Average time alive:  {avg_time:.2f}s")
        print(f"  👣 Average steps:       {avg_steps:.1f}")
        print(f"  🏆 Average score:       {avg_score:.1f}")
        
        print(f"\n  Per-episode breakdown:")
        for e in episode_stats:
            print(f"    Ep{e['episode']}: Steps={e['steps']:4d}, Time={e['time_alive']:.1f}s, Score={e['score']}")
        
        # Save summary
        summary_file = logger.filename.replace('.csv', '_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("DUELING DQN TEST SUMMARY\n")
            f.write("=" * 50 + "\n")
            f.write(f"Checkpoint: {checkpoint_path}\n")
            f.write(f"Episodes: {len(episode_stats)}\n")
            f.write(f"Date: {datetime.now().isoformat()}\n\n")
            f.write(f"Average time alive: {avg_time:.2f}s\n")
            f.write(f"Average steps:      {avg_steps:.1f}\n")
            f.write(f"Average score:      {avg_score:.1f}\n\n")
            f.write("Per-episode:\n")
            for e in episode_stats:
                f.write(f"  Ep{e['episode']}: Steps={e['steps']}, Time={e['time_alive']:.1f}s, Score={e['score']}\n")
        
        print(f"\n  Summary saved to: {summary_file}")
    
    print(f"\n  Full log: {logger.filename}")
    print("\n✓ Test complete!")


if __name__ == "__main__":
    import sys
    
    # Allow specifying checkpoint as command line arg
    # Usage: python testDuelingDQN.py checkpoints/dqn_ep30.pth
    if len(sys.argv) > 1:
        test(checkpoint_path=sys.argv[1])
    else:
        # Default: load dqn_ep30.pth
        test(checkpoint_path="checkpoints/dqn_ep30.pth")

