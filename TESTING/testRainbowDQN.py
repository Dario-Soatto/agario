"""
Rainbow DQN Test Mode for Agar.io
==================================
Loads the latest checkpoint and runs in pure evaluation mode:
- NO epsilon-greedy exploration (epsilon = 0)
- NO noisy network exploration (eval mode)
- Logs detailed CSV data per step
- Tracks average score and time alive over 5 episodes

Press Q to quit
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
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
import easyocr

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


# ============== CSV LOGGER FOR TESTING ==============
class TestCSVLogger:
    """Logs detailed test data per step"""
    
    def __init__(self, filename=None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rainbow_test_{timestamp}.csv"
        self.filename = filename
        self.file = None
        self.writer = None
        self._init_file()
    
    def _init_file(self):
        self.file = open(self.filename, 'w', newline='')
        self.writer = csv.writer(self.file)
        # Comprehensive columns for testing
        self.writer.writerow([
            'timestamp', 'episode', 'step', 
            'action', 'action_name',
            'q_up', 'q_down', 'q_left', 'q_right', 'q_chosen',
            'reward', 'cumulative_reward',
            'food_count', 'has_virus',
            'nearest_food_dx', 'nearest_food_dy', 'nearest_food_dist',
            'virus_dx', 'virus_dy', 'virus_dist',
            'steps_alive', 'time_alive_sec',
            'done'
        ])
        self.file.flush()
    
    def log(self, episode, step, action, action_name, q_values, reward, cum_reward,
            state, steps_alive, time_alive, done):
        # Extract state info
        nearest_food_dx = state[0] if len(state) > 0 else 0
        nearest_food_dy = state[1] if len(state) > 1 else 0
        nearest_food_dist = math.sqrt(nearest_food_dx**2 + nearest_food_dy**2)
        
        virus_dx = state[14] if len(state) > 14 else 0
        virus_dy = state[15] if len(state) > 15 else 0
        virus_dist = math.sqrt(virus_dx**2 + virus_dy**2) if (virus_dx != 0 or virus_dy != 0) else -1
        
        food_count = int(state[16] * 100) if len(state) > 16 else 0
        has_virus = int(state[17]) if len(state) > 17 else 0
        
        self.writer.writerow([
            datetime.now().isoformat(),
            episode, step,
            action, action_name,
            f"{q_values[0]:.4f}", f"{q_values[1]:.4f}", 
            f"{q_values[2]:.4f}", f"{q_values[3]:.4f}",
            f"{q_values[action]:.4f}",
            f"{reward:.4f}", f"{cum_reward:.4f}",
            food_count, has_virus,
            f"{nearest_food_dx:.4f}", f"{nearest_food_dy:.4f}", f"{nearest_food_dist:.4f}",
            f"{virus_dx:.4f}", f"{virus_dy:.4f}", f"{virus_dist:.4f}",
            steps_alive, f"{time_alive:.2f}",
            int(done)
        ])
        self.file.flush()  # Flush every line in test mode for safety
    
    def close(self):
        if self.file:
            self.file.close()


# ============== DETECTION (Copied from rainbowDQN.py) ==============
class GameDetector:
    """All-in-one detector"""
    
    VIRUS_BGR = (72, 244, 18)
    COLOR_TOLERANCE = 40
    
    def __init__(self):
        with mss.mss() as sct:
            if len(sct.monitors) > 2:
                self.monitor = sct.monitors[2]
            else:
                self.monitor = sct.monitors[1]
        
        padding_x = int(self.monitor['width'] * 0.05)
        padding_y = int(self.monitor['height'] * 0.05)
        self.game_region = {
            "top": self.monitor['top'] + padding_y,
            "left": self.monitor['left'] + padding_x,
            "width": self.monitor['width'] - 2 * padding_x,
            "height": self.monitor['height'] - 2 * padding_y
        }
        
        self.center_x = self.game_region['width'] // 2
        self.center_y = self.game_region['height'] // 2
        
        width_third = self.monitor['width'] // 3
        height_third = self.monitor['height'] // 3
        self.menu_region = {
            "top": self.monitor['top'] + height_third,
            "left": self.monitor['left'] + width_third,
            "width": width_third,
            "height": height_third
        }
        
        self.score_region = {
            "top": self.monitor['top'] + int(self.monitor['height'] * 0.8),
            "left": self.monitor['left'],
            "width": int(self.monitor['width'] * 0.2),
            "height": int(self.monitor['height'] * 0.2)
        }
        
        self.mon_width = self.monitor['width']
        self.mon_height = self.monitor['height']
        self.mon_left = self.monitor['left']
        self.mon_top = self.monitor['top']
        
        self._ocr_reader = None
        self.sct = mss.mss()
    
    @property
    def ocr_reader(self):
        if self._ocr_reader is None:
            self._ocr_reader = easyocr.Reader(['en'], gpu=False)
        return self._ocr_reader
    
    def _find_food_blobs(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([180, 255, 255]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        blobs = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > 10:
                (x, y), radius = cv2.minEnclosingCircle(c)
                blobs.append({'x': int(x), 'y': int(y), 'radius': int(radius), 'area': area})
        return blobs
    
    def _classify_food(self, blobs):
        min_dist_from_center = min(self.center_x, self.center_y) * 0.15
        
        food = []
        for b in blobs:
            if not (3 < b['radius'] < 60):
                continue
            dist_from_center = ((b['x'] - self.center_x)**2 + (b['y'] - self.center_y)**2)**0.5
            if dist_from_center < min_dist_from_center:
                continue
            food.append(b)
        
        food.sort(key=lambda f: ((f['x'] - self.center_x)**2 + (f['y'] - self.center_y)**2)**0.5)
        return food
    
    def _find_viruses(self, img):
        lower = np.array([
            max(0, self.VIRUS_BGR[0] - self.COLOR_TOLERANCE),
            max(0, self.VIRUS_BGR[1] - self.COLOR_TOLERANCE),
            max(0, self.VIRUS_BGR[2] - self.COLOR_TOLERANCE)
        ])
        upper = np.array([
            min(255, self.VIRUS_BGR[0] + self.COLOR_TOLERANCE),
            min(255, self.VIRUS_BGR[1] + self.COLOR_TOLERANCE),
            min(255, self.VIRUS_BGR[2] + self.COLOR_TOLERANCE)
        ])
        
        mask = cv2.inRange(img, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        min_dist_from_center = min(self.center_x, self.center_y) * 0.15
        
        viruses = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > 500:
                (x, y), radius = cv2.minEnclosingCircle(c)
                if radius > 30:
                    dist = ((int(x) - self.center_x)**2 + (int(y) - self.center_y)**2)**0.5
                    if dist >= min_dist_from_center:
                        viruses.append({'x': int(x), 'y': int(y), 'radius': int(radius)})
        
        viruses.sort(key=lambda v: ((v['x'] - self.center_x)**2 + (v['y'] - self.center_y)**2)**0.5)
        return viruses
    
    def detect_game_state(self):
        raw = np.array(self.sct.grab(self.game_region), dtype=np.uint8)
        img = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
        
        blobs = self._find_food_blobs(img)
        food_blobs = self._classify_food(blobs)
        
        food_list = []
        for f in food_blobs[:7]:
            dx = f['x'] - self.center_x
            dy = f['y'] - self.center_y
            food_list.append((dx, dy))
        
        viruses = self._find_viruses(img)
        virus = None
        if viruses:
            v = viruses[0]
            virus = (v['x'] - self.center_x, v['y'] - self.center_y)
        
        return food_list, virus, len(food_blobs)
    
    def check_continue(self):
        raw = np.array(self.sct.grab(self.menu_region), dtype=np.uint8)
        img = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
        
        results = self.ocr_reader.readtext(img)
        text = ' '.join([r[1].lower() for r in results if r[2] >= 0.9])
        return 'continue' in text
    
    def check_start(self):
        raw = np.array(self.sct.grab(self.menu_region), dtype=np.uint8)
        img = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
        
        results = self.ocr_reader.readtext(img)
        text = ' '.join([r[1].lower() for r in results if r[2] >= 0.9])
        return 'play' in text or 'start' in text
    
    def get_score(self):
        raw = np.array(self.sct.grab(self.score_region), dtype=np.uint8)
        img = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)
        white_only = cv2.bitwise_and(img, img, mask=mask)
        
        results = self.ocr_reader.readtext(white_only)
        digits = ''.join(filter(str.isdigit, ''.join([r[1] for r in results])))
        return int(digits) if digits else 0


# ============== NOISY LINEAR (Copied from rainbowDQN.py) ==============
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))
    
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign() * x.abs().sqrt()
    
    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            # In eval mode: use only the learned weights (no noise)
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


# ============== RAINBOW NETWORK (Copied from rainbowDQN.py) ==============
class RainbowNet(nn.Module):
    def __init__(self, state_dim, action_dim, n_atoms=51, v_min=-100, v_max=100):
        super().__init__()
        
        self.action_dim = action_dim
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        
        self.register_buffer('support', torch.linspace(v_min, v_max, n_atoms))
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        
        self.value_hidden = NoisyLinear(256, 256)
        self.value_out = NoisyLinear(256, n_atoms)
        
        self.advantage_hidden = NoisyLinear(256, 256)
        self.advantage_out = NoisyLinear(256, action_dim * n_atoms)
    
    def forward(self, x):
        batch_size = x.size(0)
        features = self.feature(x)
        
        value = F.relu(self.value_hidden(features))
        value = self.value_out(value).view(batch_size, 1, self.n_atoms)
        
        advantage = F.relu(self.advantage_hidden(features))
        advantage = self.advantage_out(advantage).view(batch_size, self.action_dim, self.n_atoms)
        
        q_dist = value + advantage - advantage.mean(dim=1, keepdim=True)
        q_dist = F.softmax(q_dist, dim=2)
        
        return q_dist
    
    def get_q_values(self, x):
        dist = self.forward(x)
        q = (dist * self.support.view(1, 1, -1)).sum(dim=2)
        return q
    
    def reset_noise(self):
        self.value_hidden.reset_noise()
        self.value_out.reset_noise()
        self.advantage_hidden.reset_noise()
        self.advantage_out.reset_noise()


# ============== TEST AGENT (NO EXPLORATION) ==============
class TestAgent:
    """Rainbow agent in pure evaluation mode - no exploration"""
    
    N_ATOMS = 51
    V_MIN = -100
    V_MAX = 100
    
    def __init__(self, state_dim, action_dim):
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.net = RainbowNet(state_dim, action_dim, self.N_ATOMS, self.V_MIN, self.V_MAX).to(self.device)
        # Set to evaluation mode - NoisyLinear will use only learned weights (no noise)
        self.net.eval()
    
    def act(self, state):
        """
        Select action using pure Q-value maximization
        NO epsilon-greedy, NO noisy exploration
        """
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.net.get_q_values(state_t)
            action = q_values.argmax(1).item()
            return action, q_values[0].cpu().numpy()
    
    def load(self, path):
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            if 'online' in ckpt:
                self.net.load_state_dict(ckpt['online'])
            else:
                self.net.load_state_dict(ckpt)
            # Ensure eval mode after loading
            self.net.eval()
            print(f"✓ Loaded checkpoint: {path}")
            return True
        except Exception as e:
            print(f"✗ Failed to load {path}: {e}")
            return False


# ============== TEST ENVIRONMENT ==============
class TestEnv:
    """Agar.io environment for testing (no reward calculation needed)"""
    
    CONTINUE_CLICK = (759, -566)
    START_CLICK = (754, -655)
    ACTION_DURATION = 0.25
    
    def __init__(self):
        self.detector = GameDetector()
        
        self.screen_center_x = self.detector.mon_left + self.detector.mon_width / 2
        self.screen_center_y = self.detector.mon_top + self.detector.mon_height / 2
        
        self.move_offset = min(self.detector.mon_width, self.detector.mon_height) * 0.20
        
        self.action_names = ["UP", "DOWN", "LEFT", "RIGHT"]
        self.action_dim = 4
        self.state_dim = 18
        
        self.steps_alive = 0
        self.episode_start = time.time()
        
        self.last_food = []
        self.last_virus = None
        self.last_food_count = 0
    
    def _execute_action(self, action):
        if action == 0:  # UP
            pyautogui.moveTo(self.screen_center_x, self.screen_center_y - self.move_offset)
        elif action == 1:  # DOWN
            pyautogui.moveTo(self.screen_center_x, self.screen_center_y + self.move_offset)
        elif action == 2:  # LEFT
            pyautogui.moveTo(self.screen_center_x - self.move_offset, self.screen_center_y)
        elif action == 3:  # RIGHT
            pyautogui.moveTo(self.screen_center_x + self.move_offset, self.screen_center_y)
    
    def _build_state(self, food_list, virus, food_count):
        state = np.zeros(18, dtype=np.float32)
        max_dist = self.detector.center_x
        
        for i in range(7):
            if i < len(food_list):
                dx, dy = food_list[i]
                state[i * 2] = np.clip(dx / max_dist, -1, 1)
                state[i * 2 + 1] = np.clip(dy / max_dist, -1, 1)
        
        if virus:
            state[14] = np.clip(virus[0] / max_dist, -1, 1)
            state[15] = np.clip(virus[1] / max_dist, -1, 1)
        
        state[16] = min(food_count / 100.0, 1.0)
        state[17] = 1.0 if virus else 0.0
        
        return state
    
    def _get_state(self):
        food_list, virus, food_count = self.detector.detect_game_state()
        
        self.last_food = food_list
        self.last_virus = virus
        self.last_food_count = food_count
        
        return self._build_state(food_list, virus, food_count)
    
    def reset(self):
        for _ in range(3):
            try:
                if self.detector.check_continue():
                    pyautogui.click(*self.CONTINUE_CLICK)
                    time.sleep(0.5)
                if self.detector.check_start():
                    pyautogui.click(*self.START_CLICK)
                    time.sleep(0.5)
            except:
                pass
            time.sleep(0.3)
        
        self.steps_alive = 0
        self.episode_start = time.time()
        
        return self._get_state()
    
    def step(self, action):
        start_time = time.time()
        while time.time() - start_time < self.ACTION_DURATION:
            self._execute_action(action)
            time.sleep(0.05)
        
        state = self._get_state()
        self.steps_alive += 1
        
        done = False
        try:
            if self.detector.check_continue() or self.detector.check_start():
                done = True
                pyautogui.click(*self.CONTINUE_CLICK)
                time.sleep(0.3)
                pyautogui.click(*self.START_CLICK)
                time.sleep(0.3)
        except:
            pass
        
        # Simple reward calculation for logging purposes
        reward = 0.5  # Base survival
        if self.last_food:
            nearest_dist = math.sqrt(self.last_food[0][0]**2 + self.last_food[0][1]**2)
            if nearest_dist < 100:
                reward += 0.3 * (100 - nearest_dist) / 100
        if done:
            reward = -50.0
        
        time_alive = time.time() - self.episode_start
        
        return state, reward, done, {
            'steps_alive': self.steps_alive,
            'time_alive': time_alive,
            'food_count': self.last_food_count,
            'action': self.action_names[action]
        }
    
    def get_score(self):
        """Get the current in-game score"""
        return self.detector.get_score()


# ============== FIND LATEST CHECKPOINT ==============
def find_latest_rainbow_checkpoint():
    """Find latest rainbow checkpoint file"""
    ckpts = []
    for f in glob.glob("checkpoints/rainbow_ep*.pth"):
        m = re.search(r'ep(\d+)', f)
        if m:
            ckpts.append((int(m.group(1)), f))
    
    if not ckpts:
        return None, 0
    
    ckpts.sort(key=lambda x: -x[0])
    return ckpts[0][1], ckpts[0][0]


# ============== MAIN TEST FUNCTION ==============
def test(num_episodes=5, max_steps=600):
    """Run test episodes with full logging"""
    global stop_test
    
    print("=" * 70)
    print("🧪 RAINBOW DQN TEST MODE 🧪")
    print("=" * 70)
    print()
    print("Mode: PURE EVALUATION (No exploration)")
    print("  ✗ Epsilon-greedy: DISABLED (ε = 0)")
    print("  ✗ Noisy networks: DISABLED (eval mode)")
    print("  ✓ Pure Q-value maximization")
    print()
    print(f"Testing for {num_episodes} episodes")
    print("Press Q to stop")
    print("=" * 70)
    
    # Find and load latest checkpoint
    ckpt_path, ep_num = find_latest_rainbow_checkpoint()
    if not ckpt_path:
        print("ERROR: No rainbow checkpoint found!")
        return
    
    print(f"\nUsing checkpoint: {ckpt_path} (Episode {ep_num})")
    
    env = TestEnv()
    agent = TestAgent(env.state_dim, env.action_dim)
    
    if not agent.load(ckpt_path):
        print("ERROR: Failed to load checkpoint!")
        return
    
    logger = TestCSVLogger()
    print(f"Logging to: {logger.filename}")
    
    print("\nStarting test in 3 seconds...")
    print("Make sure Agar.io is visible!")
    time.sleep(3)
    
    # Track stats across episodes
    episode_stats = []
    
    for ep in range(num_episodes):
        if stop_test:
            break
        
        print(f"\n{'='*50}")
        print(f"EPISODE {ep + 1}/{num_episodes}")
        print(f"{'='*50}")
        
        state = env.reset()
        ep_reward = 0
        ep_start = time.time()
        
        # Try to get initial score
        try:
            start_score = env.get_score()
        except:
            start_score = 0
        
        for step in range(max_steps):
            if stop_test:
                break
            
            action, q_values = agent.act(state)
            next_state, reward, done, info = env.step(action)
            
            ep_reward += reward
            time_alive = time.time() - ep_start
            
            # Log step data
            logger.log(
                episode=ep + 1,
                step=step,
                action=action,
                action_name=env.action_names[action],
                q_values=q_values,
                reward=reward,
                cum_reward=ep_reward,
                state=state,
                steps_alive=info['steps_alive'],
                time_alive=time_alive,
                done=done
            )
            
            # Print progress
            if step % 10 == 0:
                q_str = f"[{q_values[0]:.1f},{q_values[1]:.1f},{q_values[2]:.1f},{q_values[3]:.1f}]"
                print(f"\rEp{ep+1} Step{step:4d} | {info['action']:5s} | "
                      f"Food={info['food_count']:3d} | Q={q_str} | "
                      f"Time={time_alive:.1f}s", end="", flush=True)
            
            state = next_state
            
            if done:
                print(f"\n💀 Death at step {step}!")
                break
        
        if stop_test:
            break
        
        # End of episode stats
        end_time = time.time()
        time_alive = end_time - ep_start
        
        try:
            end_score = env.get_score()
        except:
            end_score = 0
        
        episode_stats.append({
            'episode': ep + 1,
            'steps': info['steps_alive'],
            'time_alive': time_alive,
            'score': max(end_score, start_score),
            'total_reward': ep_reward
        })
        
        print(f"\n📊 Episode {ep+1} Results:")
        print(f"   Steps alive: {info['steps_alive']}")
        print(f"   Time alive: {time_alive:.2f}s")
        print(f"   Score: {episode_stats[-1]['score']}")
    
    logger.close()
    
    # Print summary
    print("\n" + "=" * 70)
    print("📈 TEST SUMMARY")
    print("=" * 70)
    
    if episode_stats:
        avg_steps = np.mean([e['steps'] for e in episode_stats])
        avg_time = np.mean([e['time_alive'] for e in episode_stats])
        avg_score = np.mean([e['score'] for e in episode_stats])
        
        print(f"\nEpisodes completed: {len(episode_stats)}")
        print(f"\n  Average steps alive: {avg_steps:.1f}")
        print(f"  Average time alive:  {avg_time:.2f}s")
        print(f"  Average score:       {avg_score:.1f}")
        
        print(f"\n  Per-episode breakdown:")
        for e in episode_stats:
            print(f"    Ep{e['episode']}: Steps={e['steps']:4d}, Time={e['time_alive']:.1f}s, Score={e['score']}")
        
        # Save summary to a separate file
        summary_file = logger.filename.replace('.csv', '_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("RAINBOW DQN TEST SUMMARY\n")
            f.write("=" * 50 + "\n")
            f.write(f"Checkpoint: {ckpt_path}\n")
            f.write(f"Episodes: {len(episode_stats)}\n")
            f.write(f"Date: {datetime.now().isoformat()}\n\n")
            f.write(f"Average steps alive: {avg_steps:.1f}\n")
            f.write(f"Average time alive:  {avg_time:.2f}s\n")
            f.write(f"Average score:       {avg_score:.1f}\n\n")
            f.write("Per-episode:\n")
            for e in episode_stats:
                f.write(f"  Ep{e['episode']}: Steps={e['steps']}, Time={e['time_alive']:.1f}s, Score={e['score']}\n")
        
        print(f"\n  Summary saved to: {summary_file}")
    
    print(f"\n  Full log saved to: {logger.filename}")
    print("\n✓ Test complete!")


if __name__ == "__main__":
    test(num_episodes=5, max_steps=600)

