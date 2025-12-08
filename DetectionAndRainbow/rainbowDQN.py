"""
Rainbow DQN for Agar.io
========================
Combines 6 key improvements to DQN:
1. Double DQN - reduces overestimation bias
2. Prioritized Experience Replay (PER) - learn from important transitions
3. Dueling Networks - separate value and advantage streams
4. Multi-step Learning (N-step returns) - faster credit assignment
5. Distributional RL (C51) - model return distribution, not just mean
6. Noisy Networks - learned exploration instead of ε-greedy

State Vector (18 dims):
  [0-13]  7 nearest food: (dx, dy) each - normalized
  [14-15] 1 nearest virus: (dx, dy) - normalized  
  [16]    food density (count / 100)
  [17]    has_virus (0 or 1)

Action Space (4 cardinal directions):
  0: UP    - Move up
  1: DOWN  - Move down
  2: LEFT  - Move left
  3: RIGHT - Move right

Press Q to save and quit
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
import time
import os
import csv
from collections import deque
from datetime import datetime
from pynput import keyboard
import cv2
import mss
import pyautogui
import easyocr

# ============== Q TO QUIT ==============
stop_training = False
def on_press(key):
    global stop_training
    try:
        if key.char == 'q':
            stop_training = True
            print("\n[Q pressed] Saving and quitting...")
    except:
        pass
keyboard.Listener(on_press=on_press).start()


# ============== CSV LOGGER ==============
class CSVLogger:
    """Logs all Q-learning transitions to CSV for analysis"""
    
    def __init__(self, filename=None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rainbow_log_{timestamp}.csv"
        self.filename = filename
        self.file = None
        self.writer = None
        self._init_file()
    
    def _init_file(self):
        self.file = open(self.filename, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            'timestamp', 'episode', 'step', 'action', 'action_name',
            'reward', 'done', 'score', 'steps_alive',
            'nearest_food_dx', 'nearest_food_dy', 'food_count',
            'has_virus', 'q_value', 'td_error'
        ])
        self.file.flush()
    
    def log(self, episode, step, action, action_name, reward, done, info, 
            state, q_value=0, td_error=0):
        nearest_food_dx = state[0] if len(state) > 0 else 0
        nearest_food_dy = state[1] if len(state) > 1 else 0
        food_count = int(state[16] * 100) if len(state) > 16 else 0
        has_virus = int(state[17]) if len(state) > 17 else 0
        
        self.writer.writerow([
            datetime.now().isoformat(), episode, step, action, action_name,
            f"{reward:.4f}", int(done), info.get('score', 0),
            info.get('steps_alive', 0), f"{nearest_food_dx:.3f}", f"{nearest_food_dy:.3f}",
            food_count, has_virus, f"{q_value:.4f}", f"{td_error:.4f}"
        ])
        
        if step % 100 == 0:
            self.file.flush()
    
    def close(self):
        if self.file:
            self.file.flush()
            self.file.close()


# ============== DETECTION (Using exact logic from working files) ==============
class GameDetector:
    """
    All-in-one detector using exact logic from:
    - detectFood.py (FoodDetector)
    - detectViruses.py (VirusDetector)
    - detectContinue.py (mssMiddleRectangle)
    - detectStart.py (mssStartButton)
    - detectScore.py (mssBottomLeftCorner)
    """
    
    # Virus color from detectViruses.py: #12F448 in BGR = (72, 244, 18)
    VIRUS_BGR = (72, 244, 18)
    COLOR_TOLERANCE = 40
    
    def __init__(self):
        # Get monitor info (same pattern as all detect files)
        with mss.mss() as sct:
            if len(sct.monitors) > 2:
                self.monitor = sct.monitors[2]
            else:
                self.monitor = sct.monitors[1]
        
        # Game region with padding (from detectFood.py / detectViruses.py)
        padding_x = int(self.monitor['width'] * 0.05)
        padding_y = int(self.monitor['height'] * 0.05)
        self.game_region = {
            "top": self.monitor['top'] + padding_y,
            "left": self.monitor['left'] + padding_x,
            "width": self.monitor['width'] - 2 * padding_x,
            "height": self.monitor['height'] - 2 * padding_y
        }
        
        # Center coords (from detectViruses.py)
        self.center_x = self.game_region['width'] // 2
        self.center_y = self.game_region['height'] // 2
        
        # Middle region for continue/start detection (from detectContinue.py / detectStart.py)
        width_third = self.monitor['width'] // 3
        height_third = self.monitor['height'] // 3
        self.menu_region = {
            "top": self.monitor['top'] + height_third,
            "left": self.monitor['left'] + width_third,
            "width": width_third,
            "height": height_third
        }
        
        # Score region (from detectScore.py)
        self.score_region = {
            "top": self.monitor['top'] + int(self.monitor['height'] * 0.8),
            "left": self.monitor['left'],
            "width": int(self.monitor['width'] * 0.2),
            "height": int(self.monitor['height'] * 0.2)
        }
        
        # Screen info for mouse movement
        self.mon_width = self.monitor['width']
        self.mon_height = self.monitor['height']
        self.mon_left = self.monitor['left']
        self.mon_top = self.monitor['top']
        
        # OCR reader for menu detection (lazy load)
        self._ocr_reader = None
        
        # Persistent mss for speed
        self.sct = mss.mss()
    
    @property
    def ocr_reader(self):
        """Lazy load OCR reader (slow to initialize)"""
        if self._ocr_reader is None:
            self._ocr_reader = easyocr.Reader(['en'], gpu=False)
        return self._ocr_reader
    
    def _find_food_blobs(self, img):
        """
        Find food blobs - EXACT logic from detectFood.py _find_blobs()
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Detect saturated colors (food pellets are colorful)
        mask = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([180, 255, 255]))
        # Smaller kernel for food (they're tiny)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        blobs = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > 10:  # Lower threshold for small food
                (x, y), radius = cv2.minEnclosingCircle(c)
                blobs.append({'x': int(x), 'y': int(y), 'radius': int(radius), 'area': area})
        return blobs
    
    def _classify_food(self, blobs):
        """
        Classify food - based on detectFood.py _classify_food()
        Food is small (radius < 60) and not too tiny (radius > 3)
        
        IMPORTANT: Excludes blobs near center (within 15% of screen) 
        because that's where OUR PLAYER always is in Agar.io!
        """
        # Minimum distance from center to not be our player
        # Our player is ALWAYS at screen center in Agar.io
        min_dist_from_center = min(self.center_x, self.center_y) * 0.15
        
        food = []
        for b in blobs:
            # Must be small (food-sized)
            if not (3 < b['radius'] < 60):
                continue
            
            # Calculate distance from center
            dist_from_center = ((b['x'] - self.center_x)**2 + (b['y'] - self.center_y)**2)**0.5
            
            # EXCLUDE blobs too close to center - that's US, not food!
            if dist_from_center < min_dist_from_center:
                continue
            
            food.append(b)
        
        # Sort by distance to center (nearest food first)
        food.sort(key=lambda f: ((f['x'] - self.center_x)**2 + (f['y'] - self.center_y)**2)**0.5)
        return food
    
    def _find_viruses(self, img):
        """
        Find viruses - based on detectViruses.py _find_viruses()
        Viruses are bright green (#12F448) and fairly large.
        
        Also excludes center area where our player is.
        """
        # Create mask for virus green color in BGR
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
        
        # Minimum distance from center to not be confused with our player
        min_dist_from_center = min(self.center_x, self.center_y) * 0.15
        
        viruses = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > 500:  # Viruses are fairly large
                (x, y), radius = cv2.minEnclosingCircle(c)
                if radius > 30:  # Minimum virus size
                    # Check distance from center
                    dist = ((int(x) - self.center_x)**2 + (int(y) - self.center_y)**2)**0.5
                    # Exclude if too close to center (that's our player area)
                    if dist >= min_dist_from_center:
                        viruses.append({'x': int(x), 'y': int(y), 'radius': int(radius)})
        
        # Sort by distance to center (nearest first)
        viruses.sort(key=lambda v: ((v['x'] - self.center_x)**2 + (v['y'] - self.center_y)**2)**0.5)
        return viruses
    
    def detect_game_state(self):
        """
        Single capture for food and viruses.
        Returns: (food_list, virus, food_count)
        - food_list: [(dx, dy), ...] up to 7 nearest, relative to center
        - virus: (dx, dy) or None, relative to center
        - food_count: total food visible
        """
        # Single capture
        raw = np.array(self.sct.grab(self.game_region), dtype=np.uint8)
        img = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
        
        # Detect food (using exact detectFood.py logic)
        blobs = self._find_food_blobs(img)
        food_blobs = self._classify_food(blobs)
        
        # Convert to relative coords
        food_list = []
        for f in food_blobs[:7]:
            dx = f['x'] - self.center_x
            dy = f['y'] - self.center_y
            food_list.append((dx, dy))
        
        # Detect viruses (using exact detectViruses.py logic)
        viruses = self._find_viruses(img)
        virus = None
        if viruses:
            v = viruses[0]
            virus = (v['x'] - self.center_x, v['y'] - self.center_y)
        
        return food_list, virus, len(food_blobs)
    
    def check_continue(self):
        """
        Check for continue screen - EXACT logic from detectContinue.py hasContinue()
        """
        raw = np.array(self.sct.grab(self.menu_region), dtype=np.uint8)
        img = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
        
        results = self.ocr_reader.readtext(img)
        text = ' '.join([r[1].lower() for r in results if r[2] >= 0.9])
        return 'continue' in text
    
    def check_start(self):
        """
        Check for start screen - EXACT logic from detectStart.py hasStart()
        """
        raw = np.array(self.sct.grab(self.menu_region), dtype=np.uint8)
        img = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
        
        results = self.ocr_reader.readtext(img)
        text = ' '.join([r[1].lower() for r in results if r[2] >= 0.9])
        return 'play' in text or 'start' in text
    
    def get_score(self):
        """
        Get score - EXACT logic from detectScore.py getScore()
        """
        raw = np.array(self.sct.grab(self.score_region), dtype=np.uint8)
        img = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
        
        # Mask to extract white/bright pixels only
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)
        white_only = cv2.bitwise_and(img, img, mask=mask)
        
        results = self.ocr_reader.readtext(white_only)
        digits = ''.join(filter(str.isdigit, ''.join([r[1] for r in results])))
        return int(digits) if digits else 0


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                        RAINBOW COMPONENT 5: NOISY NETWORKS                    ║
# ╠══════════════════════════════════════════════════════════════════════════════╣
# ║  Noisy Nets replace the usual epsilon-greedy exploration method of picking   ║
# ║  random actions. Instead, it adds learned noise directly into the network's  ║
# ║  linear layers. This means the agent explores by slightly changing its       ║
# ║  Q-value estimates, and the amount of noise also depends on the state.       ║
# ║  Over time, the network can learn to reduce noise in familiar situations     ║
# ║  and keep it in unfamiliar ones. This creates a more robust model which is   ║
# ║  especially useful in Agar.io where the agent often encounters latency       ║
# ║  issues and needs to learn to make the best call/prediction.                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class NoisyLinear(nn.Module):
    """
    Noisy Linear Layer (Fortunato et al., 2018)
    """
    
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
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                     RAINBOW COMPONENT 4: DISTRIBUTIONAL RL (C51)              ║
# ╠══════════════════════════════════════════════════════════════════════════════╣
# ║  Distributional RL changes prediction: instead of predicting just one        ║
# ║  number for the expected return, it predicts a whole distribution of         ║
# ║  possible returns. This matters because in games like Agar.io, the same      ║
# ║  action can lead to very different outcomes depending on what nearby         ║
# ║  players do. By learning a distribution instead of a single average, the     ║
# ║  bot captures this uncertainty. In practice, the network outputs a set of    ║
# ║  probabilities over many "atoms," each representing a possible return        ║
# ║  value. Training then pushes this predicted distribution to match the true   ║
# ║  one, making learning more stable and helping the agent understand risky     ║
# ║  versus safe situations more clearly.                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class RainbowNet(nn.Module):
    """
    C51 Distributional RL
    """
    
    def __init__(self, state_dim, action_dim, n_atoms=51, v_min=-100, v_max=100):
        super().__init__()
        
        self.action_dim = action_dim
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        
        self.register_buffer('support', torch.linspace(v_min, v_max, n_atoms))
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        
        # Shared features
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        
        # Dueling streams with noisy layers
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


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                    RAINBOW COMPONENT 2: PRIORITIZED REPLAY                   ║
# ╠══════════════════════════════════════════════════════════════════════════════╣
# ║  Prioritized Replay fixes the issue that DQN treats all past experiences     ║
# ║  as equally important. In reality, some transitions need to be learned from  ║
# ║  much more than others. If the agent made a large prediction error on a      ║
# ║  transition, it should revisit that transition more often. So instead of     ║
# ║  sampling uniformly from the replay buffer, we sample based on how           ║
# ║  surprising each transition was (represented by TD-error). Bigger mistakes   ║
# ║  and more recent ones get higher priority, which optimizes learning.         ║
# ║                                                                              ║
# ║  Implementation uses a Sum Tree data structure for O(log n) sampling by      ║
# ║  priority, with importance sampling weights to correct for the bias          ║
# ║  introduced by non-uniform sampling.                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class SumTree:
    """
    Sum Tree for O(log n) priority sampling.
    """
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write_idx = 0
        self.n_entries = 0
    
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self):
        return self.tree[0]
    
    def add(self, priority, data):
        idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = data
        self.update(idx, priority)
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)
    
    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                    RAINBOW COMPONENT 3: MULTI-STEP LEARNING                  ║
# ╠══════════════════════════════════════════════════════════════════════════════╣
# ║  Multi-step learning lets the agent look a few steps into the future         ║
# ║  instead of learning from just one reward at a time. Instead of updating     ║
# ║  from only the next step, it adds up several rewards before bootstrapping.   ║
# ║  This is especially helpful in the Agar.io context because often taking an   ║
# ║  action that looks good right now, like moving toward food, can be a bad     ║
# ║  idea a few steps later if it pushes you straight into a larger player.      ║
# ║  Multi-step learning helps capture this, because instead of updating from    ║
# ║  only the immediate reward of grabbing food, the agent also sees the         ║
# ║  consequences—a death screen recorded later.                                 ║
# ║                                                                              ║
# ║  N-step return: R = r₁ + γr₂ + γ²r₃ + ... + γⁿ⁻¹rₙ + γⁿV(sₙ)                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay combined with N-step returns.
    """
    
    def __init__(self, capacity, n_steps=3, gamma=0.99, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.n_steps = n_steps
        self.gamma = gamma
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 0
        self.epsilon = 1e-5
        self.max_priority = 1.0
        self.n_step_buffer = deque(maxlen=n_steps)
    
    @property
    def beta(self):
        return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
    
    def push(self, state, action, reward, next_state, done):
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        if len(self.n_step_buffer) == self.n_steps:
            self._add_n_step_transition()
        
        if done:
            self._flush_n_step_buffer()
    
    def _add_n_step_transition(self):
        if len(self.n_step_buffer) == 0:
            return
        
        s_0, a_0, _, _, _ = self.n_step_buffer[0]
        n_step_return = 0.0
        final_state = None
        final_done = False
        
        for i, (s, a, r, ns, d) in enumerate(self.n_step_buffer):
            n_step_return += (self.gamma ** i) * r
            final_state = ns
            final_done = d
            if d:
                break
        
        experience = (s_0, a_0, n_step_return, final_state, final_done)
        self.tree.add(self.max_priority ** self.alpha, experience)
    
    def _flush_n_step_buffer(self):
        while len(self.n_step_buffer) > 0:
            s_0, a_0, _, _, _ = self.n_step_buffer[0]
            n_step_return = 0.0
            final_state = None
            final_done = False
            
            for i, (s, a, r, ns, d) in enumerate(self.n_step_buffer):
                n_step_return += (self.gamma ** i) * r
                final_state = ns
                final_done = d
            
            experience = (s_0, a_0, n_step_return, final_state, final_done)
            self.tree.add(self.max_priority ** self.alpha, experience)
            self.n_step_buffer.popleft()
    
    def sample(self, batch_size):
        self.frame += 1
        batch = []
        indices = []
        priorities = []
        
        segment = self.tree.total() / batch_size
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, priority, data = self.tree.get(s)
            
            if data is None or data == 0:
                s = random.uniform(0, self.tree.total())
                idx, priority, data = self.tree.get(s)
                if data is None or data == 0:
                    continue
            
            batch.append(data)
            indices.append(idx)
            priorities.append(priority)
        
        if len(batch) == 0:
            return None
        
        probs = np.array(priorities) / (self.tree.total() + 1e-8)
        weights = (self.tree.n_entries * probs) ** (-self.beta)
        weights = weights / (weights.max() + 1e-8)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
            np.array(next_states), np.array(dones, dtype=np.float32),
            indices, np.array(weights, dtype=np.float32)
        )
    
    def update_priorities(self, indices, td_errors):
        for idx, error in zip(indices, td_errors):
            priority = (abs(error) + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)
    
    def __len__(self):
        return self.tree.n_entries


# ============== RAINBOW DQN AGENT ==============
class RainbowAgent:
    """Rainbow DQN Agent with all 6 improvements"""
    
    N_ATOMS = 51
    V_MIN = -100
    V_MAX = 100
    N_STEPS = 3
    
    def __init__(self, state_dim, action_dim, lr=0.0001):
        self.action_dim = action_dim
        self.gamma = 0.99
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.online = RainbowNet(state_dim, action_dim, self.N_ATOMS, self.V_MIN, self.V_MAX).to(self.device)
        self.target = RainbowNet(state_dim, action_dim, self.N_ATOMS, self.V_MIN, self.V_MAX).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.buffer = PrioritizedReplayBuffer(capacity=100000, n_steps=self.N_STEPS, gamma=self.gamma)
        
        self.step_count = 0
        self.target_update_freq = 1000
        
        self.support = torch.linspace(self.V_MIN, self.V_MAX, self.N_ATOMS).to(self.device)
        self.delta_z = (self.V_MAX - self.V_MIN) / (self.N_ATOMS - 1)
        
        # Epsilon-greedy for initial exploration before noisy nets learn
        # Decays from 1.0 to 0.01 over first 5000 steps
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 5000  # Steps to decay over
    
    def act(self, state):
        # Epsilon-greedy exploration (especially important early in training)
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        # Reset noise before action selection for exploration
        self.online.reset_noise()
        
        # Keep model in training mode for noisy layers to apply noise
        self.online.train()
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.online.get_q_values(state_t)
            return q_values.argmax(1).item()
    
    def act_with_q(self, state):
        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            action = random.randint(0, self.action_dim - 1)
            return action, 0.0  # Return 0 for Q-value when random
        
        # Reset noise before action selection
        self.online.reset_noise()
        
        # Keep model in training mode for noisy layers
        self.online.train()
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.online.get_q_values(state_t)
            action = q_values.argmax(1).item()
            q_value = q_values[0, action].item()
            return action, q_value
    
    def decay_epsilon(self):
        """Decay epsilon after each step"""
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, 
                              1.0 - (self.step_count / self.epsilon_decay) * (1.0 - self.epsilon_min))
    
    def learn(self, batch_size=32):
        if len(self.buffer) < batch_size:
            return 0.0, 0.0
        
        self.step_count += 1
        self.decay_epsilon()  # Decay exploration rate
        
        sample = self.buffer.sample(batch_size)
        if sample is None:
            return 0.0, 0.0
        
        states, actions, rewards, next_states, dones, indices, weights = sample
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        self.online.reset_noise()
        self.target.reset_noise()
        
        batch_size_actual = states.size(0)
        current_dist = self.online(states)
        current_dist = current_dist[range(batch_size_actual), actions]
        
        # ╔══════════════════════════════════════════════════════════════════════╗
        # ║              RAINBOW COMPONENT 1: DOUBLE Q-LEARNING                  ║
        # ╠══════════════════════════════════════════════════════════════════════╣
        # ║  This step fixes overestimation that happens because Q-learning      ║
        # ║  updates by picking the action with the biggest predicted value,     ║
        # ║  and using it as the target. However, when the predictions have      ║
        # ║  noise then some actions will accidentally have values that are      ║
        # ║  too high. These inflated values then get used as training targets,  ║
        # ║  causing the network to learn them as if they were true.             ║
        # ║                                                                      ║
        # ║  Double Q-Learning fixes this by defining a "pick and evaluate"      ║
        # ║  system involving two separate networks:                             ║
        # ║    - ONLINE network SELECTS the best action: argmax_a Q_online(s',a) ║
        # ║    - TARGET network EVALUATES that action: Q_target(s', a*)          ║
        # ║                                                                      ║
        # ║  This decoupling minimizes maximization bias because the target      ║
        # ║  network's noise is independent of the selection network's noise.    ║
        # ╚══════════════════════════════════════════════════════════════════════╝
        with torch.no_grad():
            """
            Double Q-Learning:
            """
            # ONLINE network SELECTS the best action for next state
            next_q = self.online.get_q_values(next_states)
            next_actions = next_q.argmax(1)  # a* = argmax Q_online(s', a)
            
            # TARGET network EVALUATES that action (not its own argmax!)
            next_dist = self.target(next_states)
            next_dist = next_dist[range(batch_size_actual), next_actions]  # Q_target(s', a*)
            
            gamma_n = self.gamma ** self.N_STEPS
            Tz = rewards.unsqueeze(1) + gamma_n * (1 - dones.unsqueeze(1)) * self.support.unsqueeze(0)
            Tz = Tz.clamp(self.V_MIN, self.V_MAX)
            
            b = (Tz - self.V_MIN) / self.delta_z
            l = b.floor().long().clamp(0, self.N_ATOMS - 1)
            u = b.ceil().long().clamp(0, self.N_ATOMS - 1)
            
            target_dist = torch.zeros_like(next_dist)
            
            eq_mask = (l == u)
            target_dist.scatter_add_(1, l, next_dist * eq_mask.float())
            
            neq_mask = ~eq_mask
            target_dist.scatter_add_(1, l, next_dist * (u.float() - b) * neq_mask.float())
            target_dist.scatter_add_(1, u, next_dist * (b - l.float()) * neq_mask.float())
        
        loss = -(target_dist * (current_dist + 1e-8).log()).sum(1)
        weighted_loss = (weights * loss).mean()
        
        self.optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        
        td_errors = loss.detach().cpu().numpy()
        self.buffer.update_priorities(indices, td_errors)
        
        if self.step_count % self.target_update_freq == 0:
            self.target.load_state_dict(self.online.state_dict())
        
        return weighted_loss.item(), np.mean(td_errors)
    
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'online': self.online.state_dict(),
            'target': self.target.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step_count': self.step_count,
            'epsilon': self.epsilon,
        }, path)
        print(f"✓ Saved: {path} (ε={self.epsilon:.3f})")
    
    def load(self, path):
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            self.online.load_state_dict(ckpt['online'])
            self.target.load_state_dict(ckpt['target'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.step_count = ckpt.get('step_count', 0)
            self.epsilon = ckpt.get('epsilon', 0.01)  # Default to low epsilon if not saved
            self.decay_epsilon()  # Ensure epsilon is in sync with step_count
            print(f"✓ Loaded: {path} (step={self.step_count}, ε={self.epsilon:.3f})")
            return True
        except Exception as e:
            print(f"✗ Failed to load {path}: {e}")
            return False


# ============== AGARIO ENVIRONMENT (4 Cardinal Directions) ==============
class AgarioEnv:
    """
    Agar.io environment with 4 cardinal direction actions.
    
    State: 18-dim vector (food + virus info)
    Actions: UP, DOWN, LEFT, RIGHT
    """
    
    # Button locations (from detectContinue.py / detectStart.py comments)
    CONTINUE_CLICK = (759, -566)
    START_CLICK = (754, -655)
    
    ACTION_DURATION = 0.25  # 4 actions/sec
    
    def __init__(self):
        self.detector = GameDetector()
        
        # Screen center for mouse movement
        self.screen_center_x = self.detector.mon_left + self.detector.mon_width / 2
        self.screen_center_y = self.detector.mon_top + self.detector.mon_height / 2
        
        # Movement offset (20% of screen)
        self.move_offset = min(self.detector.mon_width, self.detector.mon_height) * 0.20
        
        # 4 Cardinal directions
        self.action_names = ["UP", "DOWN", "LEFT", "RIGHT"]
        self.action_dim = 4
        self.state_dim = 18
        
        # Tracking
        self.score = 0
        self.last_score = 0
        self.steps_alive = 0
        self.episode_start = time.time()
        
        # Cache for state
        self.last_food = []
        self.last_virus = None
        self.last_food_count = 0
    
    def _execute_action(self, action):
        """Move mouse in cardinal direction"""
        if action == 0:  # UP
            pyautogui.moveTo(self.screen_center_x, self.screen_center_y - self.move_offset)
        elif action == 1:  # DOWN
            pyautogui.moveTo(self.screen_center_x, self.screen_center_y + self.move_offset)
        elif action == 2:  # LEFT
            pyautogui.moveTo(self.screen_center_x - self.move_offset, self.screen_center_y)
        elif action == 3:  # RIGHT
            pyautogui.moveTo(self.screen_center_x + self.move_offset, self.screen_center_y)
    
    def _build_state(self, food_list, virus, food_count):
        """
        Build 18-dim state vector:
        [0-13]  7 food (dx, dy) normalized
        [14-15] virus (dx, dy) normalized
        [16]    food_count / 100
        [17]    has_virus (0 or 1)
        """
        state = np.zeros(18, dtype=np.float32)
        
        max_dist = self.detector.center_x  # Use half screen as normalization
        
        # 7 food items (14 values)
        for i in range(7):
            if i < len(food_list):
                dx, dy = food_list[i]
                state[i * 2] = np.clip(dx / max_dist, -1, 1)
                state[i * 2 + 1] = np.clip(dy / max_dist, -1, 1)
        
        # Virus (2 values)
        if virus:
            state[14] = np.clip(virus[0] / max_dist, -1, 1)
            state[15] = np.clip(virus[1] / max_dist, -1, 1)
        
        # Food density
        state[16] = min(food_count / 100.0, 1.0)
        
        # Has virus flag
        state[17] = 1.0 if virus else 0.0
        
        return state
    
    def _get_state(self):
        """Get current state from detector"""
        food_list, virus, food_count = self.detector.detect_game_state()
        
        self.last_food = food_list
        self.last_virus = virus
        self.last_food_count = food_count
        
        return self._build_state(food_list, virus, food_count)
    
    def reset(self):
        """Reset for new episode - handle continue/start screens"""
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
        
        self.score = 0
        self.last_score = 0
        self.steps_alive = 0
        self.episode_start = time.time()
        
        return self._get_state()
    
    def step(self, action):
        """Execute action and return (state, reward, done, info)"""
        # Execute action for duration
        start_time = time.time()
        while time.time() - start_time < self.ACTION_DURATION:
            self._execute_action(action)
            time.sleep(0.05)
        
        # Get new state
        state = self._get_state()
        self.steps_alive += 1
        
        # Check for death (menu screens)
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
        
        # Calculate reward
        reward = 0.0
        
        # Survival bonus
        reward += 0.5
        reward += 0.1 * math.log(1 + self.steps_alive)
        
        # Food proximity reward - encourage moving toward food
        if self.last_food:
            nearest_dist = math.sqrt(self.last_food[0][0]**2 + self.last_food[0][1]**2)
            if nearest_dist < 100:
                reward += 0.3 * (100 - nearest_dist) / 100
        
        # Food density bonus
        reward += 0.1 * min(self.last_food_count / 50.0, 1.0)
        
        # Virus avoidance (small penalty if near virus)
        if self.last_virus:
            virus_dist = math.sqrt(self.last_virus[0]**2 + self.last_virus[1]**2)
            if virus_dist < 150:
                reward -= 0.2 * (150 - virus_dist) / 150
        
        # Milestone bonus
        if self.steps_alive % 40 == 0:
            reward += 2.0
        
        # Death penalty
        if done:
            reward = -50.0
        
        info = {
            'score': self.score,
            'steps_alive': self.steps_alive,
            'time_alive': time.time() - self.episode_start,
            'food_count': self.last_food_count,
            'action': self.action_names[action]
        }
        
        return state, reward, done, info


# ============== TRAINING ==============
def find_latest_checkpoint():
    """Find latest checkpoint file"""
    import glob
    import re
    
    ckpts = []
    for f in glob.glob("checkpoints/rainbow_ep*.pth"):
        m = re.search(r'ep(\d+)', f)
        if m:
            ckpts.append((int(m.group(1)), f))
    
    if not ckpts:
        return None, 0
    
    ckpts.sort(key=lambda x: -x[0])
    return ckpts[0][1], ckpts[0][0]


def train(episodes=10000, max_steps=600):
    """Main training loop"""
    global stop_training
    
    print("=" * 70)
    print("🌈 RAINBOW DQN for Agar.io 🌈")
    print("=" * 70)
    print()
    print("Components:")
    print("  ✓ Double DQN")
    print("  ✓ Prioritized Experience Replay")
    print("  ✓ Dueling Networks")
    print("  ✓ Multi-step Learning (N=3)")
    print("  ✓ Distributional C51")
    print("  ✓ Noisy Networks (no epsilon!)")
    print()
    print("Actions: UP, DOWN, LEFT, RIGHT")
    print("State: 18 dims (food + virus)")
    print("Press Q to save and quit")
    print("=" * 70)
    
    env = AgarioEnv()
    agent = RainbowAgent(env.state_dim, env.action_dim)
    logger = CSVLogger()
    
    ckpt_path, start_ep = find_latest_checkpoint()
    if ckpt_path:
        agent.load(ckpt_path)
        print(f"Resuming from episode {start_ep + 1}")
    else:
        print("Starting fresh training")
    
    print("\nStarting in 3 seconds...")
    time.sleep(3)
    
    best_survival = 0
    
    for ep in range(start_ep, episodes):
        if stop_training:
            agent.save("checkpoints/rainbow_interrupted.pth")
            break
        
        state = env.reset()
        ep_reward = 0
        losses = []
        td_errors = []
        
        for step in range(max_steps):
            if stop_training:
                break
            
            action, q_value = agent.act_with_q(state)
            next_state, reward, done, info = env.step(action)
            
            agent.buffer.push(state, action, reward, next_state, done)
            
            loss, td_err = agent.learn()
            if loss > 0:
                losses.append(loss)
                td_errors.append(td_err)
            
            logger.log(ep, step, action, env.action_names[action], reward, done, info,
                      state, q_value, td_err)
            
            ep_reward += reward
            
            if step % 20 == 0:
                avg_l = np.mean(losses[-20:]) if losses else 0
                print(f"\rEp{ep+1} s{step:3d} | {info['action']:5s} "
                      f"Food={info['food_count']:3d} Alive={info['steps_alive']:3d} | "
                      f"ε={agent.epsilon:.2f} L={avg_l:.4f} Q={q_value:.2f}", end="", flush=True)
            
            state = next_state
            
            if done:
                print(f"\n💀 Death at step {step}!")
                break
        
        if stop_training:
            break
        
        survival = info.get('steps_alive', step)
        if survival > best_survival:
            best_survival = survival
        
        avg_loss = np.mean(losses) if losses else 0
        
        print(f"\nEp{ep+1:4d} | Survived: {survival:3d} (best: {best_survival}) | "
              f"ε={agent.epsilon:.3f} Loss={avg_loss:.4f} | Buf={len(agent.buffer)}")
        
        if (ep + 1) % 10 == 0:
            agent.save(f"checkpoints/rainbow_ep{ep+1}.pth")
    
    logger.close()
    
    if not stop_training:
        agent.save("checkpoints/rainbow_final.pth")
        print("\n✓ Training complete!")


if __name__ == "__main__":
    train()
