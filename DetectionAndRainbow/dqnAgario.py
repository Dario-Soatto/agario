"""
Semantic Action DQN for Agar.io - with Temporal Credit Assignment
==================================================================
State Vector (26 dims):
  [0-8]   3 nearest players: (dx, dy, radius) each
  [9-10]  1 nearest virus: (dx, dy)
  [11-24] 7 nearest food: (dx, dy) each
  [25]    self radius

Action Space (5 semantic actions):
  0: Move toward nearest food (EAT_FOOD)
  1: Move toward smaller player (HUNT_PREY)
  2: Move away from larger player (FLEE_THREAT)
  3: Move toward nearest virus (TO_VIRUS)
  4: Move away from nearest virus (FROM_VIRUS)

Credit Assignment Features:
  - Action Repeat: Each action executes for ACTION_DURATION seconds
  - N-step Returns: Rewards propagate back N_STEPS to the causing action
  - This ensures "move toward food" gets credit, not the random flail before eating

Reward: 80% score change + 20% radius change, death = -100
Press Q to save and quit
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import time
import os
from pynput import keyboard
import cv2
import mss
import pyautogui

from detectStart import mssStartButton
from detectScore import mssBottomLeftCorner
from detectContinue import mssMiddleRectangle

# ============== Q TO QUIT ==============
stop_training = False
def on_press(key):
    global stop_training
    try:
        if key.char == 'q':
            stop_training = True
    except:
        pass
keyboard.Listener(on_press=on_press).start()


# ============== UNIFIED DETECTOR (SINGLE MSS CALL) ==============
class UnifiedDetector:
    """
    Single screen capture for ALL detections:
    - Self (center blob)
    - Players (large blobs, not self, not virus)
    - Food (small blobs)
    - Viruses (green #12F448)
    """
    
    # Virus color #12F448 = BGR(72, 244, 18)
    VIRUS_BGR = (72, 244, 18)
    VIRUS_TOL = 40
    
    def __init__(self):
        self.sct = mss.mss()  # Persistent - no recreation
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
        """
        Single capture, returns everything:
        - self_radius: int or None
        - players: [(dx, dy, radius), ...] max 3, relative coords
        - food: [(dx, dy), ...] max 7, relative coords
        - virus: (dx, dy) or None, relative coords
        """
        # Single capture
        raw = np.array(self.sct.grab(self.region), dtype=np.uint8)
        img = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
        
        # Find all blobs
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
            
            # Check if virus (green #12F448)
            is_virus = False
            if 0 <= y < self.h and 0 <= x < self.w:
                px = img[y, x]
                if (abs(int(px[0]) - self.VIRUS_BGR[0]) < self.VIRUS_TOL and
                    abs(int(px[1]) - self.VIRUS_BGR[1]) < self.VIRUS_TOL and
                    abs(int(px[2]) - self.VIRUS_BGR[2]) < self.VIRUS_TOL):
                    is_virus = True
                    if r > 30:  # Real virus, not noise
                        dist = ((x - self.cx)**2 + (y - self.cy)**2)**0.5
                        viruses.append((x - self.cx, y - self.cy, dist))
            
            if is_virus:
                continue
            
            dist = ((x - self.cx)**2 + (y - self.cy)**2)**0.5
            
            # Self detection: blob closest to center
            if r >= 10 and dist < min(self.w, self.h) * 0.15:
                if self_blob is None or dist < self_blob[3]:
                    self_blob = (x, y, r, dist)
            
            # Players: radius >= 30, not at center
            elif r >= 30:
                players.append((x - self.cx, y - self.cy, r, dist))
            
            # Food: small blobs
            elif 2 <= r < 30:
                food.append((x - self.cx, y - self.cy, dist))
        
        # Sort by distance
        players.sort(key=lambda p: p[3])
        food.sort(key=lambda f: f[2])
        viruses.sort(key=lambda v: v[2])
        
        # Format output
        self_radius = self_blob[2] if self_blob else None
        players_out = [(p[0], p[1], p[2]) for p in players[:3]]  # (dx, dy, radius)
        food_out = [(f[0], f[1]) for f in food[:7]]  # (dx, dy)
        virus_out = (viruses[0][0], viruses[0][1]) if viruses else None  # (dx, dy)
        
        return self_radius, players_out, food_out, virus_out


# ============== DUELING DQN NETWORK ==============
class QNet(nn.Module):
    """
    Dueling DQN Architecture (Wang et al., 2016)
    https://arxiv.org/abs/1511.06581
    
    Implements Equation (9) from the paper:
    Q(s,a) = V(s) + (A(s,a) - mean_a'(A(s,a')))
    
    Key design choices from paper:
    - Separate value V(s) and advantage A(s,a) streams
    - Mean subtraction for identifiability (Eq. 9)
    - Gradient rescaling by 1/sqrt(2) at stream split
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        
        # Shared feature extraction (replaces conv layers for vector input)
        # Paper uses 512 units; we use 512 to match
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        
        # Value stream V(s): scalar output
        # Paper: "fully-connected layer with 512 units"
        self.value_stream = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        # Advantage stream A(s,a): |A| outputs
        # Paper: "as many outputs as there are valid actions"
        self.advantage_stream = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )
        
        # Scaling factor from paper: "rescale combined gradient by 1/sqrt(2)"
        self.scale = 1.0 / (2 ** 0.5)
    
    def forward(self, x):
        features = self.feature(x)
        
        # Scale features before splitting (paper section 4.2)
        features = features * self.scale
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Equation (9): Q = V + (A - mean(A))
        # "increases stability of optimization" - paper
        q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q


# ============== N-STEP REPLAY BUFFER ==============
class NStepReplayBuffer:
    """
    N-step replay buffer for better credit assignment.
    
    Instead of storing (s, a, r, s'), we compute n-step returns:
    R_n = r_1 + γr_2 + γ²r_3 + ... + γ^(n-1)r_n
    
    Then store (s_1, a_1, R_n, s_n, done).
    
    This propagates rewards BACKWARD to the action that caused them,
    solving the "random flail before food gets credit" problem.
    """
    
    def __init__(self, cap=100000, n_steps=5, gamma=0.99):
        self.buf = deque(maxlen=cap)
        self.n_steps = n_steps
        self.gamma = gamma
        
        # Temporary buffer to accumulate n steps
        self.n_step_buffer = deque(maxlen=n_steps)
    
    def push(self, s, a, r, ns, d):
        """
        Add transition to n-step buffer.
        When we have n steps, compute n-step return and add to main buffer.
        """
        self.n_step_buffer.append((s, a, r, ns, d))
        
        # If we have enough steps, compute n-step return
        if len(self.n_step_buffer) == self.n_steps:
            self._compute_and_store_nstep()
        
        # If episode ended, flush remaining transitions
        if d:
            self._flush_nstep_buffer()
    
    def _compute_and_store_nstep(self):
        """Compute n-step return from the buffer and store."""
        if len(self.n_step_buffer) == 0:
            return
        
        # Get first transition (the one that will get credit)
        s_0, a_0, _, _, _ = self.n_step_buffer[0]
        
        # Compute n-step return: R = r_1 + γr_2 + γ²r_3 + ...
        n_step_return = 0.0
        final_done = False
        final_next_state = None
        
        for i, (s, a, r, ns, d) in enumerate(self.n_step_buffer):
            n_step_return += (self.gamma ** i) * r
            final_next_state = ns
            final_done = d
            if d:  # Episode ended early
                break
        
        # Store: (initial_state, initial_action, n_step_return, final_state, done)
        self.buf.append((s_0, a_0, n_step_return, final_next_state, final_done))
    
    def _flush_nstep_buffer(self):
        """Flush remaining transitions at episode end."""
        while len(self.n_step_buffer) > 0:
            # Compute partial n-step return with remaining transitions
            s_0, a_0, _, _, _ = self.n_step_buffer[0]
            
            n_step_return = 0.0
            final_done = False
            final_next_state = None
            
            for i, (s, a, r, ns, d) in enumerate(self.n_step_buffer):
                n_step_return += (self.gamma ** i) * r
                final_next_state = ns
                final_done = d
            
            self.buf.append((s_0, a_0, n_step_return, final_next_state, final_done))
            self.n_step_buffer.popleft()
    
    def sample(self, n):
        batch = random.sample(self.buf, min(n, len(self.buf)))
        s, a, r, ns, d = zip(*batch)
        return (np.array(s), np.array(a), np.array(r, dtype=np.float32),
                np.array(ns), np.array(d, dtype=np.float32))
    
    def __len__(self):
        return len(self.buf)


# Legacy single-step buffer (kept for compatibility)
class ReplayBuffer:
    def __init__(self, cap=100000):
        self.buf = deque(maxlen=cap)
    
    def push(self, s, a, r, ns, d):
        self.buf.append((s, a, r, ns, d))
    
    def sample(self, n):
        batch = random.sample(self.buf, n)
        s, a, r, ns, d = zip(*batch)
        return (np.array(s), np.array(a), np.array(r, dtype=np.float32),
                np.array(ns), np.array(d, dtype=np.float32))
    
    def __len__(self):
        return len(self.buf)


# ============== ENVIRONMENT ==============
class AgarioEnv:
    """
    State Vector (26 dims):
      [0-8]   3 players: (dx, dy, radius) - normalized
      [9-10]  1 virus: (dx, dy) - normalized
      [11-24] 7 food: (dx, dy) - normalized
      [25]    self radius - normalized
    
    Action Space (5 semantic actions):
      0: EAT_FOOD - Move toward nearest food
      1: HUNT_PREY - Move toward smaller player
      2: FLEE_THREAT - Move away from larger player
      3: TO_VIRUS - Move toward nearest virus
      4: FROM_VIRUS - Move away from nearest virus
    
    Credit Assignment:
      - ACTION_DURATION: How long each action executes (seconds)
      - Actions have time to "play out" before next decision
      - Combined with N-step returns in the buffer
    """
    
    CONTINUE_CLICK = (759, -566)
    START_CLICK = (754, -655)
    
    # === TIMING PARAMETERS ===
    # 4 actions per second = faster reaction time
    ACTION_DURATION = 0.25  # Each action executes for 0.25 seconds (4 actions/sec)
    ACTION_REFRESH_RATE = 0.05  # Re-execute every 50ms for smooth movement
    
    # === SAFETY THRESHOLDS (CONSERVATIVE) ===
    # These thresholds determine what's safe to chase vs what to flee from
    SMALLER_THRESHOLD = 0.7   # Player must be 30% SMALLER to hunt (very safe)
    LARGER_THRESHOLD = 1.0    # ANY player equal or larger = THREAT (flee immediately)
    DANGER_DISTANCE = 200     # Pixels - if threat is this close, we're in danger
    
    def __init__(self):
        self.detector = UnifiedDetector()
        self.start_det = mssStartButton()
        self.score_det = mssBottomLeftCorner()
        self.cont_det = mssMiddleRectangle()
        
        self.w = self.detector.w
        self.h = self.detector.h
        
        # Get FULL monitor dimensions for mouse movements (not padded region)
        with mss.mss() as sct:
            mon = sct.monitors[2] if len(sct.monitors) > 2 else sct.monitors[1]
            self.mon_w = mon['width']
            self.mon_h = mon['height']
            self.mon_left = mon['left']
            self.mon_top = mon['top']
        
        # Center of the FULL monitor (where player always is)
        self.center_x = self.mon_left + self.mon_w / 2
        self.center_y = self.mon_top + self.mon_h / 2
        
        # Store raw detections for semantic actions (initialized empty)
        self.last_self_radius = None
        self.last_players = []      # list of (dx, dy, radius) sorted by distance
        self.last_food = []         # list of (dx, dy) sorted by distance
        self.last_virus = None      # (dx, dy) or None
        
        # Semantic action space - 5 actions based on game entities
        self.actions = [
            self._act_toward_nearest_food,      # 0: Eat food
            self._act_toward_smaller_player,    # 1: Hunt prey
            self._act_away_from_larger_player,  # 2: Flee from threats
            self._act_toward_virus,             # 3: Approach virus (can split enemies)
            self._act_away_from_virus,          # 4: Avoid viruses (danger when large)
        ]
        
        self.action_names = [
            "EAT_FOOD", "HUNT_PREY", "FLEE_THREAT", "TO_VIRUS", "FROM_VIRUS"
        ]
        
        self.state_dim = 26
        self.action_dim = len(self.actions)  # 5
        
        # Tracking
        self.self_radius = 60
        self.last_radius = 60
        self.score = 0
        self.last_score = 0
        self.last_check = 0
        
        # Survival tracking for reward shaping
        self.steps_alive = 0
        self.episode_start_time = time.time()
        self.last_milestone = 0
        
        # Track if actually playing (not in menu) - prevents recording menu actions
        self.is_playing = True
        
        # For printout
        self.food_count = 0
        self.player_count = 0
    
    # ============== MOUSE MOVEMENT HELPERS ==============
    def _move_to_relative(self, dx, dy, scale=1.0):
        """
        Move mouse to position offset from screen center by (dx, dy).
        dx, dy are in detector's pixel coordinate system (relative to center).
        """
        # Scale the offset (can amplify or reduce movement)
        sx = dx * scale
        sy = dy * scale
        
        # Target position = screen center + offset
        target_x = self.center_x + sx
        target_y = self.center_y + sy
        
        # Clamp to monitor bounds for safety
        target_x = max(self.mon_left, min(self.mon_left + self.mon_w - 1, target_x))
        target_y = max(self.mon_top, min(self.mon_top + self.mon_h - 1, target_y))
        
        pyautogui.moveTo(target_x, target_y)
    
    def _move_toward(self, dx, dy):
        """Move mouse toward entity at (dx, dy) relative to center"""
        self._move_to_relative(dx, dy, scale=1.0)
    
    def _move_away(self, dx, dy):
        """Move mouse away from entity at (dx, dy) - opposite direction"""
        self._move_to_relative(-dx, -dy, scale=1.0)
    
    # ============== SEMANTIC ACTION FUNCTIONS ==============
    def _act_toward_nearest_food(self):
        """Action 0: Move toward nearest food pellet - but avoid threats!"""
        # SAFETY CHECK: If a threat is very close, flee instead!
        if self.last_players:
            for (px, py, pr) in self.last_players:
                dist = (px**2 + py**2)**0.5
                if self.self_radius and pr >= self.self_radius * self.LARGER_THRESHOLD:
                    if dist < self.DANGER_DISTANCE * 0.75:  # Very close threat
                        # Emergency flee!
                        return self._act_away_from_larger_player()
        
        if self.last_food:
            # Find food that's not in the direction of a threat
            best_food = None
            for (fx, fy) in self.last_food[:5]:  # Check top 5 closest
                food_safe = True
                for (px, py, pr) in self.last_players:
                    if self.self_radius and pr >= self.self_radius * self.LARGER_THRESHOLD:
                        # Is this food in the direction of a threat?
                        # Simple check: same quadrant
                        if (fx * px > 0 and fy * py > 0):  # Same quadrant
                            food_safe = False
                            break
                if food_safe:
                    best_food = (fx, fy)
                    break
            
            if best_food:
                self._move_toward(best_food[0], best_food[1])
            elif self.last_food:
                # No safe food, just go to nearest
                self._move_toward(self.last_food[0][0], self.last_food[0][1])
        else:
            # No food visible - stay put or wander
            pass
    
    def _act_toward_smaller_player(self):
        """Action 1: Hunt - move toward a SIGNIFICANTLY smaller player (prey)"""
        if not self.last_players:
            # No players visible, fallback to eating food (SAFE)
            return self._act_toward_nearest_food()
        
        # SAFETY FIRST: Check if there are any threats nearby
        # If a larger player is close, DON'T HUNT - FLEE instead
        for (dx, dy, r) in self.last_players:
            dist = (dx**2 + dy**2)**0.5
            if self.self_radius and r >= self.self_radius * self.LARGER_THRESHOLD:
                if dist < self.DANGER_DISTANCE:
                    # DANGER! A threat is close - flee instead of hunting
                    return self._act_away_from_larger_player()
        
        # Find nearest player that's SIGNIFICANTLY smaller
        target = None
        for (dx, dy, r) in self.last_players:
            if self.self_radius and r < self.self_radius * self.SMALLER_THRESHOLD:
                target = (dx, dy)
                break
        
        if target is None:
            # No sufficiently smaller players - eat food instead (SAFE fallback)
            return self._act_toward_nearest_food()
        
        dx, dy = target
        self._move_toward(dx, dy)
    
    def _act_away_from_larger_player(self):
        """Action 2: Flee - move away from ANY player equal or larger (VERY SAFE)"""
        if not self.last_players:
            # No players visible, just eat food
            return self._act_toward_nearest_food()
        
        # Find ALL threats (equal size or larger) and flee from the closest one
        threats = []
        for (dx, dy, r) in self.last_players:
            if self.self_radius and r >= self.self_radius * self.LARGER_THRESHOLD:
                dist = (dx**2 + dy**2)**0.5
                threats.append((dx, dy, r, dist))
        
        if threats:
            # Flee from the CLOSEST threat (most dangerous)
            threats.sort(key=lambda t: t[3])
            dx, dy = threats[0][0], threats[0][1]
            self._move_away(dx, dy)
        else:
            # No threats - avoid the nearest player anyway (conservative)
            # This teaches the agent that avoiding players is generally good
            dx, dy, _ = self.last_players[0]
            self._move_away(dx, dy)
    
    def _act_toward_virus(self):
        """Action 3: Move toward nearest virus (strategic - can split enemies)"""
        if self.last_virus is None:
            # No virus visible, fallback to eating food
            return self._act_toward_nearest_food()
        
        dx, dy = self.last_virus
        self._move_toward(dx, dy)
    
    def _act_away_from_virus(self):
        """Action 4: Move away from nearest virus (dangerous when large)"""
        if self.last_virus is None:
            # No virus visible, fallback to eating food
            return self._act_toward_nearest_food()
        
        dx, dy = self.last_virus
        self._move_away(dx, dy)
    
    def _build_state(self, self_r, players, food, virus):
        """Build normalized 26-dim state vector"""
        state = np.zeros(26, dtype=np.float32)
        
        # Normalization factors
        max_dist = self.w  # Use screen width as max distance
        max_radius = 300   # Reasonable max player radius
        
        # [0-8] 3 players: (dx, dy, radius) normalized
        for i in range(3):
            if i < len(players):
                dx, dy, r = players[i]
                state[i*3] = np.clip(dx / max_dist, -1, 1)
                state[i*3 + 1] = np.clip(dy / max_dist, -1, 1)
                state[i*3 + 2] = min(r / max_radius, 1.0)
        
        # [9-10] 1 virus: (dx, dy) normalized
        if virus:
            state[9] = np.clip(virus[0] / max_dist, -1, 1)
            state[10] = np.clip(virus[1] / max_dist, -1, 1)
        
        # [11-24] 7 food: (dx, dy) normalized
        for i in range(7):
            if i < len(food):
                dx, dy = food[i]
                state[11 + i*2] = np.clip(dx / max_dist, -1, 1)
                state[11 + i*2 + 1] = np.clip(dy / max_dist, -1, 1)
        
        # [25] self radius normalized
        state[25] = min((self_r or 60) / max_radius, 1.0)
        
        return state
    
    def _get_state(self):
        """Single detection call, store raw detections, build state"""
        self_r, players, food, virus = self.detector.detect_all()
        
        if self_r:
            self.self_radius = self_r
        
        # Store raw detections for use by semantic action functions
        # These are in pixel coords relative to screen center
        self.last_self_radius = self_r
        self.last_players = players      # [(dx, dy, radius), ...] sorted by dist
        self.last_food = food            # [(dx, dy), ...] sorted by dist
        self.last_virus = virus          # (dx, dy) or None
        
        self.food_count = len(food)
        self.player_count = len(players)
        
        return self._build_state(self_r, players, food, virus)
    
    def reset(self):
        """
        Ensure we're in an active game - handle BOTH continue AND start screens.
        Sometimes game shows start screen directly without continue.
        """
        # Try multiple times to ensure we're in game
        for attempt in range(5):
            # Check continue screen first (appears after death)
            if self.cont_det.hasContinue():
                print(f"  [Reset] Found Continue screen, clicking...")
                pyautogui.click(*self.CONTINUE_CLICK)
                time.sleep(0.5)
            
            # Check start screen (might appear instead of or after continue)
            if self.start_det.hasStart():
                print(f"  [Reset] Found Start screen, clicking...")
                pyautogui.click(*self.START_CLICK)
                time.sleep(0.5)
            
            # Brief wait then check if we're actually in game
            time.sleep(0.3)
            
            # If neither screen is showing, we're probably in game
            if not self.cont_det.hasContinue() and not self.start_det.hasStart():
                break
        
        self.self_radius = 60
        self.last_radius = 60
        self.score = 0
        self.last_score = 0
        self.last_check = time.time()
        
        # Survival tracking for reward shaping
        self.steps_alive = 0
        self.episode_start_time = time.time()
        self.last_milestone = 0
        
        # Track if we're actually playing (not in menu)
        self.is_playing = True
        
        return self._get_state()
    
    def step(self, action):
        """
        Execute action for ACTION_DURATION seconds.
        
        This gives the action time to "play out" - if we choose "move toward food",
        we actually move toward food for 0.5s before making a new decision.
        
        This solves the credit assignment problem: the strategic action gets 
        associated with the reward, not a random flail right before eating.
        """
        start_time = time.time()
        total_reward = 0.0
        done = False
        
        # Store initial values for reward calculation
        initial_score = self.score
        initial_radius = self.self_radius
        
        # Execute action repeatedly for ACTION_DURATION
        # BUT STOP immediately if we detect death (don't record menu actions)
        while time.time() - start_time < self.ACTION_DURATION and not done:
            # Get fresh detections for semantic actions
            self._get_state()  # Updates last_food, last_players, etc.
            
            # Execute the semantic action with current detections
            self.actions[action]()
            
            # Check for death EVERY iteration (not just every second)
            # This prevents recording actions on the death/menu screen
            if time.time() - self.last_check > 0.3:  # Check every 0.3s
                self.last_check = time.time()
                
                # Check BOTH continue AND start screens
                has_continue = self.cont_det.hasContinue()
                has_start = self.start_det.hasStart()
                
                if has_continue or has_start:
                    done = True
                    self.is_playing = False
                    
                    # Click the appropriate button
                    if has_continue:
                        pyautogui.click(*self.CONTINUE_CLICK)
                        time.sleep(0.3)
                    if has_start:  # Check again in case continue led to start
                        if self.start_det.hasStart():
                            pyautogui.click(*self.START_CLICK)
                            time.sleep(0.3)
                    break  # STOP - don't execute any more actions
                
                # Update score only if actually playing
                new_score = self.score_det.getScore()
                if new_score > self.score:
                    self.score = new_score
            
            # Small sleep to not overwhelm the game
            time.sleep(self.ACTION_REFRESH_RATE)
        
        # Get final state after action completed
        state = self._get_state()
        
        # Track survival
        self.steps_alive += 1
        time_alive = time.time() - self.episode_start_time
        
        import math
        
        # === COMPREHENSIVE SAFETY-FIRST REWARD SYSTEM ===
        # Goal: Agent learns to SURVIVE first, then grow
        
        # --- 1. BASE REWARDS (small) ---
        score_delta = self.score - initial_score
        radius_delta = self.self_radius - initial_radius
        base_reward = 0.3 * score_delta + 0.1 * radius_delta
        
        # --- 2. SURVIVAL BONUS (THE MAIN REWARD) ---
        # Being alive is GOOD. The longer you survive, the better you're doing.
        # But scale carefully to not explode
        
        # Constant per-step survival bonus (adds up!)
        survival_bonus = 0.5  # +0.5 for each step alive (4 steps/sec = +2.0/sec)
        
        # Log bonus for longer survival (diminishing returns)
        log_bonus = 0.1 * math.log(1 + self.steps_alive)
        
        # --- 3. ACTION-SPECIFIC REWARDS (shape safe behavior) ---
        # Heavily reward defensive actions, moderately reward food, penalize risky hunting
        action_rewards = {
            0: 0.3,    # EAT_FOOD: Good! Growing safely
            1: -0.2,   # HUNT_PREY: Risky! Small penalty to discourage unless sure
            2: 0.8,    # FLEE_THREAT: Excellent! Staying safe
            3: -0.3,   # TO_VIRUS: Risky! Discourage
            4: 0.4,    # FROM_VIRUS: Good! Avoiding danger
        }
        action_reward = action_rewards.get(action, 0.0)
        
        # --- 4. PROXIMITY DANGER PENALTY ---
        # If there's a threat CLOSE by, that's BAD - even if we didn't die yet
        danger_penalty = 0.0
        if self.last_players:
            for (dx, dy, r) in self.last_players:
                dist = (dx**2 + dy**2)**0.5
                # Is this player a threat (equal size or larger)?
                if self.self_radius and r >= self.self_radius * self.LARGER_THRESHOLD:
                    # The closer the threat, the bigger the penalty
                    if dist < self.DANGER_DISTANCE:
                        # Penalty scales inversely with distance
                        # At dist=50: penalty = 1.5, dist=100: penalty = 1.0, dist=200: penalty = 0.5
                        danger_penalty += (self.DANGER_DISTANCE / max(dist, 1)) * 0.5
        
        # Clamp danger penalty so it doesn't dominate
        danger_penalty = min(danger_penalty, 3.0)
        
        # --- 5. FOOD PROXIMITY BONUS ---
        # Being near food is good - you can eat it!
        food_bonus = 0.0
        if self.last_food:
            nearest_food_dist = (self.last_food[0][0]**2 + self.last_food[0][1]**2)**0.5
            # Bonus if food is close (within 150 pixels)
            if nearest_food_dist < 150:
                food_bonus = 0.2 * (150 - nearest_food_dist) / 150
        
        # --- 6. MILESTONE BONUSES ---
        # Clear signal: surviving this long is GREAT
        milestone_interval = 20  # Every 20 steps (~5 seconds at 4 actions/sec)
        current_milestone = self.steps_alive // milestone_interval
        milestone_bonus = 0.0
        if current_milestone > self.last_milestone:
            milestone_bonus = 3.0  # Nice bonus for surviving another milestone
            self.last_milestone = current_milestone
        
        # --- TOTAL REWARD ---
        total_reward = (
            base_reward +           # Score/size changes (small)
            survival_bonus +        # Per-step survival (+0.5)
            log_bonus +             # Log survival bonus
            action_reward +         # Action shaping
            food_bonus +            # Near food bonus
            milestone_bonus -       # Milestone bonus
            danger_penalty          # Proximity to threats (PENALTY)
        )
        
        # --- DEATH PENALTY ---
        # Store the last N actions for death analysis
        if not hasattr(self, 'action_history'):
            self.action_history = []
        self.action_history.append((action, self.action_names[action], total_reward))
        if len(self.action_history) > 20:
            self.action_history.pop(0)
        
        if done:
            # Death is VERY BAD
            total_reward = -100.0
            
            # Print death analysis - what actions led here?
            print(f"\n{'='*60}")
            print(f"☠️  DEATH ANALYSIS - {self.steps_alive} steps survived")
            print(f"{'='*60}")
            print(f"Last {len(self.action_history)} actions before death:")
            for i, (a, name, r) in enumerate(self.action_history[-10:]):
                marker = ">>>" if i == len(self.action_history[-10:])-1 else "   "
                print(f"  {marker} Step -{len(self.action_history[-10:])-i}: {name} (r={r:.2f})")
            
            # Count action types
            action_counts = {}
            for (a, name, r) in self.action_history:
                action_counts[name] = action_counts.get(name, 0) + 1
            print(f"\nAction distribution before death:")
            for name, count in sorted(action_counts.items(), key=lambda x: -x[1]):
                print(f"  {name}: {count} ({100*count/len(self.action_history):.0f}%)")
            print(f"{'='*60}\n")
        
        # Update tracking for next step
        self.last_score = self.score
        self.last_radius = self.self_radius
        
        return state, total_reward, done, {
            'score': self.score, 
            'radius': self.self_radius,
            'steps_alive': self.steps_alive,
            'time_alive': time_alive,
            'danger_penalty': danger_penalty,
            'action': self.action_names[action]
        }


# ============== DQN AGENT ==============
class DQNAgent:
    # === CREDIT ASSIGNMENT PARAMETERS ===
    N_STEPS = 5  # How many steps back to propagate credit
    
    def __init__(self, state_dim, action_dim):
        self.action_dim = action_dim
        self.gamma = 0.99
        
        # === AGGRESSIVE EPSILON SCHEDULE ===
        # Start exploring, but quickly shift to exploitation
        self.epsilon = 1.0
        self.eps_min = 0.02  # Lower minimum - trust learned policy more
        self.eps_decay = 0.99  # MUCH faster decay: 0.99^100 = 0.366, 0.99^200 = 0.134
        self.episode_count = 0
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q = QNet(state_dim, action_dim).to(self.device)
        self.target = QNet(state_dim, action_dim).to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        
        # Higher learning rate for faster adaptation
        self.opt = torch.optim.Adam(self.q.parameters(), lr=0.001)
        
        # N-step replay buffer for proper credit assignment
        self.buffer = NStepReplayBuffer(100000, n_steps=self.N_STEPS, gamma=self.gamma)
        self.step_count = 0
    
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            return self.q(torch.FloatTensor(state).unsqueeze(0).to(self.device)).argmax(1).item()
    
    def learn(self, batch=64):
        if len(self.buffer) < batch:
            return 0.0
        
        self.step_count += 1
        # Learn EVERY step, not every 4th - faster learning
        
        s, a, r, ns, d = self.buffer.sample(batch)
        s = torch.FloatTensor(s).to(self.device)
        a = torch.LongTensor(a).to(self.device)
        r = torch.FloatTensor(r).to(self.device)
        ns = torch.FloatTensor(ns).to(self.device)
        d = torch.FloatTensor(d).to(self.device)
        
        # Q(s, a) from online network
        q = self.q(s).gather(1, a.unsqueeze(1)).squeeze()
        
        with torch.no_grad():
            # === DOUBLE DQN ===
            # Use ONLINE network to SELECT action, TARGET network to EVALUATE
            # This reduces overestimation bias
            best_actions = self.q(ns).argmax(1)  # Online selects
            nq = self.target(ns).gather(1, best_actions.unsqueeze(1)).squeeze()  # Target evaluates
            
            gamma_n = self.gamma ** self.N_STEPS
            tgt = r + gamma_n * nq * (1 - d)
        
        loss = F.smooth_l1_loss(q, tgt)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        
        # Update target network more frequently
        if self.step_count % 200 == 0:
            self.target.load_state_dict(self.q.state_dict())
        
        return loss.item()
    
    def decay_episode(self):
        """Call once per EPISODE for epsilon decay (not per step)"""
        self.episode_count += 1
        # Decay per episode, not per step - faster transition to exploitation
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)
        
        # Print epsilon schedule
        if self.episode_count % 5 == 0:
            print(f"  [Epsilon Schedule] Episode {self.episode_count}: ε = {self.epsilon:.3f}")
    
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'q': self.q.state_dict(),
            'target': self.target.state_dict(),
            'opt': self.opt.state_dict(),
            'eps': self.epsilon,
            'step': self.step_count
        }, path)
        print(f"Saved: {path}")
    
    def load(self, path):
        """
        Load checkpoint with robust handling of different formats.
        Returns True if successful, False otherwise.
        """
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            
            # Try different checkpoint formats
            q_state = None
            target_state = None
            
            # Format 1: Our current format {'q': ..., 'target': ...}
            if isinstance(ckpt, dict):
                if 'q' in ckpt:
                    q_state = ckpt['q']
                    target_state = ckpt.get('target', ckpt['q'])
                # Format 2: Old format {'q_net': ..., 'target_net': ...}
                elif 'q_net' in ckpt:
                    q_state = ckpt['q_net']
                    target_state = ckpt.get('target_net', ckpt['q_net'])
                # Format 3: Direct state dict (model.state_dict())
                elif any(k.startswith('feature') or k.startswith('net') or k.startswith('value') for k in ckpt.keys()):
                    q_state = ckpt
                    target_state = ckpt
            
            if q_state is None:
                print(f"  Unknown checkpoint format: {list(ckpt.keys())[:5]}...")
                return False
            
            # Try to load the state dict
            self.q.load_state_dict(q_state)
            self.target.load_state_dict(target_state)
            
            # Load training state
            if isinstance(ckpt, dict):
                self.epsilon = ckpt.get('eps', ckpt.get('epsilon', 0.1))
                self.step_count = ckpt.get('step', ckpt.get('learn_step', 0))
                
                # Also restore optimizer if available
                if 'opt' in ckpt:
                    try:
                        self.opt.load_state_dict(ckpt['opt'])
                    except:
                        pass  # Optimizer state may not match, that's ok
            
            return True
            
        except RuntimeError as e:
            # This usually means architecture mismatch (different layer sizes)
            print(f"  Architecture mismatch: {str(e)[:100]}...")
            return False
        except Exception as e:
            print(f"  Error loading: {type(e).__name__}: {str(e)[:100]}...")
            return False


# ============== TRAINING ==============
def find_all_checkpoints():
    """
    Find ALL checkpoint files, sorted by episode number (newest first).
    Handles various naming patterns:
      - dqn_ep10.pth
      - 128-64-32-8dqn_ep140.pth
      - dqn_interrupted.pth
      - dqn_final.pth
    """
    import glob, re
    d = "checkpoints"
    if not os.path.exists(d):
        return []
    
    all_ckpts = []
    
    # Find all .pth files
    files = glob.glob(os.path.join(d, "*.pth"))
    
    for f in files:
        # Try to extract episode number from filename
        # Matches patterns like: dqn_ep10, 128-64-32-8dqn_ep140, etc.
        m = re.search(r'ep(\d+)', f)
        if m:
            ep_num = int(m.group(1))
            all_ckpts.append((ep_num, f))
        elif "interrupted" in f:
            all_ckpts.append((-1, f))  # Interrupted has priority -1
        elif "final" in f:
            all_ckpts.append((-2, f))  # Final has priority -2
        else:
            all_ckpts.append((-3, f))  # Unknown files have lowest priority
    
    # Sort by episode number (highest first)
    all_ckpts.sort(key=lambda x: x[0], reverse=True)
    
    return all_ckpts


def find_ckpt():
    """Find the best checkpoint to load."""
    ckpts = find_all_checkpoints()
    if not ckpts:
        return None, 0
    
    # Return the highest episode checkpoint
    ep, path = ckpts[0]
    return path, max(0, ep)  # Return 0 for interrupted/final/unknown


def train(episodes=10000, max_steps=600):  # ~5 min per episode with 0.5s/step
    global stop_training
    print("="*80)
    print("🛡️  SAFETY-FIRST DQN for Agar.io - SURVIVAL MODE  🛡️")
    print("="*80)
    print(f"⚡ SPEED: {1/AgarioEnv.ACTION_DURATION:.0f} actions/second ({AgarioEnv.ACTION_DURATION}s each)")
    print(f"🎯 State: 26 dims | Actions: 5 (EAT, HUNT, FLEE, TO_VIRUS, FROM_VIRUS)")
    print()
    print("🛡️  SAFETY THRESHOLDS (Conservative):")
    print(f"   - Hunt prey only if {AgarioEnv.SMALLER_THRESHOLD*100:.0f}% smaller (VERY safe)")
    print(f"   - Flee from ANY player equal or larger (LARGER_THRESHOLD={AgarioEnv.LARGER_THRESHOLD})")
    print(f"   - Danger zone: {AgarioEnv.DANGER_DISTANCE}px (penalty for being close to threats)")
    print()
    print("💰 REWARD STRUCTURE (Survival-focused):")
    print("   + Per-step survival: +0.5 (being alive = good!)")
    print("   + Food seeking: +0.3  | Near food: up to +0.2")
    print("   + FLEE action: +0.8   | FROM_VIRUS: +0.4")
    print("   - HUNT action: -0.2   | TO_VIRUS: -0.3 (risky!)")
    print("   - Proximity penalty: up to -3.0 near threats")
    print("   + Milestone: +3.0 every 20 steps (~5 sec)")
    print("   💀 Death: -100")
    print()
    print("📊 Death Analysis: Shows last 10 actions when dying")
    print("Press Q to save and quit")
    print("="*80)
    
    env = AgarioEnv()
    agent = DQNAgent(env.state_dim, env.action_dim)
    
    # === CHECKPOINT LOADING - TRY ALL AVAILABLE ===
    all_ckpts = find_all_checkpoints()
    loaded = False
    start = 0
    
    if all_ckpts:
        print(f"\nFound {len(all_ckpts)} checkpoint(s):")
        for ep, path in all_ckpts[:5]:  # Show top 5
            print(f"  - {path} (ep={ep})")
        
        # Try loading checkpoints in order until one works
        for ep, path in all_ckpts:
            print(f"\nTrying to load: {path}...")
            if agent.load(path):
                print(f"✓ Successfully loaded! ε={agent.epsilon:.3f} step={agent.step_count}")
                start = max(0, ep)
                loaded = True
                break
            else:
                print(f"✗ Failed to load (architecture mismatch?), trying next...")
    
    if not loaded:
        print("\n⚠ No compatible checkpoint found. Starting fresh training.")
        print("  (This is expected if architecture changed)")
        start = 0
    
    print(f"\nStarting from episode {start+1}")
    time.sleep(2)
    
    best_score = 0
    
    for ep in range(start, episodes):
        if stop_training:
            agent.save("checkpoints/dqn_interrupted.pth")
            break
        
        state = env.reset()
        ep_reward = 0
        losses = []
        start_time = time.time()
        
        # Track action history for death attribution
        action_history = []  # (step, action_name, reward)
        
        for step in range(max_steps):
            if stop_training:
                break
            
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            
            # Only push to buffer if we were actually playing (not in menu)
            if env.is_playing or done:  # Include the death transition
                agent.buffer.push(state, action, reward, next_state, float(done))
            
            loss = agent.learn()
            if loss > 0:
                losses.append(loss)
            
            # NOTE: Epsilon decay is now per-EPISODE, not per-step
            # This is handled after the episode ends
            ep_reward += reward
            
            # Track action history
            act_name = env.action_names[action]
            action_history.append((step, act_name, reward))
            
            # === DETAILED PRINTOUT EVERY 10 STEPS (~5 seconds) ===
            if step % 10 == 0:
                elapsed = time.time() - start_time
                
                # State summary: nearest entities
                food_dir = "none"
                player_dir = "none"
                virus_dir = "none"
                
                if env.last_food:
                    fx, fy = env.last_food[0]
                    food_dir = f"({fx:+4.0f},{fy:+4.0f})"
                if env.last_players:
                    px, py, pr = env.last_players[0]
                    player_dir = f"({px:+4.0f},{py:+4.0f},r{pr:3.0f})"
                if env.last_virus:
                    vx, vy = env.last_virus
                    virus_dir = f"({vx:+4.0f},{vy:+4.0f})"
                
                # Safety info
                danger_pen = info.get('danger_penalty', 0)
                time_alive = info.get('time_alive', 0)
                
                # Show danger level
                danger_str = "🛡️ SAFE" if danger_pen < 0.5 else ("⚠️ NEAR" if danger_pen < 1.5 else "🚨 DANGER!")
                
                print(f"\n  Step {step:3d} | {act_name:11s} | Reward: {reward:+6.2f} | {danger_str} (pen={danger_pen:.1f}) | "
                      f"Alive: {time_alive:.0f}s | SelfR={info['radius']:3d} | Food={food_dir} Player={player_dir}")
            
            # === COMPACT STATUS EVERY 20 STEPS ===
            elif step % 20 == 0 and step > 0:
                avg_l = np.mean(losses[-20:]) if losses else 0
                elapsed = time.time() - start_time
                print(f"\rEp{ep+1} s{step:3d} {elapsed:4.0f}s | ε={agent.epsilon:.3f} L={avg_l:.4f} | {act_name:11s} R={info['radius']:3d} F={env.food_count:2d} P={env.player_count}", end="", flush=True)
            
            state = next_state
            
            # === DEATH ATTRIBUTION ===
            if done:
                print(f"\n\n{'='*60}")
                print(f"💀 DEATH at step {step}! Analyzing what caused it...")
                print(f"{'='*60}")
                
                # Show last N_STEPS actions (what the buffer will attribute credit to)
                n_steps = agent.N_STEPS
                recent = action_history[-n_steps:] if len(action_history) >= n_steps else action_history
                
                print(f"Last {len(recent)} actions before death (N-step credit goes to first one):")
                for i, (s, a, r) in enumerate(recent):
                    marker = ">>> CREDITED FOR DEATH <<<" if i == 0 else ""
                    print(f"  Step {s:3d}: {a:11s} | Reward: {r:+7.2f} {marker}")
                
                print(f"\nDeath penalty (-100) will be attributed to: {recent[0][1] if recent else 'N/A'}")
                print(f"{'='*60}\n")
                break
        
        if stop_training:
            break
        
        # === EPSILON DECAY PER EPISODE ===
        # Faster decay: 0.99 per episode means ε drops quickly
        # Episode 10: ε ≈ 0.90, Episode 50: ε ≈ 0.60, Episode 100: ε ≈ 0.37
        agent.decay_episode()
        
        # Episode summary
        ep_time = time.time() - start_time
        avg_loss = np.mean(losses) if losses else 0
        
        if info['score'] > best_score:
            best_score = info['score']
        
        time_alive = info.get('time_alive', ep_time)
        print(f"Ep{ep+1:4d} SUMMARY | Score={info['score']:4d} Best={best_score:4d} | R={info['radius']:3d} | "
              f"Survived: {step}steps/{time_alive:.0f}s | ε={agent.epsilon:.3f} L={avg_loss:.4f} | Buf={len(agent.buffer)}\n")
        
        # Save checkpoints
        if (ep + 1) % 10 == 0:
            agent.save(f"checkpoints/dqn_ep{ep+1}.pth")
        
        # Extra save every 100 episodes
        if (ep + 1) % 100 == 0:
            agent.save(f"checkpoints/dqn_ep{ep+1}_backup.pth")
    
    if not stop_training:
        agent.save("checkpoints/dqn_final.pth")
        print("\nTraining complete!")


if __name__ == "__main__":
    train()
