"""
Load and Demo the Semantic Action DQN for Agar.io
==================================================
5 Semantic Actions: EAT_FOOD, HUNT_PREY, FLEE_THREAT, TO_VIRUS, FROM_VIRUS
Press Q to quit
"""

import torch
import torch.nn as nn
import numpy as np
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
stop_demo = False
def on_press(key):
    global stop_demo
    try:
        if key.char == 'q':
            stop_demo = True
    except:
        pass
keyboard.Listener(on_press=on_press).start()


# ============== UNIFIED DETECTOR ==============
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


# ============== DUELING DQN NETWORK ==============
class QNet(nn.Module):
    """
    Dueling DQN Architecture (Wang et al., 2016)
    https://arxiv.org/abs/1511.06581
    Must match dqnAgario.py exactly for checkpoint loading.
    """
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


# ============== DEMO ENVIRONMENT ==============
class DemoEnv:
    CONTINUE_CLICK = (759, -566)
    START_CLICK = (754, -655)
    ACTION_NAMES = ["EAT_FOOD", "HUNT_PREY", "FLEE_THREAT", "TO_VIRUS", "FROM_VIRUS"]
    
    # Credit assignment parameters (must match dqnAgario.py)
    ACTION_DURATION = 0.5  # Each action executes for this long
    ACTION_REFRESH_RATE = 0.1
    
    # Thresholds for player size comparison (must match dqnAgario.py)
    SMALLER_THRESHOLD = 0.9
    LARGER_THRESHOLD = 1.1
    
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
        
        print(f"Monitor: {self.mon_w}x{self.mon_h} at ({self.mon_left}, {self.mon_top})")
        print(f"Center: ({self.center_x}, {self.center_y})")
        
        # Raw detection storage for semantic actions
        self.last_self_radius = None
        self.last_players = []
        self.last_food = []
        self.last_virus = None
        
        # Semantic actions (must match dqnAgario.py exactly)
        self.actions = [
            self._act_toward_nearest_food,
            self._act_toward_smaller_player,
            self._act_away_from_larger_player,
            self._act_toward_virus,
            self._act_away_from_virus,
        ]
        
        self.state_dim = 26
        self.action_dim = 5  # 5 semantic actions
        self.self_radius = 60
        self.score = 0
        self.food_count = 0
        self.player_count = 0
    
    # ============== MOUSE MOVEMENT HELPERS ==============
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
    
    # ============== SEMANTIC ACTIONS ==============
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
            dx, dy, _ = self.last_players[0]
        else:
            dx, dy = target
        self._move_toward(dx, dy)
    
    def _act_away_from_larger_player(self):
        if not self.last_players:
            return self._act_toward_nearest_food()
        target = None
        for (dx, dy, r) in self.last_players:
            if self.self_radius and r > self.self_radius * self.LARGER_THRESHOLD:
                target = (dx, dy)
                break
        if target is None:
            dx, dy, _ = self.last_players[0]
        else:
            dx, dy = target
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
        
        # Store raw detections for semantic action functions
        self.last_self_radius = self_r
        self.last_players = players
        self.last_food = food
        self.last_virus = virus
        
        self.food_count = len(food)
        self.player_count = len(players)
        return self._build_state(self_r, players, food, virus)
    
    def step(self, action):
        """Execute action for ACTION_DURATION seconds (matches training)."""
        start_time = time.time()
        
        while time.time() - start_time < self.ACTION_DURATION:
            self.get_state()  # Update detections
            self.actions[action]()  # Execute with fresh detections
            time.sleep(self.ACTION_REFRESH_RATE)
        
        return self.get_state()
    
    def check_game_state(self):
        """Check if we need to restart - uses same click coords as dqnAgario.py"""
        if self.cont_det.hasContinue():
            print("\n[Continue button detected - clicking...]")
            pyautogui.click(self.CONTINUE_CLICK[0], self.CONTINUE_CLICK[1])
            time.sleep(0.5)
            return "died"
        if self.start_det.hasStart():
            print("\n[Start button detected - clicking...]")
            pyautogui.click(self.START_CLICK[0], self.START_CLICK[1])
            time.sleep(0.5)
            return "started"
        return "playing"
    
    def ensure_in_game(self):
        """Make sure we're in an active game"""
        for _ in range(3):
            if self.cont_det.hasContinue():
                pyautogui.click(self.CONTINUE_CLICK[0], self.CONTINUE_CLICK[1])
                time.sleep(0.5)
            if self.start_det.hasStart():
                pyautogui.click(self.START_CLICK[0], self.START_CLICK[1])
                time.sleep(0.5)
            time.sleep(0.3)


# ============== DEMO AGENT ==============
class DemoAgent:
    def __init__(self, state_dim=26, action_dim=5):  # 5 semantic actions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q = QNet(state_dim, action_dim).to(self.device)
        self.action_dim = action_dim
    
    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        if 'q' in ckpt:
            self.q.load_state_dict(ckpt['q'])
        elif 'q_net' in ckpt:
            self.q.load_state_dict(ckpt['q_net'])
        else:
            self.q.load_state_dict(ckpt)
        print(f"Loaded checkpoint: {path}")
    
    def act(self, state):
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q(state_t)
            action = q_values.argmax(1).item()
            return action, q_values.cpu().numpy()[0]


# ============== MAIN DEMO ==============
def find_best_checkpoint():
    """Find the best checkpoint from training"""
    import glob
    
    checkpoints = [
        ("checkpoints/dqn_ep90.pth", "After best score 82,202"),
        ("checkpoints/dqn_ep100.pth", "100 episodes trained"),
        ("checkpoints/dqn_ep100_backup.pth", "100 episodes backup"),
        ("checkpoints/dqn_ep80.pth", "Before best score"),
    ]
    
    for path, desc in checkpoints:
        if os.path.exists(path):
            print(f"Found: {path} ({desc})")
            return path
    
    # Fallback: find latest
    files = glob.glob("checkpoints/dqn_ep*.pth")
    if files:
        latest = max(files, key=lambda f: int(f.split('ep')[1].split('.')[0].split('_')[0]))
        return latest
    
    return None


def demo():
    global stop_demo
    
    print("="*60)
    print("Semantic Action DQN for Agar.io - DEMO MODE")
    print("="*60)
    print("Actions: EAT_FOOD, AVOID_FOOD, HUNT_PREY, FLEE_THREAT, TO_VIRUS, FROM_VIRUS")
    print("Press Q to quit")
    print("="*60)
    
    # Find and load best checkpoint
    ckpt_path = find_best_checkpoint()
    if not ckpt_path:
        print("ERROR: No checkpoint found!")
        return
    
    env = DemoEnv()
    agent = DemoAgent()
    agent.load(ckpt_path)
    
    print("\nStarting demo in 3 seconds...")
    print("Make sure Agar.io is visible on screen!")
    time.sleep(3)
    
    # Make sure we're in a game
    print("Checking game state...")
    env.ensure_in_game()
    
    step = 0
    score_check_time = time.time()
    game_check_time = time.time()
    score = 0
    best_score = 0
    
    print("\n--- DEMO RUNNING ---\n")
    
    while not stop_demo:
        step += 1
        
        # Get state and act
        state = env.get_state()
        action, q_values = agent.act(state)
        env.step(action)
        
        # Check game state periodically (every 3 seconds)
        if time.time() - game_check_time > 3.0:
            game_check_time = time.time()
            game_state = env.check_game_state()
            if game_state == "died":
                if score > best_score:
                    best_score = score
                print(f"💀 Died! Score: {score} | Best: {best_score} | Restarting...")
                score = 0
                env.self_radius = 60
        
        # Check score periodically (every 5 seconds)
        if time.time() - score_check_time > 5:
            score_check_time = time.time()
            new_score = env.score_det.getScore()
            if new_score > score:
                score = new_score
                env.score = score
        
        # Status printout every 20 steps
        if step % 20 == 0:
            action_name = env.ACTION_NAMES[action]
            print(f"\rStep {step:5d} | {action_name:11s} | R={env.self_radius:3d} | F={env.food_count:2d} P={env.player_count} | Score={score} Best={best_score}", end="", flush=True)
    
    print(f"\n\nDemo stopped! Best score: {best_score}")


if __name__ == "__main__":
    demo()

