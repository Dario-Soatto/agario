"""
Simple RL Environment for Agar.io
Uses capture_game.py for state extraction and takeOnScreenAction.py for actions
"""

import numpy as np
import time
import pyautogui
import cv2

# Import from existing modules
from capture_game import (
    capture_game_screen,
    create_blob_mask,
    find_blobs,
    classify_blobs,
    detect_game_end,
    extract_score,
    get_game_state,
    game_state_to_vector
)
from takeOnScreenAction import ScreenAction


class AgarioEnv:
    """
    Gym-like environment for Agar.io
    
    State: 24-dimensional vector from game_state_to_vector()
    Actions: 8 discrete directions (0-7)
    Reward: Delta score between steps
    """
    
    def __init__(self, step_delay=0.1):
        """
        Args:
            step_delay: Seconds to wait after taking action (for game to update)
        """
        # Screen setup
        self.screen_width = pyautogui.size().width
        self.screen_height = pyautogui.size().height
        
        # Game capture region (same as capture_game.py)
        self.monitor = {"top": 125, "left": 0, "width": 1470, "height": 671}
        
        # Virus color for classification (same as capture_game.py)
        self.virus_color_bgr = (90, 251, 125)
        
        # Game area height for splitting capture
        self.game_area_height = 762 - 125  # 637 pixels
        
        # Action controller
        self.action_controller = ScreenAction(self.screen_width, self.screen_height)
        
        # Action mapping
        self.actions = [
            self.action_controller.move_up,
            self.action_controller.move_down,
            self.action_controller.move_left,
            self.action_controller.move_right,
            self.action_controller.move_diagonal_up_right,
            self.action_controller.move_diagonal_down_right,
            self.action_controller.move_diagonal_up_left,
            self.action_controller.move_diagonal_down_left,
        ]
        self.num_actions = len(self.actions)
        
        # State tracking
        self.last_score = 0
        self.step_delay = step_delay
        
        # State/action space info
        self.state_dim = 24
        self.action_dim = 8

    def _detect_start_button(self, img_full):
        """
        Detect if the start button is showing using template matching
        Region: screen coords (633, 362) to (837, 391)
        Returns: True if start button is visible
        """
        # Convert screen coordinates to captured image coordinates
        # Capture starts at y=125, so subtract 125 from y coords
        region_y_start = 362 - 125  # 237
        region_y_end = 391 - 125    # 266
        region_x_start = 633
        region_x_end = 837
        
        # Crop the region
        region = img_full[region_y_start:region_y_end, region_x_start:region_x_end]
        
        # Load the template
        template = cv2.imread('templates/start.png')
        if template is None:
            print("Warning: Could not load templates/start.png")
            return False
        
        # Convert both to grayscale
        region_gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        
        # Resize template if needed
        if template_gray.shape != region_gray.shape:
            template_gray = cv2.resize(template_gray, (region_gray.shape[1], region_gray.shape[0]))
        
        # Template matching
        result = cv2.matchTemplate(region_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        return max_val >= 0.3  # 40% similarity threshold
    
    def _get_state(self):
        """
        Capture screen and extract game state vector
        Returns: (state_vector, score, game_ended)
        """
        # Capture screenshot
        img_full = capture_game_screen(self.monitor)
        
        # Split into game area and score area
        img_game = img_full[:self.game_area_height, :]
        
        # Get dimensions
        img_height, img_width = img_game.shape[:2]
        
        # Process image
        mask = create_blob_mask(img_game)
        blobs, contours = find_blobs(mask)
        classified = classify_blobs(blobs, img_game, img_width, img_height, self.virus_color_bgr)
        
        # Check game state
        game_ended = detect_game_end(img_full)
        score = extract_score(img_full, debug=False)
        
        # Build game state dict
        game_state = get_game_state(classified, game_ended)
        game_state['score'] = score if score is not None else 0
        
        # Convert to vector
        state_vector = game_state_to_vector(game_state)
        
        return state_vector, game_state['score'], game_ended
    
    def _click_start_button(self):
        """Click the start button to begin a new game"""
        pyautogui.click(734, 378)
        time.sleep(0.5)  # Wait for click to register
    
    def _restart_game(self):
        """
        Restart the game if game-over screen is showing
        Flow:
        1. Sleep 5 seconds
        2. Click at (735, 455) - dismiss intermediate screen
        3. Sleep 5 seconds
        4. Template match for start button, click (734, 378) if found
        5. Wait for game to actually begin
        """
        # Step 1: Wait for any animations/ads
        print("Waiting 5 seconds...")
        time.sleep(5)
        
        # Step 2: Click to dismiss intermediate screen
        print("Clicking intermediate screen at (735, 455)...")
        pyautogui.click(735, 455)
        
        # Step 3: Wait again
        print("Waiting 5 seconds...")
        time.sleep(5)
        
        # Step 4: Capture screen and check for start button
        img_full = capture_game_screen(self.monitor)
        
        if self._detect_start_button(img_full):
            print("Start button detected, clicking at (734, 378)...")
            self._click_start_button()
            
            # Wait for game to actually start (up to 5 seconds)
            for _ in range(25):  # 25 * 0.2 = 5 seconds max
                time.sleep(0.2)
                img_full = capture_game_screen(self.monitor)
                game_ended = detect_game_end(img_full)
                if not game_ended and not self._detect_start_button(img_full):
                    print("Game started!")
                    return True
            
            print("Warning: Game did not start within timeout")
            return False
        else:
            print("Start button not detected, assuming game is ready")
        
        return True
    
    def reset(self):
        """
        Reset environment for new episode.
        Automatically restarts game if needed.
        
        Returns: initial state vector
        """
        # Restart game if needed
        self._restart_game()
        
        # Wait a moment for game to stabilize
        time.sleep(0.5)
        
        # Get initial state
        state, score, _ = self._get_state()
        self.last_score = score if score else 0
        
        return state
    
    def step(self, action):
        """
        Take an action and observe the result.
        
        Args:
            action: int from 0-7 representing direction
            
        Returns:
            state: new state vector (24,)
            reward: delta score
            done: True if game ended
            info: dict with extra info
        """
        # Execute action
        self.actions[action]()
        
        # Wait for game to update
        time.sleep(self.step_delay)
        
        # Get new state
        state, score, game_ended = self._get_state()
        
        # Calculate reward (delta score)
        current_score = score if score else 0
        reward = current_score - self.last_score
        self.last_score = current_score
        
        # Add death penalty
        done = game_ended
        if done:
            reward -= 100  # Penalty for dying
        
        # Info dict
        info = {
            'score': current_score,
            'game_ended': game_ended
        }
        
        return state, reward, done, info
    
    def sample_action(self):
        """Return a random action"""
        return np.random.randint(0, self.num_actions)


# Quick test
if __name__ == "__main__":
    print("Testing AgarioEnv...")
    print("Make sure Agar.io is open and a game is running!")
    print("Starting in 3 seconds...")
    time.sleep(3)
    
    env = AgarioEnv(step_delay=0.15)
    
    num_episodes = 5  # Run 5 episodes
    steps_per_episode = 100
    
    for episode in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"EPISODE {episode + 1}/{num_episodes}")
        print(f"{'='*60}")
        
        # Reset environment (auto-restarts game if needed)
        state = env.reset()
        print(f"Initial state shape: {state.shape}")
        
        episode_reward = 0
        step = 0
        
        while step < steps_per_episode:
            action = env.sample_action()
            state, reward, done, info = env.step(action)
            episode_reward += reward
            step += 1
            
            print(f"Step {step}: action={action}, reward={reward:.1f}, score={info['score']}, done={done}")
            
            if done:
                print(f"\nGame ended at step {step}!")
                print(f"Episode reward: {episode_reward}")
                break
        
        print(f"\nEpisode {episode + 1} finished. Total reward: {episode_reward}")
    
    print(f"\n{'='*60}")
    print("All episodes complete!")
    print(f"{'='*60}")