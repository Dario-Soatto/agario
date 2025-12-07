"""
================================================================================
AGAR.IO REINFORCEMENT LEARNING CONTROL LOOP
================================================================================

This file implements a complete RL policy execution loop for controlling an 
Agar.io bot. It connects the game state capture pipeline with action execution.

ARCHITECTURE OVERVIEW:
----------------------
    ┌─────────────────┐
    │   Game Screen   │
    └────────┬────────┘
             │ capture_game_screen()
             ▼
    ┌─────────────────┐
    │  Image Processing│ (mask, blob detection, classification)
    └────────┬────────┘
             │ get_full_game_state()
             ▼
    ┌─────────────────┐
    │   Game State    │ (dict with self_position, enemies, food, etc.)
    └────────┬────────┘
             │ game_state_to_vector()
             ▼
    ┌─────────────────┐
    │  State Vector   │ (24-element numpy array for RL)
    └────────┬────────┘
             │ policy.choose_action()
             ▼
    ┌─────────────────┐
    │  Action Index   │ (0-7, maps to 8 movement directions)
    └────────┬────────┘
             │ execute_action()
             ▼
    ┌─────────────────┐
    │ Mouse Movement  │ (pyautogui moves cursor in game)
    └─────────────────┘

MDP FORMULATION:
----------------
- State Space: 24-dimensional continuous vector containing:
    [self_x, self_y, self_radius, score,
     virus1_x, virus1_y, virus1_radius, virus2_x, ..., virus3_radius,
     player1_x, player1_y, player1_radius, player2_x, ..., player3_radius,
     food_count, game_ended]
     
- Action Space: 8 discrete actions (movement directions)
    0: Up
    1: Down  
    2: Left
    3: Right
    4: Diagonal Up-Right
    5: Diagonal Down-Right
    6: Diagonal Up-Left
    7: Diagonal Down-Left

- Reward: Defined by your RL algorithm (e.g., score delta, survival time)

HOW TO INTEGRATE A LEARNED POLICY:
----------------------------------
1. Replace SimplePolicy with your trained model
2. Implement choose_action() to:
   - Accept state vector (numpy array of shape (24,))
   - Return action index (int 0-7)
   
Example with a neural network:
    
    class NeuralNetworkPolicy:
        def __init__(self, model_path):
            self.model = load_model(model_path)  # PyTorch, TF, etc.
        
        def choose_action(self, state_vector):
            state_tensor = torch.tensor(state_vector).unsqueeze(0)
            with torch.no_grad():
                q_values = self.model(state_tensor)
            return q_values.argmax().item()

Example with a Q-table:
    
    class QTablePolicy:
        def __init__(self, q_table):
            self.q_table = q_table  # dict: state_hash -> action_values
        
        def choose_action(self, state_vector):
            state_hash = self._discretize(state_vector)
            if state_hash in self.q_table:
                return np.argmax(self.q_table[state_hash])
            return np.random.randint(0, 8)  # Random if unseen state

================================================================================
"""

import numpy as np
import time
import pyautogui

# ==============================================================================
# IMPORTS FROM PROJECT MODULES
# ==============================================================================

# Import game state capture functions from capture_game.py
from capture_game import (
    capture_game_screen,
    create_blob_mask,
    find_blobs,
    classify_blobs,
    detect_game_end,
    extract_score,
    get_game_state,
    game_state_to_vector,
)

# Import action executor from takeOnScreenAction.py
from takeOnScreenAction import ScreenAction

# Import coordinate helpers if needed for advanced policies
from find_coordinates import find_window_coordinates


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Game window capture region (adjust to your screen setup)
# Use find_coordinates.py to determine these values for your screen
MONITOR_REGION = {"top": 125, "left": 0, "width": 1470, "height": 671}

# Virus color in BGR format (adjust to match your game's virus color)
VIRUS_COLOR_BGR = (90, 251, 125)

# Game area height in pixels (from capture region top to game/score boundary)
GAME_AREA_HEIGHT = 762 - 125  # 637 pixels

# Control loop timing
ACTION_DELAY = 0.1  # Seconds between actions (adjust for game responsiveness)


# ==============================================================================
# ACTION MAPPING
# ==============================================================================

def create_action_mapping(screen_action: ScreenAction) -> dict:
    """
    Create a mapping from action indices to ScreenAction methods.
    
    Action Space (8 discrete actions):
        0: Up               - Move toward top of screen
        1: Down             - Move toward bottom of screen
        2: Left             - Move toward left of screen
        3: Right            - Move toward right of screen
        4: Diagonal Up-Right    - Move toward top-right corner
        5: Diagonal Down-Right  - Move toward bottom-right corner
        6: Diagonal Up-Left     - Move toward top-left corner
        7: Diagonal Down-Left   - Move toward bottom-left corner
    
    Parameters:
    -----------
    screen_action : ScreenAction
        An initialized ScreenAction object from takeOnScreenAction.py
        
    Returns:
    --------
    dict : Mapping from action index (int) to callable method
    """
    return {
        0: screen_action.move_up,
        1: screen_action.move_down,
        2: screen_action.move_left,
        3: screen_action.move_right,
        4: screen_action.move_diagonal_up_right,
        5: screen_action.move_diagonal_down_right,
        6: screen_action.move_diagonal_up_left,
        7: screen_action.move_diagonal_down_left,
    }


# Action names for logging/debugging
ACTION_NAMES = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT",
    4: "DIAGONAL_UP_RIGHT",
    5: "DIAGONAL_DOWN_RIGHT",
    6: "DIAGONAL_UP_LEFT",
    7: "DIAGONAL_DOWN_LEFT",
}

NUM_ACTIONS = 8


# ==============================================================================
# GAME STATE CAPTURE (WRAPPER FUNCTION)
# ==============================================================================

def get_full_game_state(debug: bool = False) -> tuple:
    """
    Capture and process the current game screen to extract the full game state.
    
    This wrapper function performs the complete capture pipeline:
    1. Capture screenshot of game window
    2. Create blob mask for image processing
    3. Find and classify all blobs (self, food, enemies, viruses)
    4. Detect if game has ended
    5. Extract current score
    6. Package into structured game state
    7. Convert to fixed-size vector for RL
    
    Returns:
    --------
    tuple: (game_state_dict, state_vector, raw_image)
        - game_state_dict: Dictionary with keys:
            'game_ended': bool
            'self_position': (x, y, radius) or None
            'viruses': list of (x, y, radius)
            'other_players': list of (x, y, radius)
            'food_count': int
            'score': int or None
        - state_vector: numpy array of shape (24,) for RL input
        - raw_image: The captured BGR image (useful for debugging)
    """
    # Step 1: Capture full screenshot (includes game area + score area)
    img_full = capture_game_screen(MONITOR_REGION)
    
    # Step 2: Split image into game area and score area
    img_game = img_full[:GAME_AREA_HEIGHT, :]  # Game area for blob detection
    img_height, img_width = img_game.shape[:2]
    
    # Step 3: Create mask and find blobs
    mask = create_blob_mask(img_game)
    blobs, contours = find_blobs(mask)
    
    # Step 4: Classify blobs into categories
    classified = classify_blobs(blobs, img_game, img_width, img_height, VIRUS_COLOR_BGR)
    
    # Step 5: Detect game end state
    game_ended = detect_game_end(img_full)
    
    # Step 6: Extract score
    score = extract_score(img_full, debug=debug)
    
    # Step 7: Build game state dictionary
    game_state = get_game_state(classified, game_ended)
    game_state['score'] = score
    
    # Step 8: Convert to fixed-size vector for RL
    state_vector = game_state_to_vector(game_state)
    
    return game_state, state_vector, img_full


# ==============================================================================
# POLICY CLASSES
# ==============================================================================

class SimplePolicy:
    """
    A simple rule-based placeholder policy for testing the control loop.
    
    This policy demonstrates the interface that any RL policy must implement:
    - __init__(): Initialize the policy (load models, set parameters)
    - choose_action(state_vector): Return an action index given a state
    
    Current behavior: Random action selection with slight bias toward
    avoiding edges (moves away from screen boundaries).
    
    REPLACE THIS with your trained RL policy!
    """
    
    def __init__(self, screen_width: int = 1470, screen_height: int = 671):
        """
        Initialize the simple policy.
        
        Parameters:
        -----------
        screen_width : int
            Width of the game capture region
        screen_height : int
            Height of the game capture region
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.center_x = screen_width / 2
        self.center_y = screen_height / 2
        
    def choose_action(self, state_vector: np.ndarray) -> int:
        """
        Choose an action based on the current state.
        
        This simple policy uses basic heuristics:
        1. If near screen edge, move toward center
        2. Otherwise, random action
        
        Parameters:
        -----------
        state_vector : np.ndarray
            24-element state vector from game_state_to_vector()
            Structure: [self_x, self_y, self_radius, score, 
                       virus1_x, virus1_y, virus1_radius, ...,
                       player1_x, player1_y, player1_radius, ...,
                       food_count, game_ended]
        
        Returns:
        --------
        int : Action index (0-7)
        """
        # Extract self position from state vector
        self_x, self_y, self_radius = state_vector[0], state_vector[1], state_vector[2]
        
        # If we don't have a valid position, move randomly
        if self_x == -1 or self_y == -1:
            return np.random.randint(0, NUM_ACTIONS)
        
        # Simple edge avoidance: if near edge, move toward center
        edge_threshold = 100  # pixels from edge
        
        # Check if near left edge
        if self_x < edge_threshold:
            return 3  # Move right
        # Check if near right edge
        if self_x > self.screen_width - edge_threshold:
            return 2  # Move left
        # Check if near top edge
        if self_y < edge_threshold:
            return 1  # Move down
        # Check if near bottom edge
        if self_y > self.screen_height - edge_threshold:
            return 0  # Move up
            
        # Default: random action
        return np.random.randint(0, NUM_ACTIONS)


class SmartPolicy:
    """
    A slightly smarter policy that considers enemies and food.
    
    Strategy:
    - Avoid larger enemies (move away from nearest threat)
    - Move toward center when no immediate threats
    - Random exploration when safe
    
    This is still a placeholder - replace with a trained policy!
    """
    
    def __init__(self, screen_width: int = 1470, screen_height: int = 671):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.center_x = screen_width / 2
        self.center_y = screen_height / 2
        
    def choose_action(self, state_vector: np.ndarray) -> int:
        """
        Choose action based on threats and opportunities.
        
        State vector structure:
        [0-2]:   self_x, self_y, self_radius
        [3]:     score
        [4-12]:  3 nearest viruses (x, y, radius each)
        [13-21]: 3 nearest other players (x, y, radius each)
        [22]:    food_count
        [23]:    game_ended
        """
        # Extract our position
        self_x, self_y, self_radius = state_vector[0:3]
        
        # Check if we have valid position
        if self_x == -1 or self_y == -1:
            return np.random.randint(0, NUM_ACTIONS)
        
        # Find nearest threat (player larger than us)
        nearest_threat = None
        min_threat_dist = float('inf')
        
        for i in range(3):  # Check 3 nearest players
            idx = 13 + i * 3
            p_x, p_y, p_radius = state_vector[idx:idx+3]
            
            if p_x == -1:  # No player at this slot
                continue
                
            # Check if this player is a threat (larger than us)
            if p_radius > self_radius:
                dist = np.sqrt((p_x - self_x)**2 + (p_y - self_y)**2)
                if dist < min_threat_dist:
                    min_threat_dist = dist
                    nearest_threat = (p_x, p_y)
        
        # If threat is close, run away
        if nearest_threat and min_threat_dist < 200:
            threat_x, threat_y = nearest_threat
            
            # Calculate direction away from threat
            dx = self_x - threat_x
            dy = self_y - threat_y
            
            # Choose action that moves away from threat
            if abs(dx) > abs(dy):
                return 3 if dx > 0 else 2  # Right or Left
            else:
                return 1 if dy > 0 else 0  # Down or Up
        
        # No immediate threat: move toward center or randomly explore
        if np.random.random() < 0.3:  # 30% random exploration
            return np.random.randint(0, NUM_ACTIONS)
        
        # Move toward center
        dx = self.center_x - self_x
        dy = self.center_y - self_y
        
        if abs(dx) > abs(dy):
            return 3 if dx > 0 else 2  # Right or Left
        else:
            return 1 if dy > 0 else 0  # Down or Up


# ==============================================================================
# ACTION EXECUTION
# ==============================================================================

def execute_action(action_index: int, action_mapping: dict) -> None:
    """
    Execute the selected action by calling the appropriate ScreenAction method.
    
    Parameters:
    -----------
    action_index : int
        Index of the action to execute (0-7)
    action_mapping : dict
        Dictionary mapping action indices to ScreenAction methods
    """
    if action_index not in action_mapping:
        print(f"Warning: Invalid action index {action_index}, defaulting to 0 (UP)")
        action_index = 0
    
    # Call the movement method
    action_mapping[action_index]()


# ==============================================================================
# MAIN CONTROL LOOP
# ==============================================================================

def run_control_loop(
    policy,
    max_iterations: int = None,
    action_delay: float = ACTION_DELAY,
    verbose: bool = True
) -> None:
    """
    Run the main RL control loop.
    
    This is the core loop that:
    1. Captures the current game state
    2. Feeds state to policy to get action
    3. Executes the action in the game
    4. Repeats until game ends or max iterations reached
    
    Parameters:
    -----------
    policy : object
        Policy object with choose_action(state_vector) method
    max_iterations : int or None
        Maximum number of iterations. None = run indefinitely
    action_delay : float
        Delay between actions in seconds
    verbose : bool
        Whether to print status information
    """
    # Initialize screen action executor
    screen_width = pyautogui.size().width
    screen_height = pyautogui.size().height
    screen_action = ScreenAction(screen_width, screen_height)
    
    # Create action mapping
    action_mapping = create_action_mapping(screen_action)
    
    if verbose:
        print("=" * 70)
        print("AGAR.IO RL CONTROL LOOP STARTED")
        print("=" * 70)
        print(f"Screen size: {screen_width} x {screen_height}")
        print(f"Action delay: {action_delay}s")
        print(f"Max iterations: {max_iterations if max_iterations else 'Unlimited'}")
        print("Press Ctrl+C to stop")
        print("=" * 70)
    
    iteration = 0
    
    try:
        while True:
            # Check iteration limit
            if max_iterations and iteration >= max_iterations:
                if verbose:
                    print(f"\nReached max iterations ({max_iterations}). Stopping.")
                break
            
            iteration += 1
            
            # =========================================
            # STEP 1: CAPTURE GAME STATE
            # =========================================
            game_state, state_vector, _ = get_full_game_state(debug=False)
            
            # Check if game has ended
            if game_state['game_ended']:
                if verbose:
                    print(f"\n[Iteration {iteration}] GAME ENDED! Final score: {game_state['score']}")
                break
            
            # =========================================
            # STEP 2: CHOOSE ACTION VIA POLICY
            # =========================================
            action = policy.choose_action(state_vector)
            
            # =========================================
            # STEP 3: EXECUTE ACTION
            # =========================================
            execute_action(action, action_mapping)
            
            # =========================================
            # LOGGING
            # =========================================
            if verbose:
                self_pos = game_state['self_position']
                pos_str = f"({self_pos[0]}, {self_pos[1]})" if self_pos else "N/A"
                print(f"[{iteration:4d}] Pos: {pos_str:12s} | "
                      f"Score: {game_state['score'] or 0:4d} | "
                      f"Action: {ACTION_NAMES[action]:20s} | "
                      f"Enemies: {len(game_state['other_players']):2d} | "
                      f"Food: {game_state['food_count']:3d}")
            
            # =========================================
            # STEP 4: DELAY BEFORE NEXT ACTION
            # =========================================
            time.sleep(action_delay)
            
    except KeyboardInterrupt:
        if verbose:
            print(f"\n\n{'=' * 70}")
            print("CONTROL LOOP STOPPED BY USER")
            print(f"Total iterations: {iteration}")
            print(f"{'=' * 70}")


# ==============================================================================
# TRAINING DATA COLLECTION (FOR FUTURE RL TRAINING)
# ==============================================================================

def collect_training_data(
    num_episodes: int = 10,
    max_steps_per_episode: int = 1000,
    action_delay: float = ACTION_DELAY
) -> list:
    """
    Collect transition data for offline RL training.
    
    This function runs episodes with a random policy and records
    (state, action, reward, next_state) tuples for later training.
    
    Parameters:
    -----------
    num_episodes : int
        Number of episodes to collect
    max_steps_per_episode : int
        Maximum steps per episode
    action_delay : float
        Delay between actions
        
    Returns:
    --------
    list : List of transitions [(s, a, r, s'), ...]
    
    NOTE: You'll need to define your own reward function!
    """
    transitions = []
    
    # Initialize screen action
    screen_width = pyautogui.size().width
    screen_height = pyautogui.size().height
    screen_action = ScreenAction(screen_width, screen_height)
    action_mapping = create_action_mapping(screen_action)
    
    for episode in range(num_episodes):
        print(f"\n=== EPISODE {episode + 1}/{num_episodes} ===")
        
        for step in range(max_steps_per_episode):
            # Capture current state
            game_state, state_vector, _ = get_full_game_state()
            
            if game_state['game_ended']:
                print(f"Episode ended at step {step}")
                break
            
            # Random action (for exploration data collection)
            action = np.random.randint(0, NUM_ACTIONS)
            
            # Execute action
            execute_action(action, action_mapping)
            time.sleep(action_delay)
            
            # Capture next state
            next_game_state, next_state_vector, _ = get_full_game_state()
            
            # =========================================
            # DEFINE YOUR REWARD FUNCTION HERE
            # =========================================
            # Example reward ideas:
            # - Score increase: next_score - current_score
            # - Survival bonus: +1 for each step alive
            # - Death penalty: -100 if game ended
            # - Food eaten: based on food count change
            
            current_score = game_state['score'] or 0
            next_score = next_game_state['score'] or 0
            
            # Simple reward: score delta + survival bonus
            reward = (next_score - current_score) + 0.1  # +0.1 per step alive
            
            if next_game_state['game_ended']:
                reward = -100  # Death penalty
            
            # Store transition
            transitions.append({
                'state': state_vector.copy(),
                'action': action,
                'reward': reward,
                'next_state': next_state_vector.copy(),
                'done': next_game_state['game_ended']
            })
            
        print(f"Collected {len(transitions)} transitions so far")
    
    return transitions


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    """
    Main entry point for running the Agar.io RL control loop.
    
    Usage:
        python examplePolicyFile.py
    
    The bot will:
    1. Start capturing the game screen
    2. Use the SimplePolicy to choose actions
    3. Execute actions by moving the mouse
    4. Continue until interrupted (Ctrl+C) or game ends
    """
    
    # Give user time to switch to game window
    print("Starting in 3 seconds... Switch to Agar.io game window!")
    time.sleep(3)
    
    # Initialize policy
    # =========================================
    # REPLACE THIS WITH YOUR TRAINED POLICY:
    # =========================================
    # Option 1: Simple rule-based policy (default)
    policy = SimplePolicy(
        screen_width=MONITOR_REGION['width'],
        screen_height=GAME_AREA_HEIGHT
    )
    
    # Option 2: Slightly smarter policy
    # policy = SmartPolicy(
    #     screen_width=MONITOR_REGION['width'],
    #     screen_height=GAME_AREA_HEIGHT
    # )
    
    # Option 3: Your trained RL policy
    # policy = YourTrainedPolicy.load("model_checkpoint.pt")
    
    # Run the control loop
    run_control_loop(
        policy=policy,
        max_iterations=None,  # Run indefinitely (or set a limit)
        action_delay=0.15,    # 150ms between actions
        verbose=True          # Print status updates
    )
