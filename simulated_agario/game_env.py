"""
Simulated Agar.io Environment - Cross-Compatible Version
State space matches EXACTLY with the live game (24-dimensional vector)
Compatible with models trained for the real Agar.io browser game.

Features:
- Multiple bots of varying sizes with different behaviors
- Food particles (not in state, but player grows when eating)
- Player starts as smallest
- Episode ends only when player gets eaten

Reward Structure (incentivizes growth, penalizes passive play):
- Eating food: +1.0 per food
- Eating bots: +50 + bot_radius (big reward for kills!)
- WINNING (largest in game): +1000 (huge win bonus!)
- Death without growth: -500 (huge penalty for dying small)
- Death after some growth: -300 to -50 (scaled by growth)
- Growth delta: +1.0 per unit of score increase
- Size bonus: +0.5 * (size_ratio - 1) per step (continuous reward for being big)
- Wall/edge penalty: -0.3 per edge touched
- Corner penalty: -0.5 extra (discourages corner camping)
- Danger penalty: -2.0 scaled when near larger bots
- NO survival bonus (don't reward passive play)

Episode ends when:
- Player is eaten (death)
- Player reaches size 80+ (SIZE WIN!)
- Player becomes 20% larger than the biggest bot (DOMINANCE WIN!)

State Vector (24 values):
- Self position: x, y, radius (3)
- Score (1)  
- 3 nearest viruses: x, y, radius each (9) - filled with -1 in simulation
- 3 nearest other players: x, y, radius each (9)
- Food count (1) - always 0 (food not visible to agent)
- Game ended flag (1)
"""

import numpy as np
import random
import math


class Food:
    """Food particle - not visible to agent but grows player when eaten"""
    __slots__ = ['x', 'y', 'radius', 'color']
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = random.uniform(4, 8)  # Slightly bigger food
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
                  '#DDA0DD', '#98D8C8', '#F7DC6F', '#85C1E9', '#00CED1']
        self.color = random.choice(colors)


class Bot:
    """A bot with linear behavior patterns - speed decreases with size like real Agar.io"""
    __slots__ = ['x', 'y', 'radius', 'start_radius', 'color', 'base_speed', 
                 'behavior', 'alive', 'direction', 'steps_in_dir', 'steps_per_side',
                 'target_x', 'target_y', 'wander_angle', 'wander_timer']
    
    # Speed scaling constants (like real Agar.io)
    # Formula: speed = base_speed * (BASE_RADIUS / radius) ^ SPEED_DECAY_POWER
    BASE_RADIUS = 15.0  # Reference radius for speed calculation
    SPEED_DECAY_POWER = 0.5  # How aggressively speed decreases (0.5 = sqrt scaling)
    MIN_SPEED_MULT = 0.3  # Minimum speed multiplier (huge blobs don't go below 30% base speed)
    
    def __init__(self, x, y, radius, color, speed=2, behavior='patrol'):
        self.x = x
        self.y = y
        self.radius = radius
        self.start_radius = radius
        self.color = color
        self.base_speed = speed  # Store base speed, actual speed calculated dynamically
        self.behavior = behavior
        self.alive = True
        
        # Movement state
        self.direction = random.randint(0, 3)
        self.steps_in_dir = 0
        self.steps_per_side = random.randint(40, 100)
        self.target_x = x
        self.target_y = y
        self.wander_angle = random.random() * 2 * math.pi
        self.wander_timer = 0
    
    @property
    def speed(self):
        """Calculate effective speed based on current radius (larger = slower)"""
        # Speed decreases as radius increases, mimicking real Agar.io
        # Small blobs (radius ~10-15) move at full speed
        # Large blobs (radius ~80+) move much slower
        speed_mult = (self.BASE_RADIUS / self.radius) ** self.SPEED_DECAY_POWER
        speed_mult = max(self.MIN_SPEED_MULT, min(1.0, speed_mult))
        return self.base_speed * speed_mult
    
    def update(self, game_width, game_height, player_x=None, player_y=None, player_radius=None):
        """Update bot position based on behavior"""
        if not self.alive:
            return
        
        # Get current effective speed (decreases with size)
        current_speed = self.speed
        
        if self.behavior == 'patrol':
            # Square patrol pattern
            if self.direction == 0:  # Right
                self.x += current_speed
            elif self.direction == 1:  # Down
                self.y += current_speed
            elif self.direction == 2:  # Left
                self.x -= current_speed
            elif self.direction == 3:  # Up
                self.y -= current_speed
            
            self.steps_in_dir += 1
            if self.steps_in_dir >= self.steps_per_side:
                self.direction = (self.direction + 1) % 4
                self.steps_in_dir = 0
        
        elif self.behavior == 'chase' and player_x is not None:
            # Chase player if bigger, otherwise wander
            if self.radius > player_radius * 1.1:
                dx = player_x - self.x
                dy = player_y - self.y
                dist = math.sqrt(dx * dx + dy * dy)
                if dist > 0 and dist < 400:  # Chase range
                    self.x += (dx / dist) * current_speed
                    self.y += (dy / dist) * current_speed
                else:
                    self._wander()
            else:
                self._wander()
        
        elif self.behavior == 'flee' and player_x is not None:
            # Flee from player if player is bigger
            if player_radius > self.radius * 1.1:
                dx = self.x - player_x
                dy = self.y - player_y
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < 300 and dist > 0:  # Flee range
                    self.x += (dx / dist) * current_speed * 1.2
                    self.y += (dy / dist) * current_speed * 1.2
                else:
                    self._wander()
            else:
                self._wander()
        
        elif self.behavior == 'smart' and player_x is not None:
            # Smart: chase if bigger, flee if smaller, otherwise patrol
            dist = math.sqrt((player_x - self.x)**2 + (player_y - self.y)**2)
            
            if self.radius > player_radius * 1.2 and dist < 350:
                # Chase
                dx = player_x - self.x
                dy = player_y - self.y
                d = math.sqrt(dx*dx + dy*dy)
                if d > 0:
                    self.x += (dx/d) * current_speed
                    self.y += (dy/d) * current_speed
            elif player_radius > self.radius * 1.2 and dist < 250:
                # Flee
                dx = self.x - player_x
                dy = self.y - player_y
                d = math.sqrt(dx*dx + dy*dy)
                if d > 0:
                    self.x += (dx/d) * current_speed * 1.3
                    self.y += (dy/d) * current_speed * 1.3
            else:
                self._wander()
        
        elif self.behavior == 'zigzag':
            # Zigzag pattern - diagonal movement
            self.wander_timer += 1
            if self.wander_timer > 30:
                self.direction = (self.direction + 1) % 4
                self.wander_timer = 0
            
            if self.direction == 0:
                self.x += current_speed
                self.y -= current_speed * 0.5
            elif self.direction == 1:
                self.x += current_speed
                self.y += current_speed * 0.5
            elif self.direction == 2:
                self.x -= current_speed
                self.y += current_speed * 0.5
            else:
                self.x -= current_speed
                self.y -= current_speed * 0.5
        
        else:
            self._wander()
        
        # Keep within bounds
        self.x = max(self.radius, min(game_width - self.radius, self.x))
        self.y = max(self.radius, min(game_height - self.radius, self.y))
    
    def _wander(self):
        """Random wander movement"""
        self.wander_timer += 1
        if self.wander_timer > 50:
            self.wander_angle += random.uniform(-0.8, 0.8)
            self.wander_timer = 0
        
        current_speed = self.speed
        self.x += math.cos(self.wander_angle) * current_speed * 0.8
        self.y += math.sin(self.wander_angle) * current_speed * 0.8
    
    def reset(self, x, y):
        """Reset bot to a position"""
        self.x = x
        self.y = y
        self.radius = self.start_radius
        self.alive = True
        self.direction = random.randint(0, 3)
        self.steps_in_dir = 0
        self.wander_angle = random.random() * 2 * math.pi


class Player:
    """The RL agent's blob - optimized for speed"""
    __slots__ = ['x', 'y', 'radius', 'start_radius', 'color', 'alive', '_score']
    
    def __init__(self, x, y, radius=15, color='#3498DB'):
        self.x = x
        self.y = y
        self.radius = radius
        self.start_radius = radius
        self.color = color
        self.alive = True
        self._score = radius
    
    @property
    def score(self):
        """Score = current size (cached)"""
        return self._score
    
    def move(self, dx, dy, game_width, game_height):
        """Move player by delta, clamping to bounds (optimized)"""
        self.x = max(self.radius, min(game_width - self.radius, self.x + dx))
        self.y = max(self.radius, min(game_height - self.radius, self.y + dy))
    
    def grow(self, amount):
        """Grow the player (area-based, optimized)"""
        # Simplified: radius = sqrt((pi * r^2 + amount) / pi) = sqrt(r^2 + amount/pi)
        self.radius = math.sqrt(self.radius * self.radius + amount * 0.3183098861837907)  # 1/pi
        self._score = int(self.radius)
    
    def reset(self, x, y):
        """Reset player to initial state"""
        self.x = x
        self.y = y
        self.radius = self.start_radius
        self.alive = True
        self._score = int(self.start_radius)


class AgarioSimEnv:
    """
    Gym-like environment for simulated Agar.io
    
    CROSS-COMPATIBLE: State space matches exactly with live game
    State: 24-dimensional vector (same as capture_game.game_state_to_vector)
    Actions: 8 discrete directions (same as rl_environment.AgarioEnv)
    
    Features:
    - Player starts as smallest blob
    - Multiple bots with varying sizes and behaviors
    - Food particles (invisible to agent but grows player)
    - Episode ends only when player is eaten
    """
    
    # Game dimensions - larger map for more exploration
    GAME_WIDTH = 3000
    GAME_HEIGHT = 2000
    
    # Speed scaling constants (same as Bot class for consistency)
    BASE_RADIUS = 15.0  # Reference radius for speed calculation
    SPEED_DECAY_POWER = 0.5  # How aggressively speed decreases (0.5 = sqrt scaling)
    MIN_SPEED_MULT = 0.3  # Minimum speed multiplier
    
    def __init__(self, width=None, height=None, num_food=150):
        self.width = width or self.GAME_WIDTH
        self.height = height or self.GAME_HEIGHT
        self.num_food = num_food
        
        # Game objects
        self.player = None
        self.bots = []
        self.food = []
        
        # Action space: 8 directions
        self.action_dim = 8
        self.base_move_speed = 10  # Base movement speed (at radius 15)
        
        # State dimension: 24 (MUST match capture_game.game_state_to_vector)
        self.state_dim = 24
        
        # Direction vectors (MUST match rl_environment.py order)
        diag = 1 / math.sqrt(2)
        self.directions = [
            (0, -1),        # 0: Up
            (0, 1),         # 1: Down
            (-1, 0),        # 2: Left
            (1, 0),         # 3: Right
            (diag, -diag),  # 4: Up-Right
            (diag, diag),   # 5: Down-Right
            (-diag, -diag), # 6: Up-Left
            (-diag, diag),  # 7: Down-Left
        ]
        
        # Stats
        self.steps = 0
        self.episode = 0
        self.total_kills = 0
        self.total_deaths = 0
        self.total_wins = 0
        self.episode_kills = 0  # Kills in current episode
        self.last_score = 0
        self.food_eaten = 0
        
        self.reset()
    
    def _spawn_food(self):
        """Spawn food particles"""
        self.food = []
        for _ in range(self.num_food):
            x = random.uniform(20, self.width - 20)
            y = random.uniform(20, self.height - 20)
            self.food.append(Food(x, y))
    
    def _spawn_bots(self):
        """Spawn bots with varying sizes and behaviors"""
        self.bots = []
        
        # Bot configurations: (count, size_range, behaviors, colors, speed_range)
        # Fewer bots on larger map - Player starts at radius 15
        configs = [
            # Tiny bot - easy prey (smaller than player)
            (1, (8, 12), ['flee', 'zigzag'], ['#98D8C8', '#85C1E9'], (3.5, 5)),
            # Small bot - slightly smaller or equal
            (1, (12, 18), ['flee', 'patrol'], ['#2ECC71', '#27AE60'], (3, 4.5)),
            # Medium bot - competitive
            (1, (20, 35), ['smart', 'chase'], ['#F39C12', '#E67E22'], (2.5, 3.5)),
            # Large bot - dangerous predator
            (1, (40, 60), ['chase', 'smart'], ['#E74C3C', '#C0392B'], (2, 3)),
            # Boss bot - very large
            (1, (70, 90), ['patrol', 'smart'], ['#8E44AD'], (1.5, 2.5)),
        ]
        
        player_x, player_y = self.width * 0.2, self.height * 0.5
        
        for count, size_range, behaviors, colors, speed_range in configs:
            for _ in range(count):
                # Spawn away from player
                for _ in range(20):
                    x = random.uniform(100, self.width - 100)
                    y = random.uniform(100, self.height - 100)
                    dist = math.sqrt((x - player_x)**2 + (y - player_y)**2)
                    if dist > 200:
                        break
                
                bot = Bot(
                    x, y,
                    radius=random.uniform(*size_range),
                    color=random.choice(colors),
                    speed=random.uniform(*speed_range),
                    behavior=random.choice(behaviors)
                )
                self.bots.append(bot)
    
    def _distance_sq(self, x1, y1, x2, y2):
        """Squared distance (faster - avoids sqrt)"""
        dx = x2 - x1
        dy = y2 - y1
        return dx * dx + dy * dy
    
    def _distance(self, x1, y1, x2, y2):
        """Euclidean distance"""
        return math.sqrt(self._distance_sq(x1, y1, x2, y2))
    
    def _check_collision(self, x1, y1, r1, x2, y2, r2):
        """Check if two circular objects are colliding (optimized)"""
        # Use squared distance to avoid sqrt
        dx = x2 - x1
        dy = y2 - y1
        dist_sq = dx * dx + dy * dy
        collision_dist = max(r1, r2) * 0.75
        return dist_sq < collision_dist * collision_dist
    
    def _get_state(self):
        """
        Get current state as numpy vector
        FORMAT MATCHES EXACTLY: capture_game.game_state_to_vector()
        
        NOTE: Food is NOT included in state - agent cannot "see" food
        """
        vector = []
        
        # 1. Self position (3 values)
        if self.player.alive:
            vector.extend([self.player.x, self.player.y, self.player.radius])
        else:
            vector.extend([-1, -1, -1])
        
        # 2. Score (1 value)
        vector.append(self.player.score if self.player.alive else 0)
        
        # 3. 3 Nearest viruses (9 values) - no viruses, fill with -1
        for _ in range(3):
            vector.extend([-1, -1, -1])
        
        # 4. 3 Nearest other players (9 values) - bots count as players (optimized)
        if self.player.alive:
            self_x, self_y = self.player.x, self.player.y
            
            # Use squared distance for sorting (faster)
            player_distances = []
            for bot in self.bots:
                if bot.alive:
                    dx = bot.x - self_x
                    dy = bot.y - self_y
                    dist_sq = dx * dx + dy * dy
                    player_distances.append((dist_sq, bot.x, bot.y, bot.radius))
            
            player_distances.sort()  # Sort by squared distance
            
            # Take 3 nearest
            for i in range(3):
                if i < len(player_distances):
                    _, p_x, p_y, p_radius = player_distances[i]
                    vector.extend([p_x, p_y, p_radius])
                else:
                    vector.extend([-1, -1, -1])
        else:
            # Dead - fill with -1
            for _ in range(3):
                vector.extend([-1, -1, -1])
        
        # 5. Food count (1 value) - always 0 (food invisible to agent)
        vector.append(0)
        
        # 6. Game ended flag (1 value)
        vector.append(1 if not self.player.alive else 0)
        
        return np.array(vector, dtype=np.float32)
    
    def reset(self):
        """Reset environment for new episode"""
        self.episode += 1
        self.steps = 0
        self.food_eaten = 0
        self.episode_kills = 0  # Reset kills for new episode
        
        # Player starts as SMALLEST (radius 15)
        start_x = self.width * 0.2
        start_y = self.height * 0.5
        self.player = Player(start_x, start_y, radius=15, color='#3498DB')
        self.last_score = self.player.score
        
        # Spawn bots and food
        self._spawn_bots()
        self._spawn_food()
        
        return self._get_state()
    
    def _get_player_speed(self):
        """Calculate player's effective speed based on radius (larger = slower)"""
        if self.player is None:
            return self.base_move_speed
        speed_mult = (self.BASE_RADIUS / self.player.radius) ** self.SPEED_DECAY_POWER
        speed_mult = max(self.MIN_SPEED_MULT, min(1.0, speed_mult))
        return self.base_move_speed * speed_mult
    
    def step(self, action):
        """Execute action and return (state, reward, done, info) - OPTIMIZED"""
        self.steps += 1
        reward = 0
        done = False
        
        # Move player with size-scaled speed (larger = slower, like real Agar.io)
        dx, dy = self.directions[action]
        current_speed = self._get_player_speed()
        self.player.move(dx * current_speed, dy * current_speed, 
                        self.width, self.height)
        
        px, py, pr = self.player.x, self.player.y, self.player.radius
        
        # Update all bots (pass player info once)
        for bot in self.bots:
            bot.update(self.width, self.height, px, py, pr)
        
        # Check food collisions (optimized - inline collision check)
        food_to_remove = []
        collision_dist_sq = (pr * 0.75) ** 2  # Pre-compute for food
        
        for i, f in enumerate(self.food):
            dx = f.x - px
            dy = f.y - py
            if dx * dx + dy * dy < collision_dist_sq:
                food_to_remove.append(i)
                self.player.grow(f.radius ** 2 * 3)
                self.food_eaten += 2
                reward += 1.0  # Increased food reward
        
        # Remove in reverse to maintain indices
        for i in reversed(food_to_remove):
            self.food.pop(i)
        
        # Respawn food
        while len(self.food) < self.num_food:
            x = random.uniform(20, self.width - 20)
            y = random.uniform(20, self.height - 20)
            self.food.append(Food(x, y))
        
        # Check bot collisions (optimized - inline checks)
        pr = self.player.radius  # Update after food growth
        starting_radius = self.player.start_radius
        
        for bot in self.bots:
            if not bot.alive:
                continue
            
            # Inline collision check
            dx = bot.x - px
            dy = bot.y - py
            dist_sq = dx * dx + dy * dy
            collision_dist = max(pr, bot.radius) * 0.75
            
            if dist_sq < collision_dist * collision_dist:
                # Simple rule: BIGGER WINS (no threshold - if you're bigger, you eat them)
                if pr > bot.radius:
                    # Player eats bot - player is bigger!
                    reward += 50 + bot.radius * 1.0  # Big reward for kills
                    self.player.grow(bot.radius ** 2)
                    bot.alive = False
                    self.total_kills += 1
                    self.episode_kills += 1  # Track kills this episode
                    # Update pr since we grew
                    pr = self.player.radius
                elif bot.radius > pr:
                    # Bot eats player - bot is bigger, GAME OVER
                    # Penalty scales with how little the player grew
                    # If died at starting size: -500, if grew a lot: smaller penalty
                    growth_ratio = pr / starting_radius  # 1.0 = no growth, higher = grew
                    if growth_ratio < 1.5:
                        # Died without growing much - HUGE penalty
                        reward -= 500
                    elif growth_ratio < 2.0:
                        reward -= 300
                    elif growth_ratio < 3.0:
                        reward -= 150
                    else:
                        # Died but grew a lot - smaller penalty
                        reward -= 50
                    
                    done = True
                    self.player.alive = False
                    self.total_deaths += 1
                    break
                # If exactly equal radius, nothing happens (rare)
        
        # Respawn dead bots (away from player, optimized)
        for bot in self.bots:
            if not bot.alive:
                # Quick spawn without too many attempts
                for _ in range(8):
                    x = random.uniform(100, self.width - 100)
                    y = random.uniform(100, self.height - 100)
                    dx = x - px
                    dy = y - py
                    if dx * dx + dy * dy > 62500:  # 250^2
                        break
                bot.reset(x, y)
                # Respawn with varied size
                bot.radius = random.uniform(10, max(50, pr * 0.7))
                bot.start_radius = bot.radius
        
        # Reward for growth (delta score) - INCREASED
        current_score = self.player.score
        score_delta = current_score - self.last_score
        reward += score_delta * 1.0  # Tripled growth reward
        self.last_score = current_score
        
        # CONTINUOUS SIZE REWARD - being big is always good!
        # Scales exponentially with size - bigger = much more reward per step
        size_bonus = (pr / starting_radius - 1.0) * 0.5  # 0 at start, grows with size
        reward += max(0, size_bonus)
        
        # NO survival bonus - don't reward passive play
        
        # WALL/CORNER PENALTY - discourage camping at edges
        edge_margin = 100
        if px < edge_margin or px > self.width - edge_margin:
            reward -= 0.3
        if py < edge_margin or py > self.height - edge_margin:
            reward -= 0.3
        
        # Corner penalty (in both edge zones = corner)
        in_x_edge = px < edge_margin or px > self.width - edge_margin
        in_y_edge = py < edge_margin or py > self.height - edge_margin
        if in_x_edge and in_y_edge:
            reward -= 0.5  # Extra penalty for corners
        
        if not done:
            # Update pr to current radius (may have grown from eating bots)
            pr = self.player.radius
            
            # Danger penalty when near larger bots
            for bot in self.bots:
                if bot.alive and bot.radius > pr * 1.1:
                    dist = self._distance(px, py, bot.x, bot.y)
                    if dist < bot.radius * 3:
                        reward -= (1 - dist / (bot.radius * 3)) * 2.0  # Doubled danger penalty
            
            # WIN CONDITIONS:
            # 1. Player reaches size 80+ (automatic win!)
            # 2. Player is 20% larger than all bots
            largest_bot_radius = max((b.radius for b in self.bots if b.alive), default=0)
            
            if pr >= 80:
                # SIZE VICTORY! Reached target size
                reward += 1000  # Huge win bonus!
                done = True
                self.total_wins += 1
            elif pr > largest_bot_radius * 1.2:
                # DOMINANCE VICTORY! Bigger than all bots
                reward += 1000  # Huge win bonus!
                done = True
                self.total_wins += 1
        
        # Episode ends when eaten OR when player becomes dominant
        
        # Track if this was a win
        won = done and self.player.alive
        
        info = {
            'score': self.player.score,
            'player_radius': self.player.radius,
            'alive_bots': sum(1 for b in self.bots if b.alive),
            'food_eaten': self.food_eaten,
            'steps': self.steps,
            'episode': self.episode,
            'total_kills': self.total_kills,
            'total_deaths': self.total_deaths,
            'total_wins': self.total_wins,
            'episode_kills': self.episode_kills,  # Kills in this episode
            'won': won
        }
        
        return self._get_state(), reward, done, info
    
    def get_game_state(self):
        """Get full game state for visualization"""
        return {
            'player': {
                'x': self.player.x,
                'y': self.player.y,
                'radius': self.player.radius,
                'color': self.player.color,
                'alive': self.player.alive,
                'score': self.player.score
            },
            'bots': [
                {
                    'x': b.x, 
                    'y': b.y, 
                    'radius': b.radius, 
                    'color': b.color,
                    'alive': b.alive,
                    'behavior': b.behavior
                }
                for b in self.bots
            ],
            'food': [
                {'x': f.x, 'y': f.y, 'radius': f.radius, 'color': f.color}
                for f in self.food
            ],
            'game_info': {
                'width': self.width,
                'height': self.height,
                'episode': self.episode,
                'steps': self.steps,
                'total_kills': self.total_kills,
                'total_deaths': self.total_deaths,
                'food_eaten': self.food_eaten
            }
        }
    
    def sample_action(self):
        """Return random action"""
        return random.randint(0, self.action_dim - 1)


# For self-play
class ModelOpponent:
    """An opponent controlled by a trained model"""
    def __init__(self, x, y, radius, model, color='#E74C3C'):
        self.x = x
        self.y = y
        self.radius = radius
        self.start_radius = radius
        self.color = color
        self.model = model
        self.alive = True
    
    def get_action(self, state):
        return self.model.select_action(state, training=False)
    
    def move(self, dx, dy, game_width, game_height):
        self.x += dx
        self.y += dy
        self.x = max(self.radius, min(game_width - self.radius, self.x))
        self.y = max(self.radius, min(game_height - self.radius, self.y))
    
    def grow(self, amount):
        current_area = math.pi * self.radius ** 2
        new_area = current_area + amount
        self.radius = math.sqrt(new_area / math.pi)
    
    def reset(self, x, y):
        self.x = x
        self.y = y
        self.radius = self.start_radius
        self.alive = True


class SelfPlayEnv(AgarioSimEnv):
    """Environment for self-play training (NFSP)"""
    def __init__(self, opponent_model=None, **kwargs):
        # Set these BEFORE super().__init__() since reset() is called there
        self.opponent_model = opponent_model
        self.opponent = None
        super().__init__(**kwargs)
    
    def reset(self):
        state = super().reset()
        
        if self.opponent_model is not None:
            self.opponent = ModelOpponent(
                x=self.width * 0.8,
                y=self.height * 0.5,
                radius=15,
                model=self.opponent_model,
                color='#E74C3C'
            )
        
        return state
    
    def _get_opponent_speed(self):
        """Calculate opponent's effective speed based on radius"""
        if self.opponent is None:
            return self.base_move_speed
        speed_mult = (self.BASE_RADIUS / self.opponent.radius) ** self.SPEED_DECAY_POWER
        speed_mult = max(self.MIN_SPEED_MULT, min(1.0, speed_mult))
        return self.base_move_speed * speed_mult
    
    def step(self, action):
        if self.opponent is not None and self.opponent_model is not None:
            opp_state = self._get_opponent_state()
            opp_action = self.opponent.get_action(opp_state)
            
            dx, dy = self.directions[opp_action]
            opp_speed = self._get_opponent_speed()
            self.opponent.move(dx * opp_speed, dy * opp_speed,
                              self.width, self.height)
        
        return super().step(action)
    
    def _get_opponent_state(self):
        vector = []
        vector.extend([self.opponent.x, self.opponent.y, self.opponent.radius])
        vector.append(int(self.opponent.radius))
        
        for _ in range(3):
            vector.extend([-1, -1, -1])
        
        all_players = [{'x': self.player.x, 'y': self.player.y, 'radius': self.player.radius}]
        for bot in self.bots:
            if bot.alive:
                all_players.append({'x': bot.x, 'y': bot.y, 'radius': bot.radius})
        
        player_distances = []
        for p in all_players:
            dist = self._distance(self.opponent.x, self.opponent.y, p['x'], p['y'])
            player_distances.append((dist, p['x'], p['y'], p['radius']))
        player_distances.sort(key=lambda p: p[0])
        
        for i in range(3):
            if i < len(player_distances):
                _, p_x, p_y, p_radius = player_distances[i]
                vector.extend([p_x, p_y, p_radius])
            else:
                vector.extend([-1, -1, -1])
        
        vector.append(0)
        vector.append(0)
        
        return np.array(vector, dtype=np.float32)


if __name__ == "__main__":
    env = AgarioSimEnv()
    state = env.reset()
    
    print(f"State dimension: {env.state_dim}")
    print(f"Action dimension: {env.action_dim}")
    print(f"Initial state shape: {state.shape}")
    print(f"Player starting radius: {env.player.radius}")
    print(f"Number of bots: {len(env.bots)}")
    print(f"Bot sizes: {[int(b.radius) for b in env.bots]}")
    print(f"Number of food: {len(env.food)}")
    print(f"\nState vector breakdown:")
    print(f"  Self (x,y,r): {state[0:3]}")
    print(f"  Score: {state[3]}")
    print(f"  Viruses: {state[4:13]}")
    print(f"  Players: {state[13:22]}")
    print(f"  Food count: {state[22]}")
    print(f"  Game ended: {state[23]}")
