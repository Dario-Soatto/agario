"""
Action Visualization for Agar.io
================================
Shows three types of action overlays on the game screen:
1. 4 Cardinal Directions (UP, DOWN, LEFT, RIGHT)
2. 8 Directions (including diagonals)
3. Arrows to all detected food centroids

Press 'q' to quit, runs every 5 seconds.
"""

import cv2
import mss
import numpy as np
import time


class ActionVisualizer:
    """Visualizes different action types on the Agar.io screen"""
    
    def __init__(self):
        # Get monitor info (same pattern as detection files)
        with mss.mss() as sct:
            if len(sct.monitors) > 2:
                self.monitor = sct.monitors[2]
            else:
                self.monitor = sct.monitors[1]
        
        # Game region with padding (from detectFood.py)
        padding_x = int(self.monitor['width'] * 0.05)
        padding_y = int(self.monitor['height'] * 0.05)
        self.region = {
            "top": self.monitor['top'] + padding_y,
            "left": self.monitor['left'] + padding_x,
            "width": self.monitor['width'] - 2 * padding_x,
            "height": self.monitor['height'] - 2 * padding_y
        }
        
        # Center of screen (where player always is)
        self.center_x = self.region['width'] // 2
        self.center_y = self.region['height'] // 2
        
        # Arrow length for direction visualization
        self.arrow_length = min(self.region['width'], self.region['height']) * 0.15
        
        # Persistent mss for speed
        self.sct = mss.mss()
    
    def _capture(self):
        """Capture the game screen"""
        raw = np.array(self.sct.grab(self.region), dtype=np.uint8)
        return cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
    
    def _find_food(self, img):
        """
        Find food blobs - EXACT logic from detectFood.py
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Detect saturated colors (food pellets are colorful)
        mask = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([180, 255, 255]))
        # Smaller kernel for food (they're tiny)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Minimum distance from center to not be our player
        min_dist_from_center = min(self.center_x, self.center_y) * 0.15
        
        food = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > 10:
                (x, y), radius = cv2.minEnclosingCircle(c)
                x, y, radius = int(x), int(y), int(radius)
                
                # Food is small (radius < 60) and not too tiny (radius > 3)
                if not (3 < radius < 60):
                    continue
                
                # Exclude blobs near center (that's our player)
                dist_from_center = ((x - self.center_x)**2 + (y - self.center_y)**2)**0.5
                if dist_from_center < min_dist_from_center:
                    continue
                
                food.append({'x': x, 'y': y, 'radius': radius, 'dist': dist_from_center})
        
        # Sort by distance
        food.sort(key=lambda f: f['dist'])
        return food
    
    def _draw_arrow(self, img, start, end, color, thickness=2, tip_length=0.3):
        """Draw an arrow with label"""
        cv2.arrowedLine(img, start, end, color, thickness, tipLength=tip_length)
    
    def show_4_directions(self):
        """
        Show screenshot with 4 cardinal direction arrows:
        UP, DOWN, LEFT, RIGHT
        """
        img = self._capture()
        
        # Define colors
        colors = {
            'UP': (0, 255, 0),      # Green
            'DOWN': (0, 0, 255),    # Red
            'LEFT': (255, 0, 0),    # Blue
            'RIGHT': (255, 255, 0)  # Cyan
        }
        
        center = (self.center_x, self.center_y)
        length = int(self.arrow_length)
        
        # Draw arrows
        # UP
        end_up = (self.center_x, self.center_y - length)
        self._draw_arrow(img, center, end_up, colors['UP'], 4)
        cv2.putText(img, "UP (0)", (end_up[0] - 30, end_up[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors['UP'], 2)
        
        # DOWN
        end_down = (self.center_x, self.center_y + length)
        self._draw_arrow(img, center, end_down, colors['DOWN'], 4)
        cv2.putText(img, "DOWN (1)", (end_down[0] - 40, end_down[1] + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors['DOWN'], 2)
        
        # LEFT
        end_left = (self.center_x - length, self.center_y)
        self._draw_arrow(img, center, end_left, colors['LEFT'], 4)
        cv2.putText(img, "LEFT (2)", (end_left[0] - 80, end_left[1] + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors['LEFT'], 2)
        
        # RIGHT
        end_right = (self.center_x + length, self.center_y)
        self._draw_arrow(img, center, end_right, colors['RIGHT'], 4)
        cv2.putText(img, "RIGHT (3)", (end_right[0] + 10, end_right[1] + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors['RIGHT'], 2)
        
        # Draw center circle (player position)
        cv2.circle(img, center, 10, (255, 255, 255), -1)
        cv2.circle(img, center, 12, (0, 0, 0), 2)
        
        # Title
        cv2.putText(img, "4 CARDINAL DIRECTIONS", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        return img
    
    def show_8_directions(self):
        """
        Show screenshot with 8 direction arrows:
        UP, DOWN, LEFT, RIGHT + 4 diagonals
        """
        img = self._capture()
        
        # Define colors for each direction
        colors = [
            (0, 255, 0),      # 0: UP - Green
            (0, 0, 255),      # 1: DOWN - Red
            (255, 0, 0),      # 2: LEFT - Blue
            (255, 255, 0),    # 3: RIGHT - Cyan
            (0, 255, 255),    # 4: UP-RIGHT - Yellow
            (255, 0, 255),    # 5: DOWN-RIGHT - Magenta
            (128, 255, 0),    # 6: UP-LEFT - Lime
            (0, 128, 255),    # 7: DOWN-LEFT - Orange
        ]
        
        labels = [
            "UP (0)", "DOWN (1)", "LEFT (2)", "RIGHT (3)",
            "UP-R (4)", "DOWN-R (5)", "UP-L (6)", "DOWN-L (7)"
        ]
        
        center = (self.center_x, self.center_y)
        length = int(self.arrow_length)
        diag = int(length * 0.707)  # 45 degree diagonal
        
        # Calculate end points for all 8 directions
        endpoints = [
            (self.center_x, self.center_y - length),      # UP
            (self.center_x, self.center_y + length),      # DOWN
            (self.center_x - length, self.center_y),      # LEFT
            (self.center_x + length, self.center_y),      # RIGHT
            (self.center_x + diag, self.center_y - diag), # UP-RIGHT
            (self.center_x + diag, self.center_y + diag), # DOWN-RIGHT
            (self.center_x - diag, self.center_y - diag), # UP-LEFT
            (self.center_x - diag, self.center_y + diag), # DOWN-LEFT
        ]
        
        # Label offsets
        label_offsets = [
            (-30, -15),   # UP
            (-40, 30),    # DOWN
            (-90, 5),     # LEFT
            (15, 5),      # RIGHT
            (15, -10),    # UP-RIGHT
            (15, 20),     # DOWN-RIGHT
            (-80, -10),   # UP-LEFT
            (-90, 20),    # DOWN-LEFT
        ]
        
        # Draw all arrows
        for i in range(8):
            self._draw_arrow(img, center, endpoints[i], colors[i], 3)
            label_pos = (endpoints[i][0] + label_offsets[i][0],
                        endpoints[i][1] + label_offsets[i][1])
            cv2.putText(img, labels[i], label_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)
        
        # Draw center circle
        cv2.circle(img, center, 10, (255, 255, 255), -1)
        cv2.circle(img, center, 12, (0, 0, 0), 2)
        
        # Title
        cv2.putText(img, "8 DIRECTIONS (with diagonals)", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        return img
    
    def show_food_arrows(self):
        """
        Show screenshot with arrows pointing to all detected food centroids
        """
        img = self._capture()
        food = self._find_food(img)
        
        center = (self.center_x, self.center_y)
        
        # Draw center crosshair (player position)
        cv2.line(img, (self.center_x - 20, self.center_y),
                (self.center_x + 20, self.center_y), (255, 255, 255), 2)
        cv2.line(img, (self.center_x, self.center_y - 20),
                (self.center_x, self.center_y + 20), (255, 255, 255), 2)
        cv2.circle(img, center, 8, (255, 255, 255), -1)
        
        # Color gradient from green (nearest) to red (farthest)
        for i, f in enumerate(food[:30]):  # Show up to 30 food items
            # Color: green for nearest, red for farthest
            ratio = min(i / 30.0, 1.0)
            color = (0, int(255 * (1 - ratio)), int(255 * ratio))  # BGR: green to red
            
            food_pos = (f['x'], f['y'])
            
            # Draw arrow from center to food
            self._draw_arrow(img, center, food_pos, color, 1, tip_length=0.2)
            
            # Draw circle at food location
            cv2.circle(img, food_pos, max(3, f['radius']), color, 2)
            
            # Label nearest 5 with numbers
            if i < 5:
                cv2.putText(img, f"{i+1}", (f['x'] + 5, f['y'] - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Title and stats
        cv2.putText(img, f"FOOD ARROWS - {len(food)} detected", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Legend
        cv2.putText(img, "Green = Nearest, Red = Farthest", (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Show nearest food info
        if food:
            dx = food[0]['x'] - self.center_x
            dy = food[0]['y'] - self.center_y
            cv2.putText(img, f"Nearest: dx={dx:+4d}, dy={dy:+4d}, dist={food[0]['dist']:.0f}",
                       (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return img


def main():
    """Main loop - shows all three visualizations every 5 seconds and saves PNGs"""
    
    # Output directory for saved images
    output_dir = "/Users/johnvincentsalvi/Downloads/RandomCodingFiles/agario"
    
    print("=" * 60)
    print("Action Type Visualizer for Agar.io")
    print("=" * 60)
    print("Showing 3 action visualizations every 5 seconds:")
    print("  1. 4 Cardinal Directions (UP, DOWN, LEFT, RIGHT)")
    print("  2. 8 Directions (with diagonals)")
    print("  3. Food Arrows (to all detected food)")
    print()
    print(f"Saving PNGs to: {output_dir}")
    print("Press 'q' on any window to quit")
    print("=" * 60)
    
    viz = ActionVisualizer()
    
    iteration = 0
    
    try:
        while True:
            iteration += 1
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            print(f"\n--- Iteration {iteration} ({timestamp}) ---")
            
            # 1. Show and save 4 cardinal directions
            img_4dir = viz.show_4_directions()
            cv2.imshow("1. Four Cardinal Directions", img_4dir)
            cv2.moveWindow("1. Four Cardinal Directions", 50, 50)
            path_4dir = f"{output_dir}/action_4dir_{timestamp}.png"
            cv2.imwrite(path_4dir, img_4dir)
            print(f"  Saved: {path_4dir}")
            
            # 2. Show and save 8 directions
            img_8dir = viz.show_8_directions()
            cv2.imshow("2. Eight Directions", img_8dir)
            cv2.moveWindow("2. Eight Directions", 100, 100)
            path_8dir = f"{output_dir}/action_8dir_{timestamp}.png"
            cv2.imwrite(path_8dir, img_8dir)
            print(f"  Saved: {path_8dir}")
            
            # 3. Show and save food arrows
            img_food = viz.show_food_arrows()
            cv2.imshow("3. Food Arrows", img_food)
            cv2.moveWindow("3. Food Arrows", 150, 150)
            path_food = f"{output_dir}/action_food_{timestamp}.png"
            cv2.imwrite(path_food, img_food)
            print(f"  Saved: {path_food}")
            
            # Wait for 5 seconds, but check for 'q' press frequently
            for _ in range(50):  # 50 * 100ms = 5 seconds
                key = cv2.waitKey(100)
                if key == ord('q') or key == ord('Q'):
                    print("\n'q' pressed - Exiting...")
                    cv2.destroyAllWindows()
                    return
            
            print(f"  (Next update in 5 seconds...)")
    
    except KeyboardInterrupt:
        print("\nKeyboard interrupt - Exiting...")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

