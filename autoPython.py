import pyautogui
import time
import mss
import numpy as np
import cv2
import signal
import sys
from pynput import keyboard
import threading
from takeOnScreenAction import ScreenAction

height_of_screen = pyautogui.size().height
width_of_screen = pyautogui.size().width

eight_degree_movement = ScreenAction(width_of_screen, height_of_screen)


eight_degree_movement.move_diagonal_up_right()
time.sleep(1)
eight_degree_movement.move_diagonal_down_right()
time.sleep(1)
eight_degree_movement.move_diagonal_up_left()
time.sleep(1)
eight_degree_movement.move_diagonal_down_left()
time.sleep(1)
eight_degree_movement.move_right()
time.sleep(1)
eight_degree_movement.move_left()
time.sleep(1)
eight_degree_movement.move_up()
time.sleep(1)
eight_degree_movement.move_down()
time.sleep(1)



# Global flag to track if 'q' is pressed
q_pressed = False

def on_press(key):
    global q_pressed
    try:
        if key.char == 'q':
            q_pressed = True
    except AttributeError:
        pass

def signal_handler(sig, frame):
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

time.sleep(5)
multipleScreen = mss.mss()
for index, monitor in enumerate(multipleScreen.monitors):
    print(index, monitor)

# Try to use monitor 2 (second physical monitor) if it exists, otherwise default to monitor 1
if len(multipleScreen.monitors) > 2:
    monitor = multipleScreen.monitors[2]
    print(f"Using monitor 2 (second physical monitor)")
else:
    monitor = multipleScreen.monitors[1]
    print(f"Using monitor 1 (first physical monitor - monitor 2 not available)")
width, height = monitor['width'], monitor['height']
center_x, center_y = width // 2, height // 2
cushion = 200
min_area = 50  # Minimum area for food particles
max_area = 2000  # Maximum area for food particles (larger = players to avoid)
# Adjust max_area if needed: smaller value = more strict (only very small food),
# larger value = more lenient (includes larger food but might include small players)
lowerColor = np.array([0, 50, 50])
upperColor = np.array([180, 255, 255])

# Enable failsafe: move mouse to top-left corner to stop the script
pyautogui.FAILSAFE = True
print("\n" + "="*60)
print("CONTROLS:")
print("  - Press 'q' key to stop the script")
print("  - Press ESC key in the display window to stop")
print("  - Move mouse to top-left corner to stop (failsafe)")
print("="*60 + "\n")

# Start keyboard listener in background thread
listener = keyboard.Listener(on_press=on_press)
listener.start()

# Frame counter for displaying every 5th mask
frame_count = 0

# Store the last mask image to keep it visible between updates
last_mask_display = None

# Mask display window size (smaller window)
mask_window_size = 400

# Create a display window for the mask
cv2.namedWindow('Mask Display', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Mask Display', mask_window_size, mask_window_size)

# Get screen dimensions for positioning (will position after first frame)
screen_width = pyautogui.size().width
screen_height = pyautogui.size().height
window_x = screen_width - mask_window_size - 10  # 10 pixels from right edge
window_y = 10  # 10 pixels from top
window_positioned = False

try:
    while True:
        # Check if 'q' key is pressed to stop
        if q_pressed:
            print("\n" + "="*60)
            print("Script stopped by user (pressed 'q')")
            print("="*60)
            listener.stop()
            cv2.destroyAllWindows()
            sys.exit(0)
        
        screenshot = multipleScreen.grab(monitor)
        screenshotArray = np.array(screenshot)
        bgrImage = screenshotArray[:, :, :3].copy()
        
        hsvImage = cv2.cvtColor(bgrImage, cv2.COLOR_BGR2HSV)
        colorMask = cv2.inRange(hsvImage, lowerColor, upperColor)
        
        frame_count += 1
        
        # Display every 5th mask image in the top right corner
        if frame_count % 5 == 0:
            # Resize mask to fit the display window
            mask_resized = cv2.resize(colorMask, (mask_window_size, mask_window_size))
            # Convert grayscale mask to BGR for display
            mask_bgr = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
            
            # Store the mask to keep it visible between updates
            last_mask_display = mask_bgr
        
        # Display the mask image if available (always update window to keep it responsive)
        if last_mask_display is not None:
            cv2.imshow('Mask Display', last_mask_display)
        else:
            # Show a black image until first mask is captured
            black_image = np.zeros((mask_window_size, mask_window_size, 3), dtype=np.uint8)
            cv2.imshow('Mask Display', black_image)
        
        # Position window in top right corner after first display
        if not window_positioned:
            cv2.moveWindow('Mask Display', window_x, window_y)
            window_positioned = True
        
        # Handle window close event (ESC key or window X button)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            print("\n" + "="*60)
            print("Script stopped by user (pressed ESC)")
            print("="*60)
            listener.stop()
            cv2.destroyAllWindows()
            sys.exit(0)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(colorMask, connectivity=8)
        
        # Identify players (large blobs to avoid)
        player_positions = [(int(centroids[i][0]), int(centroids[i][1])) 
                           for i in range(1, num_labels)
                           if stats[i, cv2.CC_STAT_AREA] > max_area  # Large blobs are players
                           and cushion <= centroids[i][0] <= width - cushion 
                           and cushion <= centroids[i][1] <= height - cushion]
        
        # Identify food particles (small blobs)
        blob_centers = [(int(centroids[i][0]), int(centroids[i][1])) 
                        for i in range(1, num_labels)
                        if min_area <= stats[i, cv2.CC_STAT_AREA] <= max_area  # Only small blobs (food), exclude large blobs (players)
                        and cushion <= centroids[i][0] <= width - cushion 
                        and cushion <= centroids[i][1] <= height - cushion]
        
        if len(blob_centers) >= 2:
            # Calculate distance from each food to nearest player, prioritize food away from players
            food_scores = []
            for x, y in blob_centers:
                # Distance to center (original logic)
                dist_to_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                
                # Distance to nearest player (want this to be large)
                if len(player_positions) > 0:
                    min_dist_to_player = min([np.sqrt((x - px)**2 + (y - py)**2) for px, py in player_positions])
                else:
                    min_dist_to_player = float('inf')  # No players, all food is safe
                
                # Combine: prioritize food far from players, secondary: close to center
                # Higher score = better (farther from players is better)
                food_scores.append((min_dist_to_player, -dist_to_center, x, y))
            
            # Sort by distance from players (descending), then by distance to center (ascending)
            food_scores.sort(reverse=True)
            _, _, target_x, target_y = food_scores[0]  # Best food (farthest from players)
            
            pyautogui.moveTo(monitor['left'] + target_x, monitor['top'] + target_y, duration=0.05)
        
        time.sleep(0.05)
except pyautogui.FailSafeException:
    print("\n" + "="*60)
    print("FAILSAFE TRIGGERED: Mouse moved to top-left corner")
    print("Script stopped safely")
    print("="*60)
    listener.stop()
    cv2.destroyAllWindows()
    sys.exit(0)
finally:
    listener.stop()
    cv2.destroyAllWindows()

