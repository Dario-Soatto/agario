import pyautogui
import time
from mss import mss

def find_window_coordinates():
    """
    Interactive script to help find game window coordinates.
    """
    print("=" * 60)
    print("WINDOW COORDINATE FINDER")
    print("=" * 60)
    
    # First, show available monitors
    print("\nAvailable monitors:")
    with mss() as sct:
        for i, monitor in enumerate(sct.monitors):
            print(f"Monitor {i}: {monitor}")
    
    print("\n" + "=" * 60)
    print("INSTRUCTIONS:")
    print("1. Move your mouse to the TOP-LEFT corner of your game window")
    print("2. Note the X and Y coordinates shown below")
    print("3. Move your mouse to the BOTTOM-RIGHT corner")
    print("4. Note those coordinates too")
    print("5. Press Ctrl+C to stop")
    print("=" * 60)
    print("\nCurrent mouse position:")
    print("(Move your mouse to see live coordinates...)\n")
    
    try:
        last_x, last_y = -1, -1
        while True:
            # Get current mouse position
            x, y = pyautogui.position()
            
            # Only print if position changed (to avoid spam)
            if x != last_x or y != last_y:
                # Clear line and print current position
                print(f"\rX: {x:4d}  Y: {y:4d}  ", end='', flush=True)
                last_x, last_y = x, y
            
            time.sleep(0.1)  # Check 10 times per second
            
    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("CALCULATING YOUR MONITOR REGION...")
        print("=" * 60)
        
        # Ask for the coordinates
        print("\nEnter the coordinates you noted:")
        try:
            top_left_x = int(input("Top-left X: "))
            top_left_y = int(input("Top-left Y: "))
            bottom_right_x = int(input("Bottom-right X: "))
            bottom_right_y = int(input("Bottom-right Y: "))
            
            # Calculate width and height
            width = bottom_right_x - top_left_x
            height = bottom_right_y - top_left_y
            
            print("\n" + "=" * 60)
            print("YOUR MONITOR REGION:")
            print("=" * 60)
            print(f'\nmonitor = {{"top": {top_left_y}, "left": {top_left_x}, "width": {width}, "height": {height}}}')
            print("\nCopy the line above into your capture_game.py file!")
            print("=" * 60)
            
        except (ValueError, EOFError):
            print("\nNo coordinates entered. Exiting...")

if __name__ == "__main__":
    find_window_coordinates()