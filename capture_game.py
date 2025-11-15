# Required libraries
import numpy as np
import cv2
from mss import mss
from PIL import Image
import time
import pytesseract


# Step 1: Screenshot function
def capture_game_screen(monitor_region):
    """Capture a screenshot of the game area"""
    with mss() as sct:
        screenshot = sct.grab(monitor_region)
        # Convert to numpy array for OpenCV
        img = np.array(screenshot)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

# Step 2: Process image to isolate blobs
def create_blob_mask(img):
    """
    Create a binary mask where blobs are white, background is black
    """
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Agar.io blobs are colored and bright
    # The background is typically darker
    # We'll detect anything with sufficient saturation and value
    lower_bound = np.array([0, 30, 30])  # Adjust these values
    upper_bound = np.array([180, 255, 255])
    
    # Create mask
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Clean up noise
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

# Step 3: Find blob positions
# Step 3: Find blob positions
def find_blobs(mask):
    """
    Find all blobs and their properties
    Returns list of (x, y, radius) tuples AND contours
    """
    # Find contours (blob outlines)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)
    
    blobs = []
    for contour in contours:
        # Calculate the center and radius
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Estimate radius from area
            area = cv2.contourArea(contour)
            radius = int(np.sqrt(area / np.pi))
            
            blobs.append((cx, cy, radius))
    
    return blobs, contours  # Return both blobs and contours

# Step 3.5: Classify blobs
def classify_blobs(blobs, img, img_width, img_height, virus_color_bgr):
    """
    Classify blobs into food, self, viruses, and other players
    New logic:
    1. Find self: closest to center with radius >= 24
    2. Find viruses: by color match
    3. Food: radius < 14 (excluding self and viruses)
    4. Other players: everything else
    
    virus_color_bgr: tuple of (B, G, R) values for virus color
    Returns a dictionary with classified blobs
    """
    classified = {
        'food': [],
        'self': None,
        'viruses': [],
        'other_players': []
    }
    
    # Center of the screen
    center_x = img_width / 2
    center_y = img_height / 2
    
    # Create list of all blobs with their info
    all_blobs = []
    for i, (x, y, radius) in enumerate(blobs):
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        all_blobs.append({
            'x': x,
            'y': y,
            'radius': radius,
            'distance': distance,
            'index': i
        })
    
    # Step 1: Find SELF - closest to center with radius >= 24
    eligible_for_self = [b for b in all_blobs if b['radius'] >= 24]
    if eligible_for_self:
        eligible_for_self.sort(key=lambda b: b['distance'])
        self_blob = eligible_for_self[0]
        classified['self'] = {
            'x': self_blob['x'],
            'y': self_blob['y'],
            'radius': self_blob['radius']
        }
        self_index = self_blob['index']
    else:
        self_index = -1  # No self found
    
    # Step 2 & 3 & 4: Classify remaining blobs
    for blob in all_blobs:
        # Skip if this is self
        if blob['index'] == self_index:
            continue
        
        # Step 2: Check if virus by color
        if is_virus_color(img, blob['x'], blob['y'], virus_color_bgr):
            classified['viruses'].append({
                'x': blob['x'],
                'y': blob['y'],
                'radius': blob['radius']
            })
        # Step 3: Check if food (small radius)
        elif blob['radius'] < 18:
            classified['food'].append({
                'x': blob['x'],
                'y': blob['y'],
                'radius': blob['radius']
            })
        # Step 4: Everything else is other players
        else:
            classified['other_players'].append({
                'x': blob['x'],
                'y': blob['y'],
                'radius': blob['radius']
            })
    
    return classified

def is_virus_color(img, x, y, target_color_bgr, tolerance=5):
    """
    Check if the color at position (x, y) matches the target virus color
    
    target_color_bgr: tuple of (B, G, R) values
    tolerance: how much deviation is allowed (default 30)
    """
    # Make sure coordinates are within image bounds
    height, width = img.shape[:2]
    if x < 0 or x >= width or y < 0 or y >= height:
        return False
    
    # Get the color at the blob center
    pixel_color = img[y, x]  # Note: img is indexed as [row, col] = [y, x]
    
    # Check if color is within tolerance
    b_match = abs(int(pixel_color[0]) - target_color_bgr[0]) <= tolerance
    g_match = abs(int(pixel_color[1]) - target_color_bgr[1]) <= tolerance
    r_match = abs(int(pixel_color[2]) - target_color_bgr[2]) <= tolerance
    
    return b_match and g_match and r_match

def extract_score(img_full, debug=False, iteration=None):
    """
    Extract score from fixed score counter region
    Score region: screen coords (55, 772) to (90, 787)
    In captured image: (55, 647) to (90, 662) since capture starts at y=125
    Returns: score as integer, or None if failed
    """
    # Convert screen coordinates to captured image coordinates
    score_y_start = 772 - 125  # 647
    score_y_end = 787 - 125    # 662
    score_x_start = 55
    score_x_end = 90
    
    # Crop the score region
    score_region = img_full[score_y_start:score_y_end, score_x_start:score_x_end]
    
    # Convert to grayscale
    gray = cv2.cvtColor(score_region, cv2.COLOR_BGR2GRAY)
    
    # Resize larger for better OCR (4x scale)
    scale = 4
    height, width = gray.shape
    resized = cv2.resize(gray, (width * scale, height * scale),
                        interpolation=cv2.INTER_CUBIC)
    
    # Tesseract config for digits only
    custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
    
    # OCR on grayscale
    text = pytesseract.image_to_string(resized, config=custom_config).strip()
    
    if debug:
        print(f"\n[DEBUG] Score Extraction:")
        print(f"  Region shape: {score_region.shape}")
        print(f"  OCR text: '{text}'")
        
        # Save debug images
        suffix = f"_iter{iteration}" if iteration is not None else ""
        cv2.imwrite(f'debug_score_region{suffix}.png', score_region)
        cv2.imwrite(f'debug_score_resized{suffix}.png', resized)
        print(f"  Saved debug images: debug_score_*{suffix}.png")
    
    # Try to parse as integer
    try:
        score = int(text)
        if debug:
            print(f"  ✓ Score: {score}")
        return score
    except ValueError:
        if debug:
            print(f"  ✗ Failed to parse score")
        return None

def detect_game_end(img_full, template_path='game-end.png'):
    """
    Detect if the game end screen is showing using template matching
    Region: screen coords (944, 126) to (1469, 157)
    In captured image: (944, 1) to (1469, 32) since capture starts at y=125
    Returns: True if template matches, False otherwise
    """
    # Convert screen coordinates to captured image coordinates
    region_y_start = 126 - 125  # 1
    region_y_end = 157 - 125    # 32
    region_x_start = 944
    region_x_end = 1469
    
    # Crop the region
    region = img_full[region_y_start:region_y_end, region_x_start:region_x_end]
    
    # Load the template image
    template = cv2.imread(template_path)
    if template is None:
        print(f"Error: Could not load template image '{template_path}'")
        return False
    
    # Convert both to grayscale for more robust matching
    region_gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    # Check if template and region have same dimensions
    if template_gray.shape != region_gray.shape:
        # Resize template to match region size
        template_gray = cv2.resize(template_gray, (region_gray.shape[1], region_gray.shape[0]))
    
    # Calculate similarity using normalized cross-correlation
    result = cv2.matchTemplate(region_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    # Threshold for match (0.4 = 40% similarity)
    threshold = 0.4
    is_match = max_val >= threshold
    
    return is_match

def get_game_state(classified, game_ended):
    """
    Extract game state into a structured format
    Returns a dictionary with game state information
    """
    game_state = {
        'game_ended': game_ended,  # Boolean: True if game end screen detected
        'self_position': None,
        'viruses': [],
        'other_players': [],
        'food_count': 0
    }
    
    # Self position
    if classified['self']:
        game_state['self_position'] = (classified['self']['x'], 
                                       classified['self']['y'], 
                                       classified['self']['radius'])
    
    # Virus positions
    for virus in classified['viruses']:
        game_state['viruses'].append((virus['x'], virus['y'], virus['radius']))
    
    # Other player positions
    for player in classified['other_players']:
        game_state['other_players'].append((player['x'], player['y'], player['radius']))
    
    # Food count
    game_state['food_count'] = len(classified['food'])
    
    return game_state

# Step 4: Main loop
def main():
    # Game window region coordinates
    monitor = {"top": 125, "left": 0, "width": 1470, "height": 671}
    
    # VIRUS COLOR - Update these BGR values to match virus color in your game
    # To find the color: take a screenshot, open in image editor, and check RGB values
    # Then convert to BGR (swap R and B)
    # Example: if virus is green (RGB: 0, 255, 0), then BGR is (0, 255, 0)
    VIRUS_COLOR_BGR = (90, 251, 125)  # Replace with actual virus color!
    
    print("=" * 80)
    print("STARTING GAME STATE MONITORING")
    print("=" * 80)
    print(f"Running 20 iterations with 2-second intervals...")
    print("=" * 80)
    
    # Run loop 20 times
    for iteration in range(1, 21):
        print(f"\n{'='*80}")
        print(f"ITERATION {iteration}/20")
        print(f"{'='*80}")
        
        # Capture full screenshot (includes game area + score area)
        img_full = capture_game_screen(monitor)
        
        # Split image: game area (top) and score area (bottom)
        # Game area: from y=0 to y=(762-125)=637 in captured image coordinates
        game_area_height = 762 - 125  # 637 pixels
        img_game = img_full[:game_area_height, :]  # Game area for blob detection
        img_score = img_full[game_area_height:, :]  # Score area for later use
        
        # Get dimensions of game area
        img_height, img_width = img_game.shape[:2]
        
        # Create mask (only on game area)
        mask = create_blob_mask(img_game)
        
        # Find blobs (only in game area)
        blobs, contours = find_blobs(mask)
        
        # Classify blobs (only in game area)
        # Classify blobs (only in game area)
        classified = classify_blobs(blobs, img_game, img_width, img_height, VIRUS_COLOR_BGR)
        
        # Detect if game has ended
        game_ended = detect_game_end(img_full)
        
        # Find and sum all numbers on screen
        score = extract_score(img_full, debug=True, iteration=iteration)
        
        # Get game state (with game end detection)
        game_state = get_game_state(classified, game_ended)
        
        # Add score to game state
        game_state['score'] = score
        
        # Print game state
        print(f"\nGame Ended: {game_state['game_ended']}")
        print(f"Score: {game_state['score']}")
        print(f"Self Position: {game_state['self_position']}")
        print(f"  Format: (x, y, radius)")
        
        print(f"\nViruses ({len(game_state['viruses'])} detected):")
        for i, virus in enumerate(game_state['viruses']):
            print(f"  Virus {i+1}: {virus}")
        
        print(f"\nOther Players ({len(game_state['other_players'])} detected):")
        for i, player in enumerate(game_state['other_players']):
            print(f"  Player {i+1}: {player}")
        
        print(f"\nFood Count: {game_state['food_count']}")
        
        # Wait 2 seconds before next iteration (except on last iteration)
        if iteration < 20:
            print(f"\nWaiting 2 seconds...")
            time.sleep(2)
    
    print(f"\n{'='*80}")
    print("MONITORING COMPLETE - 20 iterations finished")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()