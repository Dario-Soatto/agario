import cv2
import mss
import numpy as np


class FoodDetector:
    """Detects food pellets (small blobs with radius < 60)"""
    
    def __init__(self):
        with mss.mss() as sct:
            if len(sct.monitors) > 2:
                ext = sct.monitors[2]
                padding_x, padding_y = int(ext['width'] * 0.05), int(ext['height'] * 0.05)
                self.region = {"top": ext['top'] + padding_y, "left": ext['left'] + padding_x,
                               "width": ext['width'] - 2*padding_x, "height": ext['height'] - 2*padding_y}
            else:
                mon = sct.monitors[1]
                padding_x, padding_y = int(mon['width'] * 0.05), int(mon['height'] * 0.05)
                self.region = {"top": padding_y, "left": padding_x,
                               "width": mon['width'] - 2*padding_x, "height": mon['height'] - 2*padding_y}
    
    def _capture(self):
        with mss.mss() as sct:
            img = np.array(sct.grab(self.region))
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    def _find_blobs(self, img):
        """Find all colored blobs in the image"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Detect saturated colors (food pellets are colorful)
        mask = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([180, 255, 255]))
        # Smaller kernel for food (they're tiny)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        blobs = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > 10:  # Lower threshold for small food
                (x, y), radius = cv2.minEnclosingCircle(c)
                blobs.append({'x': int(x), 'y': int(y), 'radius': int(radius), 'area': area})
        return blobs
    
    def _classify_food(self, blobs, img_width, img_height):
        """
        Classify blobs as food (radius < 60, area small)
        Returns: list of food blobs sorted by distance to center
        """
        center_x, center_y = img_width // 2, img_height // 2
        
        # Food is small (radius < 60) and not too tiny (radius > 3)
        food = [b for b in blobs if 3 < b['radius'] < 60]
        
        # Sort by distance to screen center (where player usually is)
        food.sort(key=lambda f: ((f['x'] - center_x)**2 + (f['y'] - center_y)**2)**0.5)
        
        return food
    
    def getFood(self, max_count=10):
        """
        Returns list of nearest food pellets as (x, y, radius) tuples - ABSOLUTE coords
        
        Args:
            max_count: Maximum number of food items to return (default 10)
        """
        img = self._capture()
        blobs = self._find_blobs(img)
        food = self._classify_food(blobs, img.shape[1], img.shape[0])
        return [(f['x'], f['y'], f['radius']) for f in food[:max_count]]
    
    def getFoodRelative(self, max_count=7):
        """
        Returns list of 7 nearest food pellets as (dx, dy) tuples - RELATIVE to player.
        
        dx > 0 = food is to the RIGHT
        dx < 0 = food is to the LEFT
        dy > 0 = food is BELOW
        dy < 0 = food is ABOVE
        
        Args:
            max_count: Maximum number of food items to return (default 7)
        """
        img = self._capture()
        center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
        blobs = self._find_blobs(img)
        food = self._classify_food(blobs, img.shape[1], img.shape[0])
        
        result = []
        for f in food[:max_count]:
            dx = f['x'] - center_x
            dy = f['y'] - center_y
            result.append((dx, dy))
        
        return result
    
    def getFoodCount(self):
        """Returns total count of visible food pellets"""
        img = self._capture()
        blobs = self._find_blobs(img)
        food = self._classify_food(blobs, img.shape[1], img.shape[0])
        return len(food)
    
    def showFood(self, save_png=True):
        """Display debug overlay showing detected food"""
        img = self._capture()
        blobs = self._find_blobs(img)
        food = self._classify_food(blobs, img.shape[1], img.shape[0])
        
        # Draw screen center crosshair
        h, w = img.shape[:2]
        cv2.line(img, (w//2 - 20, h//2), (w//2 + 20, h//2), (255, 255, 255), 1)
        cv2.line(img, (w//2, h//2 - 20), (w//2, h//2 + 20), (255, 255, 255), 1)
        
        # Draw food (cyan circles)
        for i, f in enumerate(food[:20]):  # Show up to 20 food items
            color = (255, 255, 0)  # Cyan
            cv2.circle(img, (f['x'], f['y']), f['radius'], color, 1)
            cv2.circle(img, (f['x'], f['y']), 2, color, -1)  # Center dot
            
            # Label nearest 5 with numbers
            if i < 5:
                cv2.putText(img, f"{i+1}", (f['x']+5, f['y']-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Stats
        cv2.putText(img, f"Food count: {len(food)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow("Food Detection", img)
        cv2.moveWindow("Food Detection", 200, 200)
        cv2.waitKey(1)
        
        # Save PNG
        if save_png:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            path = f"/Users/johnvincentsalvi/Downloads/RandomCodingFiles/agario/detect_food_{timestamp}.png"
            cv2.imwrite(path, img)
            print(f"  Saved: {path}")
        
        return food


# Test loop
if __name__ == "__main__":
    import time
    
    detector = FoodDetector()
    print("Starting food detection... (Ctrl+C to stop)")
    print(f"Capture region: {detector.region}")
    
    while True:
        food = detector.showFood()
        nearest_5 = detector.getFood(5)
        print(f"Food visible: {len(food)} | Nearest 5: {nearest_5}")
        time.sleep(0.5)

