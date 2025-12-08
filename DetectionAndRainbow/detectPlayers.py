import cv2
import mss
import numpy as np


class PlayerDetector:
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
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 30, 30]), np.array([180, 255, 255]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        blobs = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > 50:
                (x, y), radius = cv2.minEnclosingCircle(c)
                blobs.append({'x': int(x), 'y': int(y), 'radius': int(radius), 'contour': c})
        return blobs
    
    def _find_self(self, blobs, img_width, img_height):
        """
        Find the player blob at screen center.
        The player is ALWAYS at the center of the screen in Agar.io.
        Returns the blob closest to center with radius > 10.
        """
        center_x, center_y = img_width // 2, img_height // 2
        
        # Find all blobs near center (within 15% of screen dimensions)
        # Use smaller minimum radius (10) to detect small starting players
        candidates = []
        for b in blobs:
            if b['radius'] >= 10:  # Minimum detectable size
                dist_to_center = ((b['x'] - center_x)**2 + (b['y'] - center_y)**2)**0.5
                # Must be within 15% of center
                if dist_to_center < min(img_width, img_height) * 0.15:
                    candidates.append((b, dist_to_center))
        
        if not candidates:
            return None
        
        # Return the blob closest to exact center
        # (in case of tie, prefer larger radius)
        candidates.sort(key=lambda x: (x[1], -x[0]['radius']))
        return candidates[0][0]
    
    def _classify_players(self, blobs, img_width, img_height, img=None):
        center_x, center_y = img_width // 2, img_height // 2
        
        # Virus color #12F448 = BGR(72, 244, 18)
        virus_b, virus_g, virus_r = 72, 244, 18
        virus_tolerance = 40
        
        # Find self first
        self_blob = self._find_self(blobs, img_width, img_height)
        
        # Find other players (radius >= 30 to be a player, not self, not virus)
        players = []
        for b in blobs:
            if b['radius'] >= 30 and b != self_blob:
                # Check if virus (green #12F448) using the already captured image
                if img is not None and 0 <= b['y'] < img_height and 0 <= b['x'] < img_width:
                    pixel = img[b['y'], b['x']]
                    is_virus = (abs(int(pixel[0]) - virus_b) < virus_tolerance and 
                               abs(int(pixel[1]) - virus_g) < virus_tolerance and 
                               abs(int(pixel[2]) - virus_r) < virus_tolerance)
                    if is_virus:
                        continue
                players.append(b)
        
        # Sort by distance to center and get 3 nearest
        players.sort(key=lambda p: ((p['x'] - center_x)**2 + (p['y'] - center_y)**2)**0.5)
        return players[:3], self_blob
    
    def getPlayers(self):
        """Returns list of 3 nearest players as (x, y, radius) tuples - ABSOLUTE coords"""
        img = self._capture()
        blobs = self._find_blobs(img)
        players, _ = self._classify_players(blobs, img.shape[1], img.shape[0], img)
        return [(p['x'], p['y'], p['radius']) for p in players]
    
    def getPlayersAndSelf(self):
        """
        Returns (players, self_radius) where:
        - players: list of 3 nearest players as (x, y, radius) tuples - ABSOLUTE coords
        - self_radius: detected radius of self blob (or None if not detected)
        """
        img = self._capture()
        blobs = self._find_blobs(img)
        players, self_blob = self._classify_players(blobs, img.shape[1], img.shape[0], img)
        self_radius = self_blob['radius'] if self_blob else None
        return [(p['x'], p['y'], p['radius']) for p in players], self_radius
    
    def getPlayersRelative(self):
        """
        Returns list of 3 nearest players as (dx, dy, radius) tuples - RELATIVE to player.
        
        dx > 0 = player is to the RIGHT
        dx < 0 = player is to the LEFT
        dy > 0 = player is BELOW
        dy < 0 = player is ABOVE
        
        Also returns self_radius as second return value.
        """
        img = self._capture()
        center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
        blobs = self._find_blobs(img)
        players, self_blob = self._classify_players(blobs, img.shape[1], img.shape[0], img)
        
        result = []
        for p in players:
            dx = p['x'] - center_x
            dy = p['y'] - center_y
            result.append((dx, dy, p['radius']))
        
        self_radius = self_blob['radius'] if self_blob else None
        return result, self_radius
    
    def getSelfRadius(self):
        """
        Returns the radius of your player blob at screen center.
        Returns None if not detected.
        
        Your player is always at the center of the screen in Agar.io.
        """
        img = self._capture()
        blobs = self._find_blobs(img)
        self_blob = self._find_self(blobs, img.shape[1], img.shape[0])
        return self_blob['radius'] if self_blob else None
    
    def showPlayers(self, save_png=True):
        """Display debug overlay showing detected players and self"""
        img = self._capture()
        blobs = self._find_blobs(img)
        players, self_blob = self._classify_players(blobs, img.shape[1], img.shape[0], img)
        
        # Draw self (green)
        if self_blob:
            cv2.circle(img, (self_blob['x'], self_blob['y']), self_blob['radius'], (0, 255, 0), 3)
            cv2.circle(img, (self_blob['x'], self_blob['y']), 3, (0, 255, 0), -1)  # Center dot
            cv2.putText(img, f"SELF r={self_blob['radius']}", (self_blob['x']-40, self_blob['y']-self_blob['radius']-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw players (yellow)
        for i, p in enumerate(players):
            cv2.circle(img, (p['x'], p['y']), p['radius'], (0, 255, 255), 2)
            cv2.circle(img, (p['x'], p['y']), 3, (0, 255, 255), -1)  # Center dot
            cv2.putText(img, f"P{i+1} r={p['radius']}", (p['x']-20, p['y']-p['radius']-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        cv2.imshow("Players", img)
        cv2.moveWindow("Players", 200, 200)
        cv2.waitKey(1)
        
        # Save PNG
        if save_png:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            path = f"/Users/johnvincentsalvi/Downloads/RandomCodingFiles/agario/detect_players_{timestamp}.png"
            cv2.imwrite(path, img)
            print(f"  Saved: {path}")
        
        return players


# Test loop
if __name__ == "__main__":
    import time
    
    detector = PlayerDetector()
    print("Starting player detection... (Ctrl+C to stop)")
    
    while True:
        detector.showPlayers()
        players, self_radius = detector.getPlayersAndSelf()
        print(f"Self radius: {self_radius} | Players: {len(players)} - {players}")
        time.sleep(0.5)

