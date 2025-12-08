import cv2
import mss
import numpy as np


class VirusDetector:
    """Detects viruses (green spiky blobs with color #12F448)"""
    
    # #12F448 in BGR = (72, 244, 18)
    VIRUS_BGR = (72, 244, 18)
    COLOR_TOLERANCE = 40  # Allow some variance
    
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
        
        self.center_x = self.region['width'] // 2
        self.center_y = self.region['height'] // 2
    
    def _capture(self):
        with mss.mss() as sct:
            img = np.array(sct.grab(self.region))
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    def _find_viruses(self, img):
        """Find blobs matching virus color #12F448"""
        # Create mask for virus green color in BGR
        # #12F448 = RGB(18, 244, 72) = BGR(72, 244, 18)
        lower = np.array([
            max(0, self.VIRUS_BGR[0] - self.COLOR_TOLERANCE),
            max(0, self.VIRUS_BGR[1] - self.COLOR_TOLERANCE),
            max(0, self.VIRUS_BGR[2] - self.COLOR_TOLERANCE)
        ])
        upper = np.array([
            min(255, self.VIRUS_BGR[0] + self.COLOR_TOLERANCE),
            min(255, self.VIRUS_BGR[1] + self.COLOR_TOLERANCE),
            min(255, self.VIRUS_BGR[2] + self.COLOR_TOLERANCE)
        ])
        
        mask = cv2.inRange(img, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        viruses = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > 500:  # Viruses are fairly large
                (x, y), radius = cv2.minEnclosingCircle(c)
                if radius > 30:  # Minimum virus size
                    viruses.append({'x': int(x), 'y': int(y), 'radius': int(radius)})
        
        return viruses
    
    def getViruses(self):
        """Returns list of all viruses as (x, y, radius) tuples, sorted by distance"""
        img = self._capture()
        viruses = self._find_viruses(img)
        
        # Sort by distance to center (player)
        viruses.sort(key=lambda v: ((v['x'] - self.center_x)**2 + (v['y'] - self.center_y)**2)**0.5)
        
        return [(v['x'], v['y'], v['radius']) for v in viruses]
    
    def getNearestVirus(self):
        """
        Returns (dx, dy) of the nearest virus relative to player (screen center).
        Returns None if no virus detected.
        
        dx: positive = virus is to the RIGHT, negative = LEFT
        dy: positive = virus is BELOW, negative = ABOVE
        """
        viruses = self.getViruses()
        
        if not viruses:
            return None
        
        vx, vy, _ = viruses[0]  # Nearest virus
        dx = vx - self.center_x
        dy = vy - self.center_y
        
        return (dx, dy)
    
    def getNearestVirusNormalized(self):
        """
        Returns (dx, dy) normalized to [-1, 1] range.
        Returns None if no virus detected.
        """
        result = self.getNearestVirus()
        if result is None:
            return None
        
        dx, dy = result
        dx_norm = dx / self.center_x
        dy_norm = dy / self.center_y
        
        return (dx_norm, dy_norm)
    
    def showViruses(self, save_png=True):
        """Display debug overlay showing detected viruses"""
        img = self._capture()
        viruses = self._find_viruses(img)
        
        # Sort by distance
        viruses.sort(key=lambda v: ((v['x'] - self.center_x)**2 + (v['y'] - self.center_y)**2)**0.5)
        
        # Draw center crosshair (player position)
        cv2.line(img, (self.center_x - 20, self.center_y), (self.center_x + 20, self.center_y), (255, 255, 255), 2)
        cv2.line(img, (self.center_x, self.center_y - 20), (self.center_x, self.center_y + 20), (255, 255, 255), 2)
        
        # Draw viruses (cyan circles)
        for i, v in enumerate(viruses):
            color = (255, 0, 255)  # Magenta for visibility
            cv2.circle(img, (v['x'], v['y']), v['radius'], color, 2)
            cv2.circle(img, (v['x'], v['y']), 3, color, -1)
            
            # Label with distance
            dx = v['x'] - self.center_x
            dy = v['y'] - self.center_y
            dist = int((dx**2 + dy**2)**0.5)
            cv2.putText(img, f"V{i+1} d={dist}", (v['x'] - 30, v['y'] - v['radius'] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw line from center to virus
            cv2.line(img, (self.center_x, self.center_y), (v['x'], v['y']), color, 1)
        
        # Stats
        cv2.putText(img, f"Viruses: {len(viruses)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        if viruses:
            dx, dy = viruses[0]['x'] - self.center_x, viruses[0]['y'] - self.center_y
            cv2.putText(img, f"Nearest: dx={dx}, dy={dy}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        cv2.imshow("Viruses", img)
        cv2.moveWindow("Viruses", 200, 200)
        cv2.waitKey(1)
        
        # Save PNG
        if save_png:
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            path = f"/Users/johnvincentsalvi/Downloads/RandomCodingFiles/agario/detect_viruses_{timestamp}.png"
            cv2.imwrite(path, img)
            print(f"  Saved: {path}")
        
        return viruses


# Test loop
if __name__ == "__main__":
    import time
    
    detector = VirusDetector()
    print("Starting virus detection... (Ctrl+C to stop)")
    print(f"Looking for color #12F448 (bright green)")
    print(f"Capture region: {detector.region}")
    
    while True:
        detector.showViruses()
        nearest = detector.getNearestVirus()
        viruses = detector.getViruses()
        
        if nearest:
            dx, dy = nearest
            print(f"Viruses: {len(viruses)} | Nearest: dx={dx:+5d}, dy={dy:+5d}")
        else:
            print(f"Viruses: 0 | No virus detected")
        
        time.sleep(0.5)

