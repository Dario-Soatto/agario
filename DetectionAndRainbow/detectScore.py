import cv2
import mss
import numpy as np
import easyocr

reader = easyocr.Reader(['en'], gpu=False)


class mssBottomLeftCorner:
    def __init__(self):
        with mss.mss() as sct:
            if len(sct.monitors) > 2:
                self.monitor = sct.monitors[2]
            else:
                self.monitor = sct.monitors[1]
        
        self.region = {
            "top": self.monitor['top'] + int(self.monitor['height'] * 0.8),
            "left": self.monitor['left'],
            "width": int(self.monitor['width'] * 0.2),
            "height": int(self.monitor['height'] * 0.2)
        }
    
    def getScore(self):
        with mss.mss() as sct:
            screenshot = sct.grab(self.region)
            screenshot = np.array(screenshot)
            screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
            
            # Mask to extract white/bright pixels only
            hsv = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 30, 255])
            mask = cv2.inRange(hsv, lower_white, upper_white)
            white_only = cv2.bitwise_and(screenshot, screenshot, mask=mask)
            
            results = reader.readtext(white_only)
            digits = ''.join(filter(str.isdigit, ''.join([r[1] for r in results])))
            return int(digits) if digits else 0
    
    def showScore(self, save_png=True):
        with mss.mss() as sct:
            screenshot = sct.grab(self.region)
            screenshot = np.array(screenshot)
            screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
            
            # Apply same white mask as getScore
            hsv = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 30, 255])
            mask = cv2.inRange(hsv, lower_white, upper_white)
            white_only = cv2.bitwise_and(screenshot, screenshot, mask=mask)
            
            cv2.imshow("Score Region (Masked)", white_only)
            cv2.moveWindow("Score Region (Masked)", 500, 500)
            cv2.waitKey(1)
            
            # Save PNG
            if save_png:
                import time
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                path = f"/Users/johnvincentsalvi/Downloads/RandomCodingFiles/agario/detect_score_{timestamp}.png"
                cv2.imwrite(path, white_only)
                print(f"  Saved: {path}")


# Test loop
if __name__ == "__main__":
    import time
    
    detector = mssBottomLeftCorner()
    print("Starting score detection... (Ctrl+C to stop)")
    print("Press any key on the image window to continue each frame")
    
    while True:
        detector.showScore()
        score = detector.getScore()
        print(f"Score: {score}")
        time.sleep(1)
