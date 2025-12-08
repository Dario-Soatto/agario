import cv2
import mss
import numpy as np
import easyocr

reader = easyocr.Reader(['en'], gpu=False)
#Locaiton of the continue button Point(x=759, y=-566)

class mssMiddleRectangle:
    def __init__(self):
        with mss.mss() as sct:
            if len(sct.monitors) > 2:
                self.monitor = sct.monitors[2]
            else:
                self.monitor = sct.monitors[1]
        
        # Middle third of screen (exclude first and last third of width/height)
        width_third = self.monitor['width'] // 3
        height_third = self.monitor['height'] // 3
        
        self.region = {
            "top": self.monitor['top'] + height_third,
            "left": self.monitor['left'] + width_third,
            "width": width_third,
            "height": height_third
        }
    
    def hasContinue(self):
        """Returns True if 'continue' text is found in middle of screen"""
        with mss.mss() as sct:
            screenshot = sct.grab(self.region)
            screenshot = np.array(screenshot)
            screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
            
            results = reader.readtext(screenshot)
            text = ' '.join([r[1].lower() for r in results if r[2] >= 0.9])
            return 'continue' in text
    
    def showRegion(self, save_png=True):
        """Display the captured region for debugging"""
        with mss.mss() as sct:
            screenshot = sct.grab(self.region)
            screenshot = np.array(screenshot)
            screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
            cv2.imshow("Middle Region", screenshot)
            cv2.moveWindow("Middle Region", 500, 500)
            cv2.waitKey(1)
            
            # Save PNG
            if save_png:
                import time
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                path = f"/Users/johnvincentsalvi/Downloads/RandomCodingFiles/agario/detect_continue_{timestamp}.png"
                cv2.imwrite(path, screenshot)
                print(f"  Saved: {path}")


# Test loop
if __name__ == "__main__":
    import time
    
    detector = mssMiddleRectangle()
    print("Starting continue detection... (Ctrl+C to stop)")
    
    while True:
        detector.showRegion()
        found = detector.hasContinue()
        print(f"Continue button found: {found}")
        time.sleep(1)

