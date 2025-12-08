import pyautogui
import time

width_of_screen = pyautogui.size().width
height_of_screen = pyautogui.size().height
print(f"width_of_screen: {width_of_screen}, height_of_screen: {height_of_screen}")
center_of_screen = (width_of_screen / 2, height_of_screen / 2)
print(f"center_of_screen: {center_of_screen}")

class ScreenAction:
    def __init__(self, screen_width, screen_height, x_offset=0, y_offset=0):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.center_on_the_x = screen_width / 2 + x_offset
        self.center_on_the_y = screen_height / 2 + y_offset
        self.dx = screen_width * 0.20       # 20% of width
        self.dy = screen_height * 0.20      # 20% of height
    
    # ============== SEMANTIC MOVEMENT (for DQN) ==============
    def move_relative(self, dx, dy, scale=1.0):
        """
        Move mouse to position offset from screen center.
        dx, dy are relative pixel offsets (e.g., from detection).
        """
        target_x = self.center_on_the_x + dx * scale
        target_y = self.center_on_the_y + dy * scale
        # Clamp to screen bounds
        target_x = max(0, min(self.screen_width - 1, target_x))
        target_y = max(0, min(self.screen_height - 1, target_y))
        pyautogui.moveTo(target_x, target_y)
    
    def move_toward(self, dx, dy):
        """Move toward entity at (dx, dy) relative to center"""
        self.move_relative(dx, dy, scale=1.0)
    
    def move_away(self, dx, dy):
        """Move away from entity at (dx, dy) - opposite direction"""
        self.move_relative(-dx, -dy, scale=1.0)
    
    # ============== FIXED DIRECTIONAL MOVEMENT (legacy) ==============
    def move_diagonal_up_right(self):
        pyautogui.moveTo(self.center_on_the_x + self.dx, self.center_on_the_y - self.dy)

    def move_diagonal_down_right(self):
        pyautogui.moveTo(self.center_on_the_x + self.dx, self.center_on_the_y + self.dy)

    def move_diagonal_up_left(self):
        pyautogui.moveTo(self.center_on_the_x - self.dx, self.center_on_the_y - self.dy)

    def move_diagonal_down_left(self):
        pyautogui.moveTo(self.center_on_the_x - self.dx, self.center_on_the_y + self.dy)

    def move_right(self):
        pyautogui.moveTo(self.center_on_the_x + self.dx, self.center_on_the_y)

    def move_left(self):
        pyautogui.moveTo(self.center_on_the_x - self.dx, self.center_on_the_y)

    def move_up(self):
        pyautogui.moveTo(self.center_on_the_x, self.center_on_the_y - self.dy)

    def move_down(self):
        pyautogui.moveTo(self.center_on_the_x, self.center_on_the_y + self.dy)

# print("Testing diagonal up right movement...")
# agario_screen_action = ScreenAction(width_of_screen, height_of_screen)
# agario_screen_action.move_diagonal_up_right()
# time.sleep(1)
# print("Testing diagonal down right movement...")
# agario_screen_action.move_diagonal_down_right()
# time.sleep(1)
# print("Testing diagonal up left movement...")
# agario_screen_action.move_diagonal_up_left()
# time.sleep(1)
# print("Testing diagonal down left movement...")
# agario_screen_action.move_diagonal_down_left()
# time.sleep(1)
# print("Testing right movement...")
# agario_screen_action.move_right()
# time.sleep(1)
# print("Testing left movement...")
# agario_screen_action.move_left()
# time.sleep(1)
# print("Testing up movement...")
# agario_screen_action.move_up()
# time.sleep(1)
# print("Testing down movement...")
# agario_screen_action.move_down()
# time.sleep(1)