import pyautogui
import os
import time
from pynput import mouse

# Define the directory to save screenshots
save_dir = r"C:\Users\EphraimMataranyika\Pictures\Slide Shows"

# Ensure the directory exists
os.makedirs(save_dir, exist_ok=True)

def on_click(x, y, button, pressed):
    if pressed:  # Take screenshot only when mouse button is pressed
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        file_path = os.path.join(save_dir, f"screenshot_{timestamp}.png")
        screenshot = pyautogui.screenshot()
        screenshot.save(file_path)
        print(f"Screenshot saved: {file_path}")

# Listen for mouse clicks
with mouse.Listener(on_click=on_click) as listener:
    listener.join()
