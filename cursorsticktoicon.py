import pyautogui
import keyboard
import time
import sys
import cv2
import numpy as np
from PIL import ImageGrab

def fast_track_cursor_position():
    print("INSTRUCTIONS:")
    print("1. Position your cursor over the line to track")
    print("2. Press ENTER to start tracking")
    print("3. Press ESC to exit")

    # Wait for ENTER key to start tracking
    keyboard.wait('Caps lock')

    # Get the current cursor position
    target_position = pyautogui.position()
    print(f"Tracking position {target_position}")

    # Capture a smaller area for faster template matching
    template_size = 50  # Smaller size for speed
    x1 = max(0, target_position[0] - template_size//2)
    y1 = max(0, target_position[1] - template_size//2)
    x2 = x1 + template_size
    y2 = y1 + template_size

    # Capture the template
    screenshot = ImageGrab.grab(bbox=(x1, y1, x2, y2))
    template = np.array(screenshot)
    template = cv2.cvtColor(template, cv2.COLOR_RGB2BGR)

    # Define search area to avoid scanning the entire screen
    screen_width, screen_height = pyautogui.size()
    search_margin = 300  # Search only this many pixels around last position

    # Start with the initial position
    last_x, last_y = target_position

    try:
        while True:
            # Check for ESC key press to exit (non-blocking)
            if keyboard.is_pressed('esc'):
                print("Exiting")
                break

            # Define search area based on last position (with boundaries)
            search_x1 = max(0, last_x - search_margin)
            search_y1 = max(0, last_y - search_margin)
            search_x2 = min(screen_width, last_x + search_margin)
            search_y2 = min(screen_height, last_y + search_margin)

            # Take a screenshot of just the search area instead of entire screen
            search_area = np.array(ImageGrab.grab(bbox=(search_x1, search_y1, search_x2, search_y2)))
            search_area = cv2.cvtColor(search_area, cv2.COLOR_RGB2BGR)

            # Template matching on smaller area
            result = cv2.matchTemplate(search_area, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            # If good match, update cursor position
            if max_val > 0.7:
                # Calculate absolute screen position
                match_x = search_x1 + max_loc[0] + template_size//2
                match_y = search_y1 + max_loc[1] + template_size//2

                # Move cursor and update last known position
                pyautogui.moveTo(match_x, match_y)
                last_x, last_y = match_x, match_y

            # Minimal sleep to reduce CPU usage but keep responsive
            time.sleep(0.01)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("Starting in 2 seconds...")
    time.sleep(2)
    fast_track_cursor_position()