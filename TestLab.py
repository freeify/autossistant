import time
import pyautogui

# Disable fail-safe for this example (in real usage, you may want to keep this enabled)
pyautogui.FAILSAFE = False


def double_click_and_drag_down(start_x, start_y, distance, steps=10, delay_between_steps=0.1):
    """
    Double-clicks at the specified position and then drags the mouse downward

    Parameters:
    start_x, start_y: Starting coordinates for the double-click
    distance: Total distance to drag downward in pixels
    steps: Number of incremental movements for the drag
    delay_between_steps: Time to wait between each step in seconds
    """
    # Move to the starting position
    pyautogui.moveTo(start_x, start_y)

    # Perform a double-click
    pyautogui.doubleClick()

    # Short pause after double-click before starting the drag
    time.sleep(0.2)

    # Press and hold the left mouse button
    pyautogui.mouseDown()

    # Calculate the distance to move in each step
    step_distance = distance / steps

    # Perform the drag operation in steps
    for i in range(1, steps + 1):
        # Calculate new y position for this step
        current_y = start_y + (step_distance * i)

        # Move to new position while holding button
        pyautogui.moveTo(start_x, current_y)

        # Pause briefly to simulate slower drag
        time.sleep(delay_between_steps)

    # Release the mouse button
    pyautogui.mouseUp()


# Example usage
if __name__ == "__main__":
    # Wait 3 seconds before starting to give time to switch to target window
    print("Preparing to double-click and drag in 3 seconds...")
    time.sleep(3)

    # Get current mouse position
    current_x, current_y = pyautogui.position()
    print(f"Starting at position: {current_x}, {current_y}")

    # Double-click and drag down 200 pixels in 10 steps
    double_click_and_drag_down(current_x, current_y, distance=2, steps=10)


    print("Double-click and drag completed")