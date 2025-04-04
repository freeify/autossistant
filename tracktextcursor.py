# --- Start of Combined Code ---

import cv2
import numpy as np
import mss
import mss.tools
import time
import sys
import threading
import tkinter as tk
import os  # New: For file/directory handling

# --- Configuration & Tuning Parameters (FROM YOUR PROVIDED SCRIPT) ---

# -- Debugging (Less relevant for overlay, but keep detection params) --
DEBUG_MODE = False  # Set to True to print candidate info in detection thread

# -- Timing --
# CRITICAL: Adjust based on your caret's blink rate. Try 0.05 to 0.15
INTER_FRAME_DELAY = 0.08  # Seconds between frame captures

# -- Detection Sensitivity --
# CRITICAL: Adjust based on contrast. Lower = more sensitive. Higher = less noise. (0-255)
DIFFERENCE_THRESHOLD = 1

# -- Region of Interest (ROI) --
# HIGHLY RECOMMENDED: Set USE_ROI=True and define the area to track in
USE_ROI = False  # Set to True to only track within a specific rectangle
# Define ROI (Top-Left X, Top-Left Y, Width, Height) - Find these manually or using a tool
ROI_X, ROI_Y, ROI_W, ROI_H = 500, 300, 800, 600  # EXAMPLE VALUES - CHANGE THESE!

# -- Shape Filtering --
# CRITICAL: Adjust based on your font size, resolution, and cursor type (I-beam vs Block)
MIN_CARET_WIDTH = 1
MAX_CARET_WIDTH = 5    # Allow slightly wider for anti-aliasing/block cursors
MIN_CARET_HEIGHT = 10
MAX_CARET_HEIGHT = 28  # Adjust based on typical line height
MIN_ASPECT_RATIO = 2.5  # (Height / Width). Lower for block cursors (e.g., 0.8)

# -- Morphological Operations --
# Kernel size for dilation (helps connect broken parts of the caret difference)
DILATE_KERNEL_SIZE = (2, 5)  # Rectangular kernel often good for vertical carets
DILATE_ITERATIONS = 1

# -- Overlay Marker Appearance --
MARKER_SIZE = 6  # Size of the red square marker in pixels
MARKER_COLOR = 'red'
UPDATE_INTERVAL_MS = 30  # How often Tkinter checks for position updates (milliseconds)

# --- File Saving Configuration ---
SAVE_DIRECTORY = r"C:\Users\EphraimMataranyika\Pictures\Screenshots\Omni Parser\organized_icons"

# --- Shared State (between threads) ---
caret_position = None  # Stores (absolute_x, absolute_y) tuple or None
caret_saved = False    # New: Flag to ensure we save the caret image only once
request_stop = False   # Flag to signal threads to stop

# --- Detection Logic (runs in a separate thread) ---
def detection_thread_func():
    global caret_position, request_stop, caret_saved

    print("[Detection Thread] Starting...")
    print("[Detection Thread] --- TUNING IS LIKELY REQUIRED ---")
    print(f"[Detection Thread] DEBUG_MODE: {DEBUG_MODE}")
    print(f"[Detection Thread] INTER_FRAME_DELAY: {INTER_FRAME_DELAY}")
    print(f"[Detection Thread] DIFFERENCE_THRESHOLD: {DIFFERENCE_THRESHOLD}")
    print(f"[Detection Thread] USE_ROI: {USE_ROI} (Region: {ROI_X},{ROI_Y} {ROI_W}x{ROI_H})")
    print(f"[Detection Thread] Shape Filters: W=({MIN_CARET_WIDTH}-{MAX_CARET_WIDTH}), H=({MIN_CARET_HEIGHT}-{MAX_CARET_HEIGHT}), MinAspect={MIN_ASPECT_RATIO}")

    try:
        sct = mss.mss()
    except Exception as e:
        print(f"[Detection Thread] Error initializing mss: {e}")
        request_stop = True
        return

    # Define monitor region and absolute offset
    absolute_offset_x = 0
    absolute_offset_y = 0
    monitor = None

    if USE_ROI:
        monitor = {"top": ROI_Y, "left": ROI_X, "width": ROI_W, "height": ROI_H}
        if ROI_W <= 0 or ROI_H <= 0:
            print(f"[Detection Thread] ERROR: Invalid ROI dimensions.")
            request_stop = True
            return
        absolute_offset_x = ROI_X
        absolute_offset_y = ROI_Y
        print(f"[Detection Thread] Using ROI: {monitor}")
    else:
        try:
            # Use primary monitor. Assume its top-left is (0,0) for simplicity
            monitor = sct.monitors[1]
            absolute_offset_x = monitor.get('left', 0)
            absolute_offset_y = monitor.get('top', 0)
            print(f"[Detection Thread] Using Primary Monitor: {monitor}")
            monitor = {"top": absolute_offset_y, "left": absolute_offset_x,
                       "width": monitor['width'], "height": monitor['height']}
        except Exception as e:
            print(f"[Detection Thread] Error getting monitor info: {e}")
            request_stop = True
            return

    prev_gray = None
    last_capture_time = time.time()
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, DILATE_KERNEL_SIZE)

    # Ensure the save directory exists
    os.makedirs(SAVE_DIRECTORY, exist_ok=True)

    while not request_stop:
        current_time = time.time()
        # --- Frame Rate Control ---
        time_since_last = current_time - last_capture_time
        if time_since_last < INTER_FRAME_DELAY:
            sleep_duration = INTER_FRAME_DELAY - time_since_last
            if sleep_duration > 0:
                time.sleep(sleep_duration)

        # --- Capture Frame ---
        try:
            sct_img = sct.grab(monitor)
            last_capture_time = time.time()  # Record actual capture time
            current_bgr = np.array(sct_img)
            current_gray = cv2.cvtColor(current_bgr, cv2.COLOR_BGRA2GRAY)
        except mss.ScreenShotError as ex:
            print(f"[Detection Thread] ScreenShotError: {ex}. Retrying...")
            time.sleep(0.5)
            prev_gray = None
            continue
        except Exception as e:
            print(f"[Detection Thread] Error capturing screen: {e}")
            time.sleep(0.5)
            prev_gray = None
            continue

        if prev_gray is None:
            prev_gray = current_gray.copy()
            continue

        # --- Calculate Difference ---
        frame_diff = cv2.absdiff(prev_gray, current_gray)

        # --- Threshold Difference ---
        _, thresh_diff = cv2.threshold(frame_diff, DIFFERENCE_THRESHOLD, 255, cv2.THRESH_BINARY)

        # --- Morphological Dilation ---
        dilated_diff = cv2.dilate(thresh_diff, dilate_kernel, iterations=DILATE_ITERATIONS)

        # --- Find Contours ---
        contours, _ = cv2.findContours(dilated_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        potential_carets = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # --- Filter Contours based on shape ---
            is_size_ok = (MIN_CARET_WIDTH <= w <= MAX_CARET_WIDTH and
                          MIN_CARET_HEIGHT <= h <= MAX_CARET_HEIGHT)

            if is_size_ok:
                aspect_ratio = float(h) / w if w > 0 else 0
                is_aspect_ok = aspect_ratio >= MIN_ASPECT_RATIO
                is_plausible_block = (aspect_ratio < MIN_ASPECT_RATIO and
                                      w >= MIN_CARET_WIDTH and h >= MIN_CARET_HEIGHT * 0.8)

                if is_aspect_ok or is_plausible_block:
                    potential_carets.append((x, y, w, h))
                    if DEBUG_MODE:
                        abs_x_debug = absolute_offset_x + x
                        abs_y_debug = absolute_offset_y + y
                        print(f"[Detection Thread] Candidate @({abs_x_debug},{abs_y_debug}) Rel({x},{y}) Size: {w}x{h} Aspect: {aspect_ratio:.2f}")

        # --- Identify Best Caret Candidate ---
        best_caret_rect = potential_carets[0] if potential_carets else None

        # --- Update Shared State with ABSOLUTE Coordinates ---
        if best_caret_rect:
            x_rel, y_rel, w, h = best_caret_rect
            # Calculate center and ABSOLUTE screen coordinates
            center_x_abs = absolute_offset_x + x_rel + w // 2
            center_y_abs = absolute_offset_y + y_rel + h // 2
            caret_position = (center_x_abs, center_y_abs)

            # --- Save the Caret Image Once ---
            if not caret_saved:
                # Crop the detected caret region from the current BGR frame
                caret_img = current_bgr[y_rel:y_rel+h, x_rel:x_rel+w]
                timestamp = int(time.time())
                save_path = os.path.join(SAVE_DIRECTORY, f"caret_{timestamp}.png")
                if cv2.imwrite(save_path, caret_img):
                    print(f"[Detection Thread] Caret image saved to: {save_path}")
                else:
                    print(f"[Detection Thread] ERROR: Failed to save caret image to: {save_path}")
                caret_saved = True  # Set flag to prevent repeated saving
        else:
            caret_position = None

        # --- Update Previous Frame ---
        prev_gray = current_gray.copy()

    sct.close()
    print("[Detection Thread] Stopped.")


# --- Tkinter Overlay Logic (runs in the main thread) ---
def update_marker_position(root, marker_window):
    global caret_position, request_stop

    if request_stop:
        try:
            root.quit()
        except tk.TclError:
            pass
        return

    current_pos = caret_position

    try:
        if current_pos:
            x, y = current_pos
            marker_x = x - MARKER_SIZE // 2
            marker_y = y - MARKER_SIZE // 2
            marker_window.geometry(f"{MARKER_SIZE}x{MARKER_SIZE}+{marker_x}+{marker_y}")
            if marker_window.state() == 'withdrawn':
                marker_window.deiconify()
            marker_window.lift()
        else:
            if marker_window.state() == 'normal':
                marker_window.withdraw()
    except tk.TclError as e:
        if "invalid command name" not in str(e):
            print(f"[Main Thread] Tkinter Error: {e}")
        request_stop = True
        try:
            root.quit()
        except tk.TclError:
            pass
        return

    root.after(UPDATE_INTERVAL_MS, update_marker_position, root, marker_window)

def on_close(root):
    global request_stop
    if not request_stop:
        print("[Main Thread] Close requested. Stopping detection thread...")
        request_stop = True

# --- Main Execution ---
if __name__ == "__main__":
    print("[Main Thread] Starting Tkinter...")
    root = tk.Tk()
    root.withdraw()

    try:
        marker_window = tk.Toplevel(root)
        marker_window.overrideredirect(True)
        marker_window.geometry(f"{MARKER_SIZE}x{MARKER_SIZE}+0+0")
        marker_window.config(bg=MARKER_COLOR)
        marker_window.attributes("-topmost", True)
        marker_window.withdraw()
    except Exception as e:
        print(f"[Main Thread] Failed to create marker window: {e}")
        sys.exit(1)

    print("[Main Thread] Starting Detection Thread...")
    detection_thread = threading.Thread(target=detection_thread_func, daemon=True)
    detection_thread.start()

    root.protocol("WM_DELETE_WINDOW", lambda: on_close(root))
    root.after(UPDATE_INTERVAL_MS, update_marker_position, root, marker_window)

    print("[Main Thread] Starting Tkinter main loop. Press Ctrl+C in terminal to quit.")
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\n[Main Thread] KeyboardInterrupt received. Stopping...")
        on_close(root)

    print("[Main Thread] Waiting for detection thread to stop...")
    detection_thread.join(timeout=1.0)
    print("[Main Thread] Exiting.")

# --- End of Combined Code ---
