import cv2
import numpy as np
import mss
import mss.tools
import time
import threading
import pyautogui
import os


class AccurateCaretTracker:
    """
    An efficient caret tracker class that detects and tracks text caret position on screen.
    Uses optimized frame differencing to detect the blinking cursor with high precision.
    """

    def __init__(self, debug_mode=False):
        # --- Configuration & Tuning Parameters ---
        self.DEBUG_MODE = debug_mode  # Set to True to print detailed candidate info

        # -- Adaptive Timing Parameters --
        self.MIN_INTER_FRAME_DELAY = 0.02  # Minimum seconds between frame captures
        self.MAX_INTER_FRAME_DELAY = 0.1  # Maximum seconds between frame captures
        self.INTER_FRAME_DELAY = 0.06  # Initial delay, will auto-adjust
        self.last_caret_detection = 0  # Time of last caret detection
        self.frames_without_detection = 0  # Count frames with no caret detected

        # -- Dynamic Detection Sensitivity --
        # Baseline threshold that will be adjusted dynamically
        self.BASE_DIFFERENCE_THRESHOLD = 1
        self.DIFFERENCE_THRESHOLD = self.BASE_DIFFERENCE_THRESHOLD
        self.MAX_DIFFERENCE_THRESHOLD = 5

        # -- Region of Interest (ROI) --
        self.USE_ROI = True
        # Initial ROI (will be updated dynamically)
        self.ROI_X, self.ROI_Y, self.ROI_W, self.ROI_H = 500, 300, 800, 600
        # Padding around caret for new ROI
        self.ROI_PADDING_X = 100
        self.ROI_PADDING_Y = 50
        # Whether to auto-update ROI based on caret movement
        self.AUTO_UPDATE_ROI = True

        # -- Shape Filtering --
        self.MIN_CARET_WIDTH = 1
        self.MAX_CARET_WIDTH = 5  # Allow slightly wider for anti-aliasing/block cursors
        self.MIN_CARET_HEIGHT = 10
        self.MAX_CARET_HEIGHT = 28  # Adjust based on typical line height
        self.MIN_ASPECT_RATIO = 2.0  # (Height / Width) - slightly lower for better detection

        # -- Optimized Morphological Operations --
        # Kernel size for dilation (helps connect broken parts of the caret difference)
        self.DILATE_KERNEL_SIZE = (2, 5)  # Rectangular kernel often good for vertical carets
        self.DILATE_ITERATIONS = 1
        # Only apply dilation to areas of interest
        self.SELECTIVE_DILATION = True
        # Maximum number of regions to dilate
        self.MAX_DILATION_REGIONS = 10

        # -- Performance Monitoring --
        self.processing_times = []  # Track processing time per frame
        self.MAX_TIMES_TO_TRACK = 50  # Limit history to prevent memory growth

        # -- State Variables --
        self.caret_position = None  # Stores (absolute_x, absolute_y) tuple or None
        self.prev_caret_position = None  # Previous caret position for change detection
        self.caret_saved = False  # Flag to ensure we save the caret image only once
        self.is_tracking = False
        self.pause_tracking = False  # Flag to temporarily pause tracking
        self.track_thread = None
        self.dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.DILATE_KERNEL_SIZE)

        # Time to pause tracking after a selection operation
        self.PAUSE_AFTER_SELECTION = 0.5  # seconds

        # -- Optional: Save Directory for Caret Images --
        self.SAVE_DIRECTORY = None  # Set this to a path if you want to save caret images

        # Log initialization
        print(f"[AccurateCaretTracker] Initialized with adaptive parameters")
        print(f"[AccurateCaretTracker] DEBUG_MODE: {self.DEBUG_MODE}")
        print(f"[AccurateCaretTracker] Initial DIFFERENCE_THRESHOLD: {self.DIFFERENCE_THRESHOLD}")
        print(
            f"[AccurateCaretTracker] Shape Filters: W=({self.MIN_CARET_WIDTH}-{self.MAX_CARET_WIDTH}), H=({self.MIN_CARET_HEIGHT}-{self.MAX_CARET_HEIGHT}), MinAspect={self.MIN_ASPECT_RATIO}")

    def set_roi(self, x, y, width, height):
        """Set a region of interest for caret detection."""
        self.USE_ROI = True
        self.ROI_X, self.ROI_Y, self.ROI_W, self.ROI_H = x, y, width, height
        if self.DEBUG_MODE:
            print(f"[AccurateCaretTracker] Set ROI to: {x}, {y}, {width}x{height}")

    def disable_roi(self):
        """Disable region of interest tracking."""
        self.USE_ROI = False
        print("[AccurateCaretTracker] Disabled ROI")

    def set_save_directory(self, directory_path):
        """Set directory to save caret images."""
        self.SAVE_DIRECTORY = directory_path
        os.makedirs(self.SAVE_DIRECTORY, exist_ok=True)
        print(f"[AccurateCaretTracker] Set save directory to: {directory_path}")

    def start_tracking(self):
        """Start caret tracking in a separate thread."""
        if not self.is_tracking:
            self.is_tracking = True
            self.track_thread = threading.Thread(target=self._tracking_loop, daemon=True)
            self.track_thread.start()
            print("[AccurateCaretTracker] Started tracking.")

    def stop_tracking(self):
        """Stop caret tracking."""
        self.is_tracking = False
        if self.track_thread:
            self.track_thread.join(timeout=1.0)
            self.track_thread = None
        print("[AccurateCaretTracker] Stopped tracking.")

    def pause(self, duration=None):
        """Temporarily pause tracking for a specified duration."""
        self.pause_tracking = True
        if duration:
            threading.Timer(duration, self.resume).start()
            if self.DEBUG_MODE:
                print(f"[AccurateCaretTracker] Paused tracking for {duration} seconds")
        else:
            if self.DEBUG_MODE:
                print("[AccurateCaretTracker] Paused tracking indefinitely")

    def resume(self):
        """Resume tracking after a pause."""
        self.pause_tracking = False
        if self.DEBUG_MODE:
            print("[AccurateCaretTracker] Resumed tracking")

    def get_caret_position(self):
        """Get the current caret position."""
        return self.caret_position

    def _update_roi_based_on_caret(self):
        """Update ROI based on current caret position to keep it in view."""
        if not self.caret_position or not self.AUTO_UPDATE_ROI:
            return

        x, y = self.caret_position

        # Check if caret is near edges of current ROI
        edge_threshold = 50  # pixels from edge
        update_needed = False

        if x < self.ROI_X + edge_threshold or x > self.ROI_X + self.ROI_W - edge_threshold:
            update_needed = True
        if y < self.ROI_Y + edge_threshold or y > self.ROI_Y + self.ROI_H - edge_threshold:
            update_needed = True

        if update_needed:
            # Create new ROI centered around caret position
            new_x = max(0, x - self.ROI_PADDING_X * 2)
            new_y = max(0, y - self.ROI_PADDING_Y * 2)

            self.set_roi(new_x, new_y, self.ROI_PADDING_X * 4, self.ROI_PADDING_Y * 4)

            if self.DEBUG_MODE:
                print(f"[AccurateCaretTracker] Updated ROI to follow caret at ({x}, {y})")

    def _adjust_parameters_dynamically(self, current_gray):
        """Adjust detection parameters based on current conditions."""
        # Adjust threshold based on image brightness
        avg_brightness = np.mean(current_gray)
        brightness_factor = avg_brightness / 128.0  # Normalize around middle gray

        # Adjust threshold - brighter images need higher thresholds
        new_threshold = self.BASE_DIFFERENCE_THRESHOLD
        if brightness_factor > 1.2:  # Bright image
            new_threshold = min(self.MAX_DIFFERENCE_THRESHOLD,
                                self.BASE_DIFFERENCE_THRESHOLD * brightness_factor)
        elif brightness_factor < 0.8:  # Dark image
            new_threshold = max(1, self.BASE_DIFFERENCE_THRESHOLD * brightness_factor)

        self.DIFFERENCE_THRESHOLD = new_threshold

        # Adjust frame rate based on detection success
        current_time = time.time()
        time_since_detection = current_time - self.last_caret_detection

        if self.caret_position is None:
            self.frames_without_detection += 1
            # If no detection for a while, speed up frame rate
            if self.frames_without_detection > 10:
                self.INTER_FRAME_DELAY = max(self.MIN_INTER_FRAME_DELAY,
                                             self.INTER_FRAME_DELAY * 0.9)
        else:
            self.frames_without_detection = 0
            self.last_caret_detection = current_time
            # When consistently detecting, can slow down frame rate to save resources
            self.INTER_FRAME_DELAY = min(self.MAX_INTER_FRAME_DELAY,
                                         self.INTER_FRAME_DELAY * 1.05)

    def _selective_dilation(self, thresh_diff):
        """Apply dilation only to regions that might contain a caret."""
        if not self.SELECTIVE_DILATION:
            # Fall back to standard dilation of the entire image
            return cv2.dilate(thresh_diff, self.dilate_kernel, iterations=self.DILATE_ITERATIONS)

        # Find contours in the thresholded difference image
        contours, _ = cv2.findContours(thresh_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a mask for dilation
        mask = np.zeros_like(thresh_diff)

        # Sort contours by size (small to large) and take only the smallest ones
        # as they're more likely to be caret changes
        contours = sorted(contours, key=cv2.contourArea)[:self.MAX_DILATION_REGIONS]

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Filter by potential caret size before dilating
            if w <= self.MAX_CARET_WIDTH * 3 and h <= self.MAX_CARET_HEIGHT * 2:
                # Add padding for dilation
                padding = 3
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(thresh_diff.shape[1], x + w + padding)
                y2 = min(thresh_diff.shape[0], y + h + padding)

                # Extract region and dilate it
                region = thresh_diff[y1:y2, x1:x2]
                dilated_region = cv2.dilate(region, self.dilate_kernel, iterations=self.DILATE_ITERATIONS)

                # Place back into mask
                mask[y1:y2, x1:x2] = dilated_region

        # If no regions were dilated, just return the original
        if np.sum(mask) == 0:
            return thresh_diff

        # Combine with original threshold using max operation
        return np.maximum(thresh_diff, mask)

    def _tracking_loop(self):
        """Main tracking loop running in a separate thread with optimized processing."""
        print("[AccurateCaretTracker] Starting tracking loop...")
        print(
            f"[AccurateCaretTracker] USE_ROI: {self.USE_ROI} (Region: {self.ROI_X},{self.ROI_Y} {self.ROI_W}x{self.ROI_H})")

        try:
            sct = mss.mss()
        except Exception as e:
            print(f"[AccurateCaretTracker] Error initializing mss: {e}")
            self.is_tracking = False
            return

        # Define monitor region and absolute offset
        absolute_offset_x = 0
        absolute_offset_y = 0
        monitor = None

        if self.USE_ROI:
            monitor = {"top": self.ROI_Y, "left": self.ROI_X, "width": self.ROI_W, "height": self.ROI_H}
            if self.ROI_W <= 0 or self.ROI_H <= 0:
                print(f"[AccurateCaretTracker] ERROR: Invalid ROI dimensions.")
                self.is_tracking = False
                return
            absolute_offset_x = self.ROI_X
            absolute_offset_y = self.ROI_Y
        else:
            try:
                # Use primary monitor with a focused area in the center for better performance
                monitor = sct.monitors[1]
                screen_width = monitor.get('width', 1920)
                screen_height = monitor.get('height', 1080)

                # Focus on center area where text editing is likely happening
                center_x = screen_width // 2
                center_y = screen_height // 2
                roi_width = min(800, screen_width)
                roi_height = min(600, screen_height)

                # Calculate ROI coordinates
                roi_left = max(0, center_x - (roi_width // 2))
                roi_top = max(0, center_y - (roi_height // 2))

                absolute_offset_x = roi_left
                absolute_offset_y = roi_top

                monitor = {"top": roi_top, "left": roi_left,
                           "width": roi_width, "height": roi_height}

                print(f"[AccurateCaretTracker] Using optimized screen area: {monitor}")
            except Exception as e:
                print(f"[AccurateCaretTracker] Error getting monitor info: {e}")
                self.is_tracking = False
                return

        prev_gray = None
        last_capture_time = time.time()

        # Ensure the save directory exists if specified
        if self.SAVE_DIRECTORY:
            os.makedirs(self.SAVE_DIRECTORY, exist_ok=True)

        frame_count = 0

        while self.is_tracking:
            frame_count += 1
            if self.pause_tracking:
                # When paused, sleep but continue the loop
                time.sleep(0.1)
                last_capture_time = time.time()  # Reset timer to avoid immediate capture after resuming
                continue

            current_time = time.time()
            # --- Frame Rate Control ---
            time_since_last = current_time - last_capture_time
            if time_since_last < self.INTER_FRAME_DELAY:
                sleep_duration = self.INTER_FRAME_DELAY - time_since_last
                if sleep_duration > 0:
                    time.sleep(sleep_duration)

            # Adaptive ROI update
            if self.USE_ROI and self.AUTO_UPDATE_ROI and frame_count % 5 == 0:
                self._update_roi_based_on_caret()
                # Update monitor with new ROI if it changed
                if self.USE_ROI:
                    monitor = {"top": self.ROI_Y, "left": self.ROI_X,
                               "width": self.ROI_W, "height": self.ROI_H}
                    absolute_offset_x = self.ROI_X
                    absolute_offset_y = self.ROI_Y

            # --- Capture Frame ---
            processing_start = time.time()
            try:
                sct_img = sct.grab(monitor)
                last_capture_time = time.time()  # Record actual capture time
                current_bgr = np.array(sct_img)
                current_gray = cv2.cvtColor(current_bgr, cv2.COLOR_BGRA2GRAY)
            except mss.ScreenShotError as ex:
                print(f"[AccurateCaretTracker] ScreenShotError: {ex}. Retrying...")
                time.sleep(0.5)
                prev_gray = None
                continue
            except Exception as e:
                print(f"[AccurateCaretTracker] Error capturing screen: {e}")
                time.sleep(0.5)
                prev_gray = None
                continue

            if prev_gray is None or prev_gray.shape != current_gray.shape:
                prev_gray = current_gray.copy()
                continue

            # --- Dynamic Parameter Adjustment ---
            if frame_count % 10 == 0:  # Adjust every 10 frames
                self._adjust_parameters_dynamically(current_gray)

            # --- Calculate Difference ---
            frame_diff = cv2.absdiff(prev_gray, current_gray)

            # --- Threshold Difference ---
            _, thresh_diff = cv2.threshold(frame_diff, self.DIFFERENCE_THRESHOLD, 255, cv2.THRESH_BINARY)

            # --- Optimized Morphological Dilation ---
            dilated_diff = self._selective_dilation(thresh_diff)

            # --- Find Contours ---
            contours, _ = cv2.findContours(dilated_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            potential_carets = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)

                # --- Filter Contours based on shape ---
                is_size_ok = (self.MIN_CARET_WIDTH <= w <= self.MAX_CARET_WIDTH and
                              self.MIN_CARET_HEIGHT <= h <= self.MAX_CARET_HEIGHT)

                if is_size_ok:
                    aspect_ratio = float(h) / w if w > 0 else 0
                    is_aspect_ok = aspect_ratio >= self.MIN_ASPECT_RATIO
                    is_plausible_block = (aspect_ratio < self.MIN_ASPECT_RATIO and
                                          w >= self.MIN_CARET_WIDTH and h >= self.MIN_CARET_HEIGHT * 0.8)

                    if is_aspect_ok or is_plausible_block:
                        potential_carets.append((x, y, w, h, aspect_ratio))
                        if self.DEBUG_MODE:
                            abs_x_debug = absolute_offset_x + x
                            abs_y_debug = absolute_offset_y + y
                            print(
                                f"[AccurateCaretTracker] Candidate @({abs_x_debug},{abs_y_debug}) Rel({x},{y}) Size: {w}x{h} Aspect: {aspect_ratio:.2f}")

            # --- Identify Best Caret Candidate ---
            best_caret_rect = None

            # Sort candidates by aspect ratio in descending order (prefer taller, thinner shapes)
            if potential_carets:
                potential_carets.sort(key=lambda c: c[4], reverse=True)
                best_caret_rect = potential_carets[0][:4]  # Take the first 4 elements (x,y,w,h)

            # Store the previous position before updating
            self.prev_caret_position = self.caret_position

            # --- Update Shared State with ABSOLUTE Coordinates ---
            if best_caret_rect:
                x_rel, y_rel, w, h = best_caret_rect
                # Calculate center and ABSOLUTE screen coordinates
                center_x_abs = absolute_offset_x + x_rel + w // 2
                center_y_abs = absolute_offset_y + y_rel + h // 2

                # Only log position changes beyond a small threshold
                position_changed = (self.caret_position is None or
                                    abs(self.caret_position[0] - center_x_abs) > 2 or
                                    abs(self.caret_position[1] - center_y_abs) > 2)

                if position_changed and self.DEBUG_MODE:
                    print(f"[AccurateCaretTracker] Detected caret at: ({center_x_abs}, {center_y_abs})")

                self.caret_position = (center_x_abs, center_y_abs)
                self.last_caret_detection = time.time()

                # --- Save the Caret Image Once if directory is set ---
                if self.SAVE_DIRECTORY and not self.caret_saved:
                    # Crop the detected caret region from the current BGR frame
                    caret_img = current_bgr[y_rel:y_rel + h, x_rel:x_rel + w]
                    timestamp = int(time.time())
                    save_path = os.path.join(self.SAVE_DIRECTORY, f"caret_{timestamp}.png")
                    if cv2.imwrite(save_path, caret_img):
                        print(f"[AccurateCaretTracker] Caret image saved to: {save_path}")
                    else:
                        print(f"[AccurateCaretTracker] ERROR: Failed to save caret image to: {save_path}")
                    self.caret_saved = True  # Set flag to prevent repeated saving
            else:
                self.caret_position = None

            # --- Update Previous Frame ---
            prev_gray = current_gray.copy()

            # --- Track processing performance ---
            processing_time = time.time() - processing_start
            self.processing_times.append(processing_time)

            # Limit the size of the processing times list
            if len(self.processing_times) > self.MAX_TIMES_TO_TRACK:
                self.processing_times = self.processing_times[-self.MAX_TIMES_TO_TRACK:]

            # Every 100 frames, output performance metrics if in debug mode
            if self.DEBUG_MODE and frame_count % 100 == 0:
                avg_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
                print(f"[AccurateCaretTracker] Avg processing time: {avg_time * 1000:.2f}ms, " +
                      f"Frame rate: {1 / self.INTER_FRAME_DELAY:.1f}fps, " +
                      f"Threshold: {self.DIFFERENCE_THRESHOLD:.1f}")

        sct.close()
        print("[AccurateCaretTracker] Tracking loop ended.")

    def double_click_and_drag_down(self, start_x, start_y, num_lines, line_height=20, steps=10,
                                   delay_between_steps=0.05):
        """
        Double-clicks at the specified position and then drags the mouse downward based on number of lines.
        Modified to ensure selection starts precisely at the first line.

        Parameters:
        start_x, start_y: Starting coordinates for the double-click
        num_lines: Number of lines to select
        line_height: Average height of a single line in pixels
        steps: Number of incremental movements for the drag
        delay_between_steps: Time to wait between each step in seconds
        """
        # Pause tracking during selection to prevent interference
        self.pause()

        try:
            # Calculate total distance based on line count and line height
            distance = num_lines * line_height

            # Move precisely to the starting position - this is critical
            print(f"[AccurateCaretTracker] Moving precisely to first line position ({start_x}, {start_y})")
            pyautogui.moveTo(start_x, start_y)
            time.sleep(0.3)  # Longer pause for stability

            # Ensure we're exactly at the beginning of the line
            # Note: We don't press any keys that would move us to another line
            pyautogui.press('home')
            time.sleep(0.2)

            # Perform a single click to ensure focus without changing position
            pyautogui.click(start_x, start_y)
            time.sleep(0.2)

            print(f"[AccurateCaretTracker] Double-clicking to start selection at ({start_x}, {start_y})")
            # Perform a double-click exactly at the specified position
            pyautogui.doubleClick(start_x, start_y)

            # If only selecting one line, we're done after double-click
            if num_lines <= 1:
                print(f"[AccurateCaretTracker] Selected single line with double-click")
                return

            # Short pause after double-click before starting the drag
            time.sleep(0.3)

            print(
                f"[AccurateCaretTracker] Starting drag FROM FIRST LINE to select {num_lines} lines ({distance} pixels)")
            # Press and hold the left mouse button
            pyautogui.mouseDown()

            # Calculate the distance to move in each step
            step_distance = distance / steps

            # Perform the drag operation in steps - only moving DOWN from the first line
            for i in range(1, steps + 1):
                # Calculate new y position for this step
                current_y = start_y + (step_distance * i)

                # Move to new position while holding button - only changing Y coordinate
                pyautogui.moveTo(start_x, current_y)

                # Pause briefly to simulate slower drag
                time.sleep(delay_between_steps)

            # Release the mouse button
            pyautogui.mouseUp()
            print(
                f"[AccurateCaretTracker] Drag completed - selected approximately {num_lines} lines starting from the first line")

        finally:
            # Resume tracking after a short delay
            threading.Timer(self.PAUSE_AFTER_SELECTION, self.resume).start()

    def force_caret_appearance(self, x=None, y=None):
        """
        Force the caret to appear by performing minimal keyboard navigation at the specified position.
        Performs careful movements to ensure the cursor stays on the same line.

        Parameters:
        x, y: Optional coordinates to click before forcing caret appearance
        """
        # Pause tracking during keyboard operations
        self.pause()

        try:
            # Store current position if we need to return to it
            current_pos = None
            if x is None or y is None:
                current_pos = pyautogui.position()

            # If coordinates provided, move and click there first
            if x is not None and y is not None:
                pyautogui.moveTo(x, y)
                pyautogui.click()
                time.sleep(0.2)

            # Sequence of keyboard actions to make the caret visible
            print("[AccurateCaretTracker] Forcing caret to appear with minimal movement...")

            # Technique 1: End+Home (stays on same line)
            pyautogui.press('end')
            time.sleep(0.1)
            pyautogui.press('home')
            time.sleep(0.2)

            # Resume tracking briefly to check if caret is detected
            self.resume()
            time.sleep(0.2)

            if self.caret_position is None:
                # Pause again for the second technique
                self.pause()

                # Technique 2: Minimal Right+Left movement
                pyautogui.press('right')
                time.sleep(0.1)
                pyautogui.press('left')
                time.sleep(0.2)

            if self.caret_position is None:
                # Pause again for the third technique
                self.pause()

                # Technique 3: Home+Right+Left (still on same line)
                pyautogui.press('home')
                time.sleep(0.1)
                pyautogui.press('right')
                time.sleep(0.1)
                pyautogui.press('left')
                time.sleep(0.2)

            # If we still haven't detected the caret, try a more aggressive approach
            # but only as a last resort
            if self.caret_position is None:
                print("[AccurateCaretTracker] Minimal movements failed, trying last resort technique")

                # Click again to ensure focus
                if x is not None and y is not None:
                    pyautogui.click(x, y)
                else:
                    pyautogui.click()
                time.sleep(0.2)

                # Press escape to cancel any potential selection or dialog
                pyautogui.press('escape')
                time.sleep(0.1)

                # Try quick Home+End
                pyautogui.press('home')
                time.sleep(0.1)
                pyautogui.press('end')
                time.sleep(0.1)
                pyautogui.press('home')
                time.sleep(0.2)

        finally:
            # Resume tracking
            self.resume()

            # Give tracker a moment to detect the caret
            time.sleep(0.3)

            # Return to original position if needed
            if current_pos is not None and (x is None or y is None):
                pyautogui.moveTo(current_pos.x, current_pos.y)

            # Return result
            if self.caret_position:
                print(f"[AccurateCaretTracker] Caret now visible at {self.caret_position}")
                return True
            else:
                print("[AccurateCaretTracker] Failed to make caret appear")
                return False


# Added an example to show how to use this module directly
if __name__ == "__main__":
    print("Starting AccurateCaretTracker test...")
    tracker = AccurateCaretTracker(debug_mode=True)

    # Set a more focused ROI for better performance
    screen_width, screen_height = pyautogui.size()
    center_x = screen_width // 2
    center_y = screen_height // 2
    tracker.set_roi(center_x - 400, center_y - 300, 800, 600)

    tracker.start_tracking()

    try:
        print("Caret tracker running. Press Ctrl+C to exit.")
        while True:
            time.sleep(1)
            position = tracker.get_caret_position()
            if position:
                print(f"Current caret position: {position}")
    except KeyboardInterrupt:
        print("Stopping tracker...")
        tracker.stop_tracking()
        print("Tracker stopped.")