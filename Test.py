import os
import pyautogui
from pynput import mouse, keyboard
import socket
import time
import threading
import pyperclip
import anthropic
import re
import pygetwindow as gw
import difflib
from typing import Tuple, List, Dict
import tempfile
import subprocess

from caret_tracker import AccurateCaretTracker

# Configuration
CONFIG = {
    'SERVER_HOST': '127.0.0.1',
    'SERVER_PORT': 66439,
    'API_KEY': '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',
    # Delay in seconds between clicks, adjust based on system performance
    'CLICK_DELAY': 0.1,
    'COPY_PASTE_DELAY': 0.2,
    'LINE_HEIGHT': 20  # Adjust based on your editor's settings
}

pyautogui.FAILSAFE = False


class State:
    def __init__(self):
        self.chat_app_position = None
        self.chat_app_click_recorded = False
        self.first_run = True
        self.stored_chunk = None
        self.out = None
        self.starting_line_number = None
        self.insertion_point = None

        # New attributes for line counting
        self.sent_lines_count = 0
        self.llm_lines_count = 0
        self.initial_line_count = 0


# --------------------- Language and Block Detection --------------------- #

# Improved language detection with better Java patterns
def detect_language(code: str) -> str:
    """Detect programming language from code with improved Java detection."""
    patterns = {
        'python': (r'\b(def|class|if|elif|else|for|while|try|except|with)\b.*:', r'import\s+[\w\s,]+'),
        'javascript': (r'\b(function|class|if|else|for|while|try|catch)\b.*{', r'const|let|var'),
        'java': (
            r'\b(public|private|protected|class|interface|void|static)\b.*[{;]', r'import\s+[\w.]+;|package\s+[\w.]+;'),
    }
    code_sample = '\n'.join(code.split('\n')[:20])  # Look at more lines for better detection
    scores = {
        lang: len(re.findall(block_pat, code_sample, re.M)) * 2 +
              len(re.findall(extra_pat, code_sample, re.M))
        for lang, (block_pat, extra_pat) in patterns.items()
    }

    # Add Java-specific scoring boosts
    if 'public class' in code_sample or 'private class' in code_sample:
        scores['java'] += 5
    if re.search(r'\w+\s+\w+\([^)]*\)\s*{', code_sample):  # Method declarations
        scores['java'] += 3

    best_match = max(scores.items(), key=lambda x: x[1])
    return best_match[0] if best_match[1] > 0 else 'unknown'


def get_block_info(language: str) -> Dict:
    """Get language-specific block patterns with improved Java support."""
    patterns = {
        'python': {
            'indent_based': True,
            'block_start': r'^(def|class|if|elif|else|for|while|try|except|with)\b.*:$',
            'keywords': ['def', 'class', 'if', 'elif', 'else', 'for', 'while', 'try', 'except', 'with']
        },
        'javascript': {
            'indent_based': False,
            'block_start': r'^.*{$',
            'keywords': ['function', 'class', 'if', 'else', 'for', 'while', 'try', 'catch']
        },
        'java': {
            'indent_based': False,
            'block_start': r'^.*[{]$',
            'keywords': ['public', 'private', 'protected', 'class', 'interface', 'void', 'static', 'final', 'abstract'],
            'method_pattern': r'\b\w+\s+\w+\([^)]*\)\s*{'  # Pattern to detect method declarations
        },
        'unknown': {
            'indent_based': False,
            'block_start': r'^.*[{:]$',
            'keywords': ['block']
        }
    }
    return patterns.get(language, patterns['unknown'])


def find_block_boundaries(lines: List[str], cursor_line: int, language: str) -> Tuple[int, int, str]:
    """Find the start and end of the current code block with comprehensive Java support."""
    patterns = get_block_info(language)
    print(f"Language detected: {language}")
    cursor_indent = len(lines[cursor_line]) - len(lines[cursor_line].lstrip())

    # Find block start
    start = cursor_line
    block_type = 'block'

    # Special handling for Java
    if language == 'java':
        # Track the number of open braces at the cursor position
        cursor_brace_level = 0
        for i in range(0, cursor_line + 1):
            cursor_brace_level += lines[i].count('{') - lines[i].count('}')

        # Find the start of the current block by looking for the opening brace
        target_brace_level = cursor_brace_level - 1
        current_brace_level = cursor_brace_level

        # Search backwards to find the start line
        for i in range(cursor_line, -1, -1):
            line = lines[i]
            current_brace_level -= line.count('{')
            current_brace_level += line.count('}')

            if current_brace_level <= target_brace_level:
                # Found the line with the opening brace
                # Now search further up for the method/class declaration
                for j in range(i, -1, -1):
                    if lines[j].strip() and not lines[j].strip().startswith('//') and not lines[j].strip().startswith(
                            '*'):
                        # Found the declaration line
                        start = j
                        # Determine block type
                        if 'class' in lines[j]:
                            block_type = 'class'
                        elif 'interface' in lines[j]:
                            block_type = 'interface'
                        elif 'enum' in lines[j]:
                            block_type = 'enum'
                        elif re.search(r'\b\w+\s+\w+\([^)]*\)', lines[j]):
                            block_type = 'method'
                        else:
                            for keyword in patterns['keywords']:
                                if keyword in lines[j]:
                                    block_type = keyword
                                    break
                        break
                break

        # Find the end of the block
        end = cursor_line
        brace_count = 0
        # Count braces from start to cursor
        for i in range(start, cursor_line + 1):
            brace_count += lines[i].count('{') - lines[i].count('}')

        # Continue counting braces until we reach the matching closing brace
        for i in range(cursor_line + 1, len(lines)):
            line = lines[i]
            brace_count += line.count('{') - line.count('}')
            if brace_count <= 0:
                end = i
                break
            end = i

    else:
        # Original logic for Python and JavaScript
        for i in range(cursor_line, -1, -1):
            line = lines[i].strip()
            if not line:
                continue
            line_indent = len(lines[i]) - len(lines[i].lstrip())
            if line_indent < cursor_indent:
                if re.match(patterns['block_start'], line):
                    start = i
                    keyword_match = re.match(r'^(\w+)', line)
                    if keyword_match and keyword_match.group(1) in patterns['keywords']:
                        block_type = keyword_match.group(1)
                    break

        # Find block end
        end = cursor_line
        if patterns['indent_based']:
            # For Python-like languages
            block_indent = len(lines[start]) - len(lines[start].lstrip())
            for i in range(start + 1, len(lines)):
                if lines[i].strip() and (len(lines[i]) - len(lines[i].lstrip())) <= block_indent:
                    end = i - 1
                    break
                end = i
        else:
            # For brace-based languages like JavaScript
            brace_count = 1
            for i in range(start + 1, len(lines)):
                line = lines[i]
                brace_count += line.count('{') - line.count('}')
                if brace_count <= 0:
                    end = i
                    break
                end = i

    # Debug information
    print(f"Block detected: Type={block_type}, Start={start}, End={end}")
    print(f"Block content preview: {lines[start][:50]}...{lines[end][-50:] if end < len(lines) else ''}")

    return start, end, block_type


# --------------------- Clipboard and Cursor Helpers --------------------- #


def get_text_and_line_number(x: int, y: int) -> Tuple[str, int]:
    """Get text from the entire editor and determine the line number at the given position."""
    temp_pos = pyautogui.position()

    # Move to position and ensure we're clicked in
    pyautogui.moveTo(x, y)
    pyautogui.click()
    time.sleep(CONFIG['CLICK_DELAY'] * 2)  # Double delay for stability

    # Ensure we're at the start of the current line
    pyautogui.press('home')
    time.sleep(CONFIG['CLICK_DELAY'])

    # Get all text first
    pyautogui.hotkey('ctrl', 'a')
    time.sleep(CONFIG['COPY_PASTE_DELAY'])
    pyautogui.hotkey('ctrl', 'c')
    full_text = pyperclip.paste()

    # Return to the line we were on
    pyautogui.moveTo(x, y)
    pyautogui.click()
    time.sleep(CONFIG['CLICK_DELAY'])

    # Get current line text
    pyautogui.press('home')
    time.sleep(CONFIG['CLICK_DELAY'])
    pyautogui.keyDown('shift')
    pyautogui.press('end')
    pyautogui.keyUp('shift')
    pyautogui.hotkey('ctrl', 'c')
    current_line = pyperclip.paste()

    # Find the line number by matching the current line
    lines = full_text.splitlines()
    line_number = None

    # Try to find exact match first
    for i, line in enumerate(lines):
        if line == current_line:
            line_number = i
            break

    # If no exact match, try matching stripped content
    if line_number is None:
        current_line_stripped = current_line.strip()
        for i, line in enumerate(lines):
            if line.strip() == current_line_stripped:
                line_number = i
                break

    # If still no match, default to 0
    if line_number is None:
        print("\nWarning: Could not determine line number accurately")
        line_number = 0

    # Return to original position
    pyautogui.moveTo(temp_pos.x, temp_pos.y)

    return full_text, line_number


def preview_code_block(x: int, y: int) -> str:
    """Preview the code block at the current position without selecting it."""
    try:
        # Store initial cursor position
        initial_pos = pyautogui.position()

        # Move to target position and click
        pyautogui.moveTo(x, y)
        pyautogui.click()
        time.sleep(CONFIG['CLICK_DELAY'])

        # Get all text in editor
        pyautogui.hotkey('ctrl', 'a')
        time.sleep(CONFIG['COPY_PASTE_DELAY'])
        pyautogui.hotkey('ctrl', 'c')
        full_text = pyperclip.paste()

        # Return to target position (not initial position yet)
        pyautogui.moveTo(x, y)
        pyautogui.click()
        time.sleep(CONFIG['CLICK_DELAY'])

        # Get current line
        pyautogui.keyDown('home')
        pyautogui.keyDown('shift')
        time.sleep(CONFIG['CLICK_DELAY'])
        pyautogui.press('end')
        pyautogui.keyUp('shift')
        pyautogui.keyUp('home')
        pyautogui.hotkey('ctrl', 'c')
        current_line = pyperclip.paste()

        # Find line number
        lines = full_text.splitlines()
        cursor_line = 0
        for i, line in enumerate(lines):
            if line.strip() == current_line.strip():
                cursor_line = i
                break

        # Detect language and get block boundaries
        language = detect_language(full_text)
        start, end, block_type = find_block_boundaries(lines, cursor_line, language)

        # Store the starting line number in state
        state.preview_line_number = start
        state.starting_line_number = start

        # Extract the block with original indentation
        block_text = '\n'.join(lines[start:end + 1])

        # Store the block with original indentation in state.out
        state.out = block_text

        # Return to initial cursor position
        pyautogui.moveTo(initial_pos.x, initial_pos.y)
        pyautogui.click()

        print(f"\nPreviewed block (starting at line {start}):")
        print("-" * 40)
        print(block_text)
        print("-" * 40)

        return block_text

    except Exception as e:
        print(f"\nError in preview_code_block: {str(e)}")
        return None


import cv2
import numpy as np
import mss
import time
import threading


class CaretTracker:
    """
    A class to detect and track text caret position on screen.
    Uses frame differencing to detect the blinking cursor.
    """

    def __init__(self):
        # Detection sensitivity
        self.DIFFERENCE_THRESHOLD = 1

        # Shape filtering
        self.MIN_CARET_WIDTH = 1
        self.MAX_CARET_WIDTH = 5
        self.MIN_CARET_HEIGHT = 10
        self.MAX_CARET_HEIGHT = 28
        self.MIN_ASPECT_RATIO = 2.5

        # Morphological operations
        self.DILATE_KERNEL_SIZE = (2, 5)
        self.DILATE_ITERATIONS = 1

        # Inter-frame delay for caret blinking detection
        self.INTER_FRAME_DELAY = 0.08  # Seconds between frame captures

        # Tracking state
        self.caret_position = None  # (x, y) or None
        self.is_tracking = False
        self.track_thread = None
        self.dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.DILATE_KERNEL_SIZE)

        # ROI settings
        self.USE_ROI = False
        self.ROI_X, self.ROI_Y, self.ROI_W, self.ROI_H = 0, 0, 0, 0

    def set_roi(self, x, y, width, height):
        """Set a region of interest for caret detection."""
        self.USE_ROI = True
        self.ROI_X, self.ROI_Y, self.ROI_W, self.ROI_H = x, y, width, height

    def disable_roi(self):
        """Disable region of interest tracking."""
        self.USE_ROI = False

    def start_tracking(self):
        """Start caret tracking in a separate thread."""
        if not self.is_tracking:
            self.is_tracking = True
            self.track_thread = threading.Thread(target=self._tracking_loop, daemon=True)
            self.track_thread.start()
            print("[CaretTracker] Started tracking.")

    def stop_tracking(self):
        """Stop caret tracking."""
        self.is_tracking = False
        if self.track_thread:
            self.track_thread.join(timeout=1.0)
            self.track_thread = None
        print("[CaretTracker] Stopped tracking.")

    def get_caret_position(self):
        """Get the current caret position."""
        return self.caret_position

    def _tracking_loop(self):
        """Main tracking loop running in a separate thread."""
        try:
            sct = mss.mss()
        except Exception as e:
            print(f"[CaretTracker] Error initializing screen capture: {e}")
            self.is_tracking = False
            return

        # Define monitor region and absolute offset
        absolute_offset_x = 0
        absolute_offset_y = 0
        monitor = None

        if self.USE_ROI:
            monitor = {"top": self.ROI_Y, "left": self.ROI_X, "width": self.ROI_W, "height": self.ROI_H}
            absolute_offset_x = self.ROI_X
            absolute_offset_y = self.ROI_Y
        else:
            try:
                # Use primary monitor
                monitor = sct.monitors[1]
                absolute_offset_x = monitor.get('left', 0)
                absolute_offset_y = monitor.get('top', 0)
                monitor = {"top": absolute_offset_y, "left": absolute_offset_x,
                           "width": monitor['width'], "height": monitor['height']}
            except Exception as e:
                print(f"[CaretTracker] Error getting monitor info: {e}")
                self.is_tracking = False
                return

        prev_gray = None
        last_capture_time = time.time()

        while self.is_tracking:
            current_time = time.time()
            # Frame rate control
            time_since_last = current_time - last_capture_time
            if time_since_last < self.INTER_FRAME_DELAY:
                sleep_duration = self.INTER_FRAME_DELAY - time_since_last
                if sleep_duration > 0:
                    time.sleep(sleep_duration)

            # Capture frame
            try:
                sct_img = sct.grab(monitor)
                last_capture_time = time.time()
                current_bgr = np.array(sct_img)
                current_gray = cv2.cvtColor(current_bgr, cv2.COLOR_BGRA2GRAY)
            except Exception as e:
                print(f"[CaretTracker] Error capturing screen: {e}")
                time.sleep(0.5)
                prev_gray = None
                continue

            if prev_gray is None:
                prev_gray = current_gray.copy()
                continue

            # Calculate difference
            frame_diff = cv2.absdiff(prev_gray, current_gray)

            # Threshold difference
            _, thresh_diff = cv2.threshold(frame_diff, self.DIFFERENCE_THRESHOLD, 255, cv2.THRESH_BINARY)

            # Morphological dilation
            dilated_diff = cv2.dilate(thresh_diff, self.dilate_kernel, iterations=self.DILATE_ITERATIONS)

            # Find contours
            contours, _ = cv2.findContours(dilated_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            potential_carets = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)

                # Filter contours based on shape
                is_size_ok = (self.MIN_CARET_WIDTH <= w <= self.MAX_CARET_WIDTH and
                              self.MIN_CARET_HEIGHT <= h <= self.MAX_CARET_HEIGHT)

                if is_size_ok:
                    aspect_ratio = float(h) / w if w > 0 else 0
                    is_aspect_ok = aspect_ratio >= self.MIN_ASPECT_RATIO
                    is_plausible_block = (aspect_ratio < self.MIN_ASPECT_RATIO and
                                          w >= self.MIN_CARET_WIDTH and h >= self.MIN_CARET_HEIGHT * 0.8)

                    if is_aspect_ok or is_plausible_block:
                        potential_carets.append((x, y, w, h))

            # Identify best caret candidate
            best_caret_rect = potential_carets[0] if potential_carets else None

            # Update caret position
            if best_caret_rect:
                x_rel, y_rel, w, h = best_caret_rect
                # Calculate center and ABSOLUTE screen coordinates
                center_x_abs = absolute_offset_x + x_rel + w // 2
                center_y_abs = absolute_offset_y + y_rel + h // 2
                self.caret_position = (center_x_abs, center_y_abs)
            else:
                self.caret_position = None

            # Update previous frame
            prev_gray = current_gray.copy()

        sct.close()
        print("[CaretTracker] Tracking loop ended.")


# Update the process_and_update_text function to integrate the CaretTracker class
import cv2
import numpy as np
import mss
import time
import threading


class CaretTracker:
    """
    A class to detect and track text caret position on screen.
    Uses frame differencing to detect the blinking cursor.
    """

    def __init__(self):
        # Increased sensitivity for better detection
        self.DIFFERENCE_THRESHOLD = 1

        # Adjusted shape filtering to better match typical carets
        self.MIN_CARET_WIDTH = 1
        self.MAX_CARET_WIDTH = 8  # Increased to catch wider carets
        self.MIN_CARET_HEIGHT = 8  # Lowered to catch shorter carets
        self.MAX_CARET_HEIGHT = 35  # Increased for larger fonts
        self.MIN_ASPECT_RATIO = 1.8  # Reduced to catch more block-type carets

        # Enhanced morphological operations
        self.DILATE_KERNEL_SIZE = (3, 7)  # Larger kernel to connect broken caret parts
        self.DILATE_ITERATIONS = 2  # More iterations for better connection

        # Adjusted timing for caret blinking detection
        self.INTER_FRAME_DELAY = 0.05  # Faster frame capture to ensure we catch the blink

        # Tracking state
        self.caret_position = None  # (x, y) or None
        self.is_tracking = False
        self.track_thread = None
        self.dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.DILATE_KERNEL_SIZE)

        # ROI settings
        self.USE_ROI = False
        self.ROI_X, self.ROI_Y, self.ROI_W, self.ROI_H = 0, 0, 0, 0

    def set_roi(self, x, y, width, height):
        """Set a region of interest for caret detection."""
        self.USE_ROI = True
        self.ROI_X, self.ROI_Y, self.ROI_W, self.ROI_H = x, y, width, height

    def disable_roi(self):
        """Disable region of interest tracking."""
        self.USE_ROI = False

    def start_tracking(self):
        """Start caret tracking in a separate thread."""
        if not self.is_tracking:
            self.is_tracking = True
            self.track_thread = threading.Thread(target=self._tracking_loop, daemon=True)
            self.track_thread.start()
            print("[CaretTracker] Started tracking.")

    def stop_tracking(self):
        """Stop caret tracking."""
        self.is_tracking = False
        if self.track_thread:
            self.track_thread.join(timeout=1.0)
            self.track_thread = None
        print("[CaretTracker] Stopped tracking.")

    def get_caret_position(self):
        """Get the current caret position."""
        return self.caret_position

    def _tracking_loop(self):
        """Main tracking loop running in a separate thread."""
        try:
            sct = mss.mss()
        except Exception as e:
            print(f"[CaretTracker] Error initializing screen capture: {e}")
            self.is_tracking = False
            return

        # Define monitor region and absolute offset
        absolute_offset_x = 0
        absolute_offset_y = 0
        monitor = None

        if self.USE_ROI:
            monitor = {"top": self.ROI_Y, "left": self.ROI_X, "width": self.ROI_W, "height": self.ROI_H}
            absolute_offset_x = self.ROI_X
            absolute_offset_y = self.ROI_Y
            print(f"[CaretTracker] Using ROI: {monitor}")
        else:
            try:
                # Use primary monitor
                monitor = sct.monitors[1]
                absolute_offset_x = monitor.get('left', 0)
                absolute_offset_y = monitor.get('top', 0)
                print(f"[CaretTracker] Using primary monitor: {monitor}")

                # If primary monitor is too large, reduce to a more reasonable size
                # to improve performance and accuracy for caret detection
                screen_width = monitor.get('width', 1920)
                screen_height = monitor.get('height', 1080)

                # Focus on the center area where text editing is likely happening
                center_x = screen_width // 2
                center_y = screen_height // 2
                roi_width = min(800, screen_width)
                roi_height = min(600, screen_height)

                # Calculate ROI coordinates
                roi_left = max(0, center_x - (roi_width // 2))
                roi_top = max(0, center_y - (roi_height // 2))

                monitor = {"top": roi_top, "left": roi_left,
                           "width": roi_width, "height": roi_height}
                absolute_offset_x = roi_left
                absolute_offset_y = roi_top
                print(f"[CaretTracker] Using optimized screen region: {monitor}")
            except Exception as e:
                print(f"[CaretTracker] Error getting monitor info: {e}")
                self.is_tracking = False
                return

        prev_gray = None
        last_capture_time = time.time()

        while self.is_tracking:
            current_time = time.time()
            # Frame rate control
            time_since_last = current_time - last_capture_time
            if time_since_last < self.INTER_FRAME_DELAY:
                sleep_duration = self.INTER_FRAME_DELAY - time_since_last
                if sleep_duration > 0:
                    time.sleep(sleep_duration)

            # Capture frame
            try:
                sct_img = sct.grab(monitor)
                last_capture_time = time.time()
                current_bgr = np.array(sct_img)
                current_gray = cv2.cvtColor(current_bgr, cv2.COLOR_BGRA2GRAY)
            except Exception as e:
                print(f"[CaretTracker] Error capturing screen: {e}")
                time.sleep(0.5)
                prev_gray = None
                continue

            if prev_gray is None:
                prev_gray = current_gray.copy()
                continue

            # Calculate difference
            frame_diff = cv2.absdiff(prev_gray, current_gray)

            # Threshold difference
            _, thresh_diff = cv2.threshold(frame_diff, self.DIFFERENCE_THRESHOLD, 255, cv2.THRESH_BINARY)

            # Morphological dilation
            dilated_diff = cv2.dilate(thresh_diff, self.dilate_kernel, iterations=self.DILATE_ITERATIONS)

            # Find contours
            contours, _ = cv2.findContours(dilated_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            potential_carets = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)

                # Filter contours based on shape
                is_size_ok = (self.MIN_CARET_WIDTH <= w <= self.MAX_CARET_WIDTH and
                              self.MIN_CARET_HEIGHT <= h <= self.MAX_CARET_HEIGHT)

                if is_size_ok:
                    aspect_ratio = float(h) / w if w > 0 else 0
                    is_aspect_ok = aspect_ratio >= self.MIN_ASPECT_RATIO
                    is_plausible_block = (aspect_ratio < self.MIN_ASPECT_RATIO and
                                          w >= self.MIN_CARET_WIDTH and h >= self.MIN_CARET_HEIGHT * 0.8)

                    if is_aspect_ok or is_plausible_block:
                        potential_carets.append((x, y, w, h))

            # Identify best caret candidate - prioritize by height/aspect ratio
            if potential_carets:
                # Sort by aspect ratio (height/width) descending to prefer tall, thin carets
                potential_carets.sort(key=lambda rect: float(rect[3]) / rect[2] if rect[2] > 0 else 0, reverse=True)

                # Debug information about candidates
                if len(potential_carets) > 0:
                    print(f"[CaretTracker] Found {len(potential_carets)} potential carets.")
                    for i, (x, y, w, h) in enumerate(potential_carets[:3]):  # Show top 3
                        aspect = float(h) / w if w > 0 else 0
                        print(f"[CaretTracker] Candidate {i + 1}: pos=({x},{y}), size={w}x{h}, aspect={aspect:.2f}")

                best_caret_rect = potential_carets[0]

                x_rel, y_rel, w, h = best_caret_rect
                # Calculate center and ABSOLUTE screen coordinates
                center_x_abs = absolute_offset_x + x_rel + w // 2
                center_y_abs = absolute_offset_y + y_rel + h // 2

                # If this is a new position, print debug info
                if self.caret_position != (center_x_abs, center_y_abs):
                    print(f"[CaretTracker] New caret position: ({center_x_abs}, {center_y_abs})")

                self.caret_position = (center_x_abs, center_y_abs)
            else:
                self.caret_position = None

            # Update previous frame
            prev_gray = current_gray.copy()

        sct.close()
        print("[CaretTracker] Tracking loop ended.")


# Update the process_and_update_text function to integrate the CaretTracker class
def double_click_and_drag_down(start_x, start_y, num_lines, line_height=20, steps=10, delay_between_steps=0.05):
    """
    Double-clicks at the specified position and then drags the mouse downward based on number of lines.

    Parameters:
    start_x, start_y: Starting coordinates for the double-click
    num_lines: Number of lines to select
    line_height: Average height of a single line in pixels
    steps: Number of incremental movements for the drag
    delay_between_steps: Time to wait between each step in seconds
    """
    # Calculate total distance based on line count and line height
    distance = num_lines * line_height

    # Move to the starting position
    pyautogui.moveTo(start_x, start_y)
    time.sleep(0.2)  # Short pause for stability

    print(f"Double-clicking at position ({start_x}, {start_y})")
    # Perform a double-click
    pyautogui.doubleClick()

    # If only selecting one line, we're done after double-click
    if num_lines <= 1:
        print(f"Selected single line with double-click")
        return

    # Short pause after double-click before starting the drag
    time.sleep(0.3)

    print(f"Starting drag to select {num_lines} lines ({distance} pixels)")
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
    print(f"Drag completed - selected approximately {num_lines} lines")


# Update the process_and_update_text function to use the double-click and drag functionality
import cv2
import numpy as np
import mss
import time
import threading

import cv2
import numpy as np
import mss
import time
import threading
import pyautogui


class CaretTracker:
    """
    A class to detect and track text caret position on screen.
    Uses frame differencing to detect the blinking cursor.
    """

    def __init__(self):
        # Debug mode
        self.DEBUG_MODE = False  # Set to True for detailed logging

        # Detection sensitivity
        self.DIFFERENCE_THRESHOLD = 1

        # Shape filtering - using parameters from first implementation
        self.MIN_CARET_WIDTH = 1
        self.MAX_CARET_WIDTH = 5
        self.MIN_CARET_HEIGHT = 10
        self.MAX_CARET_HEIGHT = 28
        self.MIN_ASPECT_RATIO = 2.5

        # Morphological operations - more conservative approach
        self.DILATE_KERNEL_SIZE = (2, 5)
        self.DILATE_ITERATIONS = 1

        # Inter-frame delay for caret blinking detection
        self.INTER_FRAME_DELAY = 0.08  # Seconds between frame captures

        # Tracking state
        self.caret_position = None  # (x, y) or None
        self.is_tracking = False
        self.track_thread = None
        self.dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.DILATE_KERNEL_SIZE)

        # ROI settings
        self.USE_ROI = False
        self.ROI_X, self.ROI_Y, self.ROI_W, self.ROI_H = 0, 0, 0, 0

        print(f"[CaretTracker] Initialized with parameters:")
        print(f"[CaretTracker] DIFFERENCE_THRESHOLD: {self.DIFFERENCE_THRESHOLD}")
        print(
            f"[CaretTracker] Shape filters: W=({self.MIN_CARET_WIDTH}-{self.MAX_CARET_WIDTH}), H=({self.MIN_CARET_HEIGHT}-{self.MAX_CARET_HEIGHT}), MinAspect={self.MIN_ASPECT_RATIO}")
        print(
            f"[CaretTracker] DILATE_KERNEL_SIZE: {self.DILATE_KERNEL_SIZE}, DILATE_ITERATIONS: {self.DILATE_ITERATIONS}")
        print(f"[CaretTracker] INTER_FRAME_DELAY: {self.INTER_FRAME_DELAY}")

    def set_roi(self, x, y, width, height):
        """Set a region of interest for caret detection."""
        self.USE_ROI = True
        self.ROI_X, self.ROI_Y, self.ROI_W, self.ROI_H = x, y, width, height
        print(f"[CaretTracker] Set ROI to: {x}, {y}, {width}x{height}")

    def disable_roi(self):
        """Disable region of interest tracking."""
        self.USE_ROI = False
        print("[CaretTracker] Disabled ROI")

    def start_tracking(self):
        """Start caret tracking in a separate thread."""
        if not self.is_tracking:
            self.is_tracking = True
            self.track_thread = threading.Thread(target=self._tracking_loop, daemon=True)
            self.track_thread.start()
            print("[CaretTracker] Started tracking.")

    def stop_tracking(self):
        """Stop caret tracking."""
        self.is_tracking = False
        if self.track_thread:
            self.track_thread.join(timeout=1.0)
            self.track_thread = None
        print("[CaretTracker] Stopped tracking.")

    def get_caret_position(self):
        """Get the current caret position."""
        return self.caret_position

    def _tracking_loop(self):
        """Main tracking loop running in a separate thread."""
        print("[CaretTracker] Starting tracking loop...")

        try:
            sct = mss.mss()
        except Exception as e:
            print(f"[CaretTracker] Error initializing screen capture: {e}")
            self.is_tracking = False
            return

        # Define monitor region and absolute offset
        absolute_offset_x = 0
        absolute_offset_y = 0
        monitor = None

        if self.USE_ROI:
            monitor = {"top": self.ROI_Y, "left": self.ROI_X, "width": self.ROI_W, "height": self.ROI_H}
            if self.ROI_W <= 0 or self.ROI_H <= 0:
                print(f"[CaretTracker] ERROR: Invalid ROI dimensions.")
                self.is_tracking = False
                return
            absolute_offset_x = self.ROI_X
            absolute_offset_y = self.ROI_Y
            print(f"[CaretTracker] Using ROI: {monitor}")
        else:
            try:
                # Use primary monitor. Assume its top-left is (0,0) for simplicity
                monitor = sct.monitors[1]
                absolute_offset_x = monitor.get('left', 0)
                absolute_offset_y = monitor.get('top', 0)
                print(f"[CaretTracker] Using Primary Monitor: {monitor}")
                monitor = {"top": absolute_offset_y, "left": absolute_offset_x,
                           "width": monitor['width'], "height": monitor['height']}
            except Exception as e:
                print(f"[CaretTracker] Error getting monitor info: {e}")
                self.is_tracking = False
                return

        prev_gray = None
        last_capture_time = time.time()

        while self.is_tracking:
            current_time = time.time()
            # Frame rate control
            time_since_last = current_time - last_capture_time
            if time_since_last < self.INTER_FRAME_DELAY:
                sleep_duration = self.INTER_FRAME_DELAY - time_since_last
                if sleep_duration > 0:
                    time.sleep(sleep_duration)

            # Capture frame
            try:
                sct_img = sct.grab(monitor)
                last_capture_time = time.time()
                current_bgr = np.array(sct_img)
                current_gray = cv2.cvtColor(current_bgr, cv2.COLOR_BGRA2GRAY)
            except mss.ScreenShotError as ex:
                print(f"[CaretTracker] ScreenShotError: {ex}. Retrying...")
                time.sleep(0.5)
                prev_gray = None
                continue
            except Exception as e:
                print(f"[CaretTracker] Error capturing screen: {e}")
                time.sleep(0.5)
                prev_gray = None
                continue

            if prev_gray is None:
                prev_gray = current_gray.copy()
                continue

            # Calculate difference
            frame_diff = cv2.absdiff(prev_gray, current_gray)

            # Threshold difference
            _, thresh_diff = cv2.threshold(frame_diff, self.DIFFERENCE_THRESHOLD, 255, cv2.THRESH_BINARY)

            # Morphological dilation
            dilated_diff = cv2.dilate(thresh_diff, self.dilate_kernel, iterations=self.DILATE_ITERATIONS)

            # Find contours
            contours, _ = cv2.findContours(dilated_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            potential_carets = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)

                # Filter contours based on shape
                is_size_ok = (self.MIN_CARET_WIDTH <= w <= self.MAX_CARET_WIDTH and
                              self.MIN_CARET_HEIGHT <= h <= self.MAX_CARET_HEIGHT)

                if is_size_ok:
                    aspect_ratio = float(h) / w if w > 0 else 0
                    is_aspect_ok = aspect_ratio >= self.MIN_ASPECT_RATIO
                    is_plausible_block = (aspect_ratio < self.MIN_ASPECT_RATIO and
                                          w >= self.MIN_CARET_WIDTH and h >= self.MIN_CARET_HEIGHT * 0.8)

                    if is_aspect_ok or is_plausible_block:
                        potential_carets.append((x, y, w, h))
                        if self.DEBUG_MODE:
                            abs_x_debug = absolute_offset_x + x
                            abs_y_debug = absolute_offset_y + y
                            print(
                                f"[CaretTracker] Candidate @({abs_x_debug},{abs_y_debug}) Rel({x},{y}) Size: {w}x{h} Aspect: {aspect_ratio:.2f}")

            # Identify best caret candidate - simplified selection
            best_caret_rect = potential_carets[0] if potential_carets else None

            # Update caret position
            if best_caret_rect:
                x_rel, y_rel, w, h = best_caret_rect
                # Calculate center and ABSOLUTE screen coordinates
                center_x_abs = absolute_offset_x + x_rel + w // 2
                center_y_abs = absolute_offset_y + y_rel + h // 2

                # Only log if position changed significantly
                if self.caret_position is None or abs(self.caret_position[0] - center_x_abs) > 2 or abs(
                        self.caret_position[1] - center_y_abs) > 2:
                    print(f"[CaretTracker] Detected caret at: ({center_x_abs}, {center_y_abs})")

                self.caret_position = (center_x_abs, center_y_abs)
            else:
                self.caret_position = None

            # Update previous frame
            prev_gray = current_gray.copy()

        sct.close()
        print("[CaretTracker] Tracking loop ended.")

    def double_click_and_drag_down(self, start_x, start_y, num_lines, line_height=20, steps=10,
                                   delay_between_steps=0.05):
        """
        Double-clicks at the specified position and then drags the mouse downward based on number of lines.

        Parameters:
        start_x, start_y: Starting coordinates for the double-click
        num_lines: Number of lines to select
        line_height: Average height of a single line in pixels
        steps: Number of incremental movements for the drag
        delay_between_steps: Time to wait between each step in seconds
        """
        # Calculate total distance based on line count and line height
        distance = num_lines * line_height

        # Move to the starting position
        pyautogui.moveTo(start_x, start_y)
        time.sleep(0.2)  # Short pause for stability

        print(f"[CaretTracker] Double-clicking at position ({start_x}, {start_y})")
        # Perform a double-click
        pyautogui.doubleClick()

        # If only selecting one line, we're done after double-click
        if num_lines <= 1:
            print(f"[CaretTracker] Selected single line with double-click")
            return

        # Short pause after double-click before starting the drag
        time.sleep(0.3)

        print(f"[CaretTracker] Starting drag to select {num_lines} lines ({distance} pixels)")
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
        print(f"[CaretTracker] Drag completed - selected approximately {num_lines} lines")
# Update the process_and_update_text function to integrate the CaretTracker class
def process_and_update_text():
    """Process text with line-by-line replacement while maintaining exact spaces and handling line insertions."""
    global state
    try:
        if not all([state.insertion_point, state.starting_line_number, state.out, state.chat_app_position]):
            print("Error: Missing required state information")
            return

        # Save current cursor position
        temp_pos = pyautogui.position()

        # Move to chat app to retrieve current text
        pyautogui.moveTo(state.chat_app_position[0], state.chat_app_position[1])
        pyautogui.click()
        time.sleep(CONFIG['CLICK_DELAY'])
        pyautogui.hotkey('ctrl', 'a')
        time.sleep(CONFIG['COPY_PASTE_DELAY'])
        pyautogui.hotkey('ctrl', 'c')
        current_text = pyperclip.paste()

        # Store initial line count on first run
        if state.initial_line_count == 0:
            state.initial_line_count = len(state.out.split('\n'))
            state.sent_lines_count = state.initial_line_count

        # Prepare input for the LLM
        input_text = (f"Previous text:\n{state.out}\n\nCurrent text:\n{current_text}\n\n"
                      f"Starting line number: {state.starting_line_number}\n\n"
                      f"Initial line count: {state.initial_line_count}")
        print("input: " + input_text)
        client = anthropic.Anthropic(api_key=CONFIG['API_KEY'])
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": (f"{input_text}\n\nPlease provide ONLY the modified code while "
                            "maintaining exact indentation and spacing. Do not include any explanations, comments, or additional text.")
            }]
        )

        # Print the Claude response for debugging
        print("\nClaude Response:")
        print("-" * 40)
        print(response.content[0].text)
        print("-" * 40)

        # Extract response and remove any code block markers if present
        new_text = response.content[0].text
        if new_text.startswith('```'):
            language_marker = new_text.split('\n')[0].replace('```', '').strip()
            lines = new_text.split('\n')[1:]
            if lines and lines[-1].startswith('```'):
                lines = lines[:-1]
            # Remove any trailing empty lines
            while lines and not lines[-1].strip():
                lines.pop()
            new_text = '\n'.join(lines)

            # Format the code based on language
            try:
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=f'.{language_marker}') as temp_file:
                    temp_file.write(new_text)
                    temp_path = temp_file.name

                # Clean up temp file
                os.unlink(temp_path)

            except Exception as format_error:
                print(f"Formatting error (continuing with unformatted code): {format_error}")

        # Store the LLM response for pasting after selection
        paste_text = new_text

        # Print line count information
        print(f"Sent Lines: {state.sent_lines_count}")
        llm_lines_count = len(paste_text.splitlines())
        state.llm_lines_count = llm_lines_count
        print(f"LLM Lines: {state.llm_lines_count}")

        # Move to the insertion point
        pyautogui.moveTo(state.insertion_point[0], state.insertion_point[1])
        pyautogui.click()
        time.sleep(CONFIG['CLICK_DELAY'])

        # Calculate current cursor position's line number
        current_pos = pyautogui.position()
        _, current_line_num = get_text_and_line_number(current_pos.x, current_pos.y)

        # Move to the starting line
        lines_to_move = state.starting_line_number - current_line_num
        if lines_to_move > 0:
            for _ in range(lines_to_move):
                pyautogui.press('down')
                time.sleep(0.05)
        else:
            for _ in range(abs(lines_to_move)):
                pyautogui.press('up')
                time.sleep(0.05)

        # We should now be at the starting line
        # Position cursor at beginning of line
        pyautogui.press('home')
        time.sleep(0.2)

        # Initialize AccurateCaretTracker if not already done
        if not hasattr(state, 'caret_tracker'):
            print("Creating new AccurateCaretTracker instance")
            state.caret_tracker = AccurateCaretTracker(debug_mode=True)

            # Set a focused ROI around the current cursor position
            current_x, current_y = pyautogui.position()
            roi_x = max(0, current_x - 200)
            roi_y = max(0, current_y - 100)
            state.caret_tracker.set_roi(roi_x, roi_y, 400, 200)

            state.caret_tracker.start_tracking()
            print(f"Started caret tracker with ROI around current position")
            time.sleep(0.5)  # Give tracker time to initialize

        # *** CRITICAL CHANGE: Make caret visible WITHOUT moving cursor first ***
        # This is the key - force the caret to appear on the current line without moving
        print("Making caret visible on the first line that needs changes")

        # Make a minimal movement to trigger caret visibility
        pyautogui.press('right')
        time.sleep(0.1)
        pyautogui.press('left')
        time.sleep(0.3)  # Wait for caret to become visible

        # Now try to get caret position from tracker
        caret_pos = state.caret_tracker.get_caret_position()

        current_x, current_y = pyautogui.position()  # Store current position

        if not caret_pos:
            # If caret not detected, try force appearance without moving to a different position
            print("Caret not detected, using force_caret_appearance on current line")
            state.caret_tracker.force_caret_appearance(current_x, current_y)
            time.sleep(0.3)
            caret_pos = state.caret_tracker.get_caret_position()

        # Get position for selection - use detected caret or current position
        if caret_pos:
            print(f"Using detected caret position: {caret_pos}")
            selection_x, selection_y = caret_pos
        else:
            print(f"Caret not detected, using current position: ({current_x}, {current_y})")
            selection_x, selection_y = current_x, current_y

        # Calculate the number of lines to select
        lines_to_select = state.initial_line_count
        line_height = CONFIG.get('LINE_HEIGHT', 20)

        print(f"Starting double-click and drag to select {lines_to_select} lines")

        # Use the AccurateCaretTracker's double_click_and_drag_down method
        state.caret_tracker.double_click_and_drag_down(
            selection_x, selection_y, lines_to_select, line_height
        )

        # Verify the selection by copying
        time.sleep(0.3)
        pyautogui.hotkey('ctrl', 'c')
        time.sleep(0.2)
        selected_text = pyperclip.paste()
        selected_lines = len(selected_text.splitlines())
        print(f"Selected {selected_lines} lines (expected {lines_to_select})")

        # Now paste the LLM response over the selection
        pyperclip.copy(paste_text)
        time.sleep(0.2)
        pyautogui.hotkey('ctrl', 'v')
        time.sleep(CONFIG['COPY_PASTE_DELAY'])
        print(f"Pasted {llm_lines_count} lines over the selected {selected_lines} lines")

        # Return to original position
        pyautogui.moveTo(temp_pos.x, temp_pos.y)

    except Exception as e:
        print(f"Error in process_and_update_text: {e}")
        import traceback
        traceback.print_exc()

        # Restore cursor position on error
        pyautogui.moveTo(temp_pos.x, temp_pos.y)

def extract_code(response: str) -> str:
    """Extract code from LLM response."""
    match = re.search(r'```(?:python)?\s*(.*?)\s*```', response, re.DOTALL)
    return match.group(1).strip() if match else response.strip()


def get_active_window_at_cursor(x: int, y: int):
    """Get active window at cursor position."""
    windows = gw.getAllWindows()
    for window in windows:
        if (window.visible and
                window.left <= x <= window.left + window.width and
                window.top <= y <= window.top + window.height):
            return window
    return None


# --------------------- Event Handlers --------------------- #

def on_click(x, y, button, pressed):
    """Handle mouse clicks."""
    global state
    if pressed:
        window = get_active_window_at_cursor(x, y)
        # If clicking in a window with "chat app" or "text box only" in the title
        if window and ("chat" in window.title.lower() or "text box only" in window.title.lower()):
            if not state.chat_app_click_recorded:
                state.chat_app_position = (x, y)
                state.chat_app_click_recorded = True
            else:
                window_right = window.left + window.width
                if window_right - 30 <= x <= window_right:
                    process_and_update_text()
        else:
            if state.first_run:
                state.first_run = False
                state.starting_line_number = None  # Reset starting line number
                state.insertion_point = (x, y)  # Store the first click position
            else:
                state.insertion_point = (x, y)


def predefined_block_rewrite():
    """
    Rewrite code block when no text is explicitly selected.
    Uses a predefined strategy to identify and modify code.
    """
    global state
    try:
        # Use current cursor position to determine context
        current_pos = pyautogui.position()
        full_text, cursor_line = get_text_and_line_number(current_pos.x, current_pos.y)

        # Detect language of the current file
        language = detect_language(full_text)
        lines = full_text.splitlines()

        # Find the nearest meaningful block (function, class, or logical block)
        start, end, block_type = find_block_boundaries(lines, cursor_line, language)

        # Store context for later processing
        state.starting_line_number = start
        state.stored_chunk = "\n".join(lines[start:end + 1])

        # Prepare input for LLM
        input_text = (f"Language: {language}\n"
                      f"Block Type: {block_type}\n"
                      f"Current Block:\n{state.stored_chunk}\n\n"
                      "Please provide a refined or improved version of this code block.")

        # Use Anthropic API to generate improved code
        client = anthropic.Anthropic(api_key=CONFIG['API_KEY'])
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": input_text
            }]
        )

        # Extract and clean the code response
        new_text = response.content[0].text
        new_text = extract_code(new_text)

        # Update the text in-place
        pyautogui.moveTo(current_pos)
        state.insertion_point = current_pos
        state.stored_chunk = new_text

        # Trigger text replacement
        process_and_update_text()

    except Exception as e:
        print(f"Error in predefined block rewrite: {e}")
        import traceback
        traceback.print_exc()


# Modify the on_key_press function to include predefined block rewrite
def on_key_press(key):
    """Handle keyboard input with preview functionality."""
    global state
    if key == keyboard.Key.f2:
        process_and_update_text()
    elif key == keyboard.Key.left:
        # First preview/determine the code block
        current_pos = pyautogui.position()
        preview_text = preview_code_block(current_pos.x, current_pos.y)

    elif key == keyboard.Key.right:
        current_pos = pyautogui.position()
        preview_code_block(current_pos.x, current_pos.y)
        time.sleep(0.5)
        predefined_block_rewrite()


def get_selected_code_block(current_position: tuple, text: str, line_number: int) -> tuple:
    """
    Get the currently selected code block or use predefined block detection.

    Args:
        current_position: (x, y) coordinates of cursor
        text: Full text content
        line_number: Current line number

    Returns:
        tuple: (selected_text, start_line, end_line, is_predefined)
    """
    # First try to get manually selected text
    pyautogui.moveTo(current_position[0], current_position[1])
    pyautogui.hotkey('ctrl', 'c')
    time.sleep(0.2)
    selected_text = pyperclip.paste()

    # If no text is selected, use predefined block detection
    if not selected_text.strip():
        lines = text.splitlines()
        language = detect_language(text)
        start, end, _ = find_block_boundaries(lines, line_number, language)
        selected_text = '\n'.join(lines[start:end + 1])
        return selected_text, start, end, True

    # For manually selected text, find its boundaries
    full_lines = text.splitlines()
    start_line = None
    end_line = None

    # Find the boundaries of selected text
    selected_lines = selected_text.splitlines()
    for i, line in enumerate(full_lines):
        if line.strip() == selected_lines[0].strip():
            start_line = i
            if len(selected_lines) == 1:
                end_line = i
                break
        elif start_line is not None and line.strip() == selected_lines[-1].strip():
            end_line = i
            break

    return selected_text, start_line, end_line, False


def compare_code_blocks(original: str, modified: str) -> bool:
    """
    Compare original and modified code blocks to determine if they're different.

    Args:
        original: Original code block
        modified: Modified code block

    Returns:
        bool: True if blocks are different, False if they're the same
    """

    # Normalize both texts by removing extra whitespace and empty lines
    def normalize(text):
        return '\n'.join(line.strip() for line in text.splitlines() if line.strip())

    original_normalized = normalize(original)
    modified_normalized = normalize(modified)

    # Use difflib to compute similarity ratio
    similarity = difflib.SequenceMatcher(None, original_normalized, modified_normalized).ratio()

    # Return True if texts are different (similarity < 0.95)
    return similarity < 0.95


def handle_code_selection(current_position: tuple, text: str, line_number: int) -> tuple:
    """
    Main handler for code selection and comparison.

    Args:
        current_position: (x, y) coordinates of cursor
        text: Full text content
        line_number: Current line number

    Returns:
        tuple: (code_block, start_line, end_line, should_use_predefined)
    """
    # Get current selection or block
    selected_code, start, end, is_predefined = get_selected_code_block(current_position, text, line_number)

    # If using predefined block, always use it
    if is_predefined:
        return selected_code, start, end, True

    # Store the previous selection in temporary storage
    temp_file = ".temp_selection"
    previous_selection = ""
    if os.path.exists(temp_file):
        with open(temp_file, 'r') as f:
            previous_selection = f.read()

    # Compare current selection with previous
    if compare_code_blocks(previous_selection, selected_code):
        # Blocks are different, use current selection
        with open(temp_file, 'w') as f:
            f.write(selected_code)
        return selected_code, start, end, False
    else:
        # Blocks are same, use predefined block detection
        lines = text.splitlines()
        language = detect_language(text)
        start, end, _ = find_block_boundaries(lines, line_number, language)
        predefined_block = '\n'.join(lines[start:end + 1])
        return predefined_block, start, end, True


# --------------------- TCP Server --------------------- #

def tcp_server():
    global state
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        # Add this line to allow socket reuse
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((CONFIG['SERVER_HOST'], CONFIG['SERVER_PORT']))
        server_socket.listen()

        while True:
            client_socket, addr = server_socket.accept()
            with client_socket:
                while True:
                    data = client_socket.recv(1024)
                    if not data:
                        break
                    if data.decode('utf-8') == "get_cursor_position":
                        response = str(state.insertion_point)
                        client_socket.sendall(response.encode('utf-8'))


# --------------------- Main --------------------- #

def main():
    global state
    state = State()

    server_thread = threading.Thread(target=tcp_server, daemon=True)
    server_thread.start()

    mouse_listener = mouse.Listener(on_click=on_click)
    mouse_listener.start()

    keyboard_listener = keyboard.Listener(on_press=on_key_press)
    keyboard_listener.start()

    print("Application started:")
    print("- First click sets starting line")
    print("- Second click sets insertion point")
    print("- Press F2 to update code")
    print("- Press Left Arrow to select current code block")
    print("- Press Ctrl+C to exit")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        mouse_listener.stop()
        keyboard_listener.stop()
        print("Application terminated.")


if __name__ == "__main__":
    main()