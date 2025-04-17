import tkinter as tk
import sys
import os
import time
import threading
import traceback
import re
import keyboard
import pyperclip
from pynput import mouse, keyboard as kb
import pyautogui
import anthropic

# Platform detection
if sys.platform == 'win32':
    try:
        import pygetwindow as gw
        import win32gui
        import win32con

        PLATFORM = "windows"
    except ImportError:
        PLATFORM = "limited"
elif sys.platform == 'darwin':
    try:
        import pygetwindow as gw
        import AppKit

        PLATFORM = "mac"
    except ImportError:
        PLATFORM = "limited"
else:
    try:
        import Xlib.display

        PLATFORM = "linux"
    except ImportError:
        PLATFORM = "limited"

# Configuration
CONFIG = {

    'CLICK_DELAY': 0.1,
    'COPY_PASTE_DELAY': 0.2,
    'LINE_HEIGHT': 20
}


# Global state
class State:
    def __init__(self):
        self.last_active_window = None
        self.last_active_window_title = None
        self.insertion_point = None
        self.last_inserted_text = None
        self.starting_line_number = None
        self.out = None  # Current code block content


state = State()


# --------------------- Language and Block Detection --------------------- #

def detect_language(code: str) -> str:
    """Detect programming language from code."""
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


def get_block_info(language: str):
    """Get language-specific block patterns."""
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


def find_block_boundaries(lines, cursor_line, language):
    """Find the start and end of the current code block."""
    patterns = get_block_info(language)
    print(f"Language detected: {language}")

    # Handle empty file or invalid cursor position
    if not lines or cursor_line >= len(lines):
        return 0, 0, 'unknown'

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
            if i >= len(lines):
                continue

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
    if start < len(lines) and end < len(lines):
        print(f"Block content preview: {lines[start][:50]}...{lines[end][-50:] if end < len(lines) else ''}")

    return start, end, block_type


# --------------------- Clipboard and Cursor Helpers --------------------- #

def get_text_and_line_number(x, y):
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

    return full_text, line_number, lines


def preview_code_block(x, y):
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

        return block_text, start, end, language, block_type

    except Exception as e:
        print(f"\nError in preview_code_block: {str(e)}")
        return None, None, None, None, None


def select_and_process_code_block(x, y, input_text):
    """Select code block, process with LLM and user's input text, and replace it."""
    try:
        # First preview the code block to get information
        block_text, start, end, language, block_type = preview_code_block(x, y)

        if not block_text:
            print("No code block detected")
            return False

        # Calculate number of lines in code block
        lines_to_select = end - start + 1
        print(f"Found code block with {lines_to_select} lines")

        # Process block with LLM and input text
        processed_text = process_code_with_llm(block_text, input_text, language, block_type)

        # Now select the code block
        # Move to the start line
        pyautogui.moveTo(x, y)
        pyautogui.click()
        time.sleep(CONFIG['CLICK_DELAY'])

        # Get current position's line number
        full_text, current_line, _ = get_text_and_line_number(x, y)

        # Move to the starting line of the block
        lines_to_move = start - current_line
        if lines_to_move > 0:
            for _ in range(lines_to_move):
                pyautogui.press('down')
                time.sleep(0.05)
        else:
            for _ in range(abs(lines_to_move)):
                pyautogui.press('up')
                time.sleep(0.05)

        # Position cursor at beginning of line
        pyautogui.press('home')
        time.sleep(0.2)

        # Current position for selection
        current_x, current_y = pyautogui.position()

        # Double-click and drag to select the block
        # Perform a double-click to select the first line
        pyautogui.doubleClick()
        time.sleep(0.3)  # Wait for selection to register

        # If more than one line, drag to select additional lines
        if lines_to_select > 1:
            # Press and hold the left mouse button
            pyautogui.mouseDown()

            # Calculate distance to drag
            distance = (lines_to_select - 1) * CONFIG.get('LINE_HEIGHT', 20)

            # Number of steps for a smoother drag
            steps = 10
            step_distance = distance / steps

            # Perform the drag operation in steps
            for i in range(1, steps + 1):
                # Calculate new y position for this step
                current_y_step = current_y + (step_distance * i)

                # Move to new position while holding button
                pyautogui.moveTo(current_x, current_y_step)

                # Small pause between steps
                time.sleep(0.05)

            # Release the mouse button
            pyautogui.mouseUp()

        # Verify the selection by copying
        time.sleep(0.3)
        pyautogui.hotkey('ctrl', 'c')
        time.sleep(0.2)
        selected_text = pyperclip.paste()
        selected_lines = len(selected_text.splitlines())
        print(f"Selected {selected_lines} lines (expected {lines_to_select})")

        # Now paste the processed text over the selection
        pyperclip.copy(processed_text)
        time.sleep(0.2)
        pyautogui.hotkey('ctrl', 'v')
        time.sleep(CONFIG['COPY_PASTE_DELAY'])
        print(f"Replaced code block with processed text")

        return True
    except Exception as e:
        print(f"Error processing code block: {e}")
        traceback.print_exc()
        return False


def process_code_with_llm(code_block, input_text, language, block_type):
    """Process code block with LLM using both the code and user input."""
    try:
        client = anthropic.Anthropic(api_key=CONFIG['API_KEY'])
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": (
                    f"Language: {language}\n"
                    f"Block Type: {block_type}\n"
                    f"Current code block:\n{code_block}\n\n"
                    f"User requested modification: {input_text}\n\n"
                    "Please modify the code block according to the user's request. "
                    "Return only the modified code block with the same indentation, no explanations."
                )
            }]
        )
        processed_text = response.content[0].text

        # Remove code block markers if present
        if processed_text.startswith('```'):
            lines = processed_text.split('\n')
            start_idx = 1  # Skip the first line with ```
            if lines[0].strip() != '```':
                # Handle language marker in the first line
                start_idx = 1
            end_idx = len(lines)
            for i, line in enumerate(lines):
                if i > 0 and line.strip() == '```':
                    end_idx = i
                    break
            processed_text = '\n'.join(lines[start_idx:end_idx])

        return processed_text
    except Exception as e:
        print(f"Error processing with LLM: {e}")
        return code_block  # Return original code block on error


class OverlayTextApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Text Overlay")
        self.setup_ui()

        # Initialize state variables
        self.last_active_window = None
        self.last_active_window_title = None
        self.is_visible = False
        self.debug_mode = True  # Set to True to see debug information
        self.last_inserted_text = None

        # Start monitors and set up hooks
        self.setup_keyboard_hook()
        self.start_monitoring_thread()

        # Log startup
        self.log("Application started")
        self.log(f"Running on: {PLATFORM} platform")

    def setup_ui(self):
        """Set up the user interface"""
        # Configure the main window
        self.root.attributes('-topmost', True)
        self.root.attributes('-alpha', 0.9)

        # On Windows, we can make a tool window
        if sys.platform == 'win32':
            self.root.attributes("-toolwindow", True)

        # Size and position
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = 400
        window_height = 170
        position_x = (screen_width - window_width) // 2
        position_y = (screen_height - window_height) // 3
        self.root.geometry(f"{window_width}x{window_height}+{position_x}+{position_y}")

        # Frame for content
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Instructions label
        instructions = tk.Label(main_frame, text="Type text to insert at last cursor position:")
        instructions.pack(pady=(0, 5))

        # Text entry
        self.text_entry = tk.Entry(main_frame, width=40, font=("Arial", 12))
        self.text_entry.pack(fill=tk.X, pady=5)
        self.text_entry.bind("<Return>", self.insert_text_and_process)
        self.text_entry.bind("<Escape>", lambda e: self.hide_overlay())

        # Buttons frame
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=5)

        # Insert button
        self.insert_button = tk.Button(button_frame, text="Insert Text",
                                       command=self.insert_text_and_process)
        self.insert_button.pack(side=tk.LEFT, padx=5)

        # Cancel button
        self.cancel_button = tk.Button(button_frame, text="Cancel",
                                       command=self.hide_overlay)
        self.cancel_button.pack(side=tk.LEFT, padx=5)

        # Status frame
        status_frame = tk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=5)

        # Status label
        self.status_label = tk.Label(status_frame, text="Ready", fg="gray",
                                     anchor="w", font=("Arial", 9))
        self.status_label.pack(fill=tk.X)

        # Information label
        info_label = tk.Label(status_frame, text="Text will be inserted, then code will be processed",
                              fg="blue", font=("Arial", 8))
        info_label.pack(side=tk.BOTTOM)

        # Initially hide the window
        self.root.withdraw()

    def setup_keyboard_hook(self):
        """Set up keyboard hooks for activation"""
        try:
            # For global hotkey activation
            keyboard.add_hotkey('ctrl+shift+t', self.show_overlay)

            # Start keyboard listener to track typing activity
            self.kb_listener = kb.Listener(on_press=self.on_key_press)
            self.kb_listener.daemon = True  # Allow the program to exit
            self.kb_listener.start()

            self.log("Keyboard hooks set up successfully")
        except Exception as e:
            self.log(f"Failed to set up keyboard hooks: {e}", error=True)

    def start_monitoring_thread(self):
        """Start thread to monitor active window"""
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self.monitor_active_window, daemon=True)
        self.monitor_thread.start()
        self.log("Window monitoring thread started")

    def monitor_active_window(self):
        """Monitor and record the active window information"""
        while self.monitoring_active:
            try:
                # Platform-specific active window detection
                if PLATFORM == "windows":
                    hwnd = win32gui.GetForegroundWindow()
                    if hwnd:
                        window_title = win32gui.GetWindowText(hwnd)
                        if window_title and window_title != "Text Overlay":
                            self.last_active_window = hwnd
                            self.last_active_window_title = window_title
                            # Update global state
                            global state
                            state.last_active_window = hwnd
                            state.last_active_window_title = window_title
                            self.update_status(f"Last window: {window_title[:30]}...")

                elif PLATFORM == "mac":
                    # macOS approach - if pygetwindow is working
                    try:
                        active_window = gw.getActiveWindow()
                        if active_window and active_window.title != "Text Overlay":
                            self.last_active_window = active_window
                            self.last_active_window_title = active_window.title
                            # Update global state
                            state.last_active_window = active_window
                            state.last_active_window_title = active_window.title
                            self.update_status(f"Last window: {active_window.title[:30]}...")
                    except:
                        # Fallback using AppKit
                        app = AppKit.NSWorkspace.sharedWorkspace().activeApplication()
                        if app:
                            app_name = app['NSApplicationName']
                            self.last_active_window_title = app_name
                            state.last_active_window_title = app_name
                            self.update_status(f"Last window: {app_name}")

                elif PLATFORM == "linux":
                    # Basic X11 approach
                    display = Xlib.display.Display()
                    window = display.get_input_focus().focus
                    wmname = window.get_wm_name()
                    if wmname and "Text Overlay" not in wmname:
                        self.last_active_window = window
                        self.last_active_window_title = wmname
                        # Update global state
                        state.last_active_window = window
                        state.last_active_window_title = wmname
                        self.update_status(f"Last window: {wmname[:30]}...")

                elif PLATFORM == "limited":
                    # Limited mode - just track if our window is active or not
                    if not self.is_visible:
                        self.last_active_window_title = "Last application"
                        state.last_active_window_title = "Last application"

            except Exception as e:
                self.log(f"Error monitoring window: {e}", error=True)

            time.sleep(0.5)  # Check every half second

    def on_key_press(self, key):
        """Track key presses to infer cursor position"""
        # Only track when our window is not visible and in a tracked window
        if not self.is_visible and self.last_active_window_title:
            # We don't need to do anything special here, just knowing user was typing
            # in the last active window is enough for our simplified approach
            pass

    def show_overlay(self):
        """Show the overlay window"""
        if not self.is_visible:
            try:
                # Save the current active window before we take focus
                self.update_status("Ready to insert text")

                # Show the window
                self.root.deiconify()
                self.is_visible = True

                # Reset and focus text entry
                self.text_entry.delete(0, tk.END)
                self.text_entry.focus_set()

                self.log("Overlay shown")
            except Exception as e:
                self.log(f"Error showing overlay: {e}", error=True)

    def hide_overlay(self):
        """Hide the overlay window"""
        if self.is_visible:
            self.root.withdraw()
            self.is_visible = False
            self.log("Overlay hidden")

    def insert_text_and_process(self, event=None):
        """First insert text at cursor, then process code block with input text."""
        input_text = self.text_entry.get().strip()
        if not input_text:
            self.hide_overlay()
            return

        if not self.last_active_window_title:
            self.update_status("No previous window detected", error=True)
            return

        self.log(f"Got input text: {input_text}")

        # Save original clipboard content
        original_clipboard = pyperclip.paste()

        try:
            # Hide our window first
            self.hide_overlay()
            time.sleep(0.3)  # Wait briefly

            # Save current cursor position
            current_pos = pyautogui.position()

            # Update global state
            self.last_inserted_text = input_text
            global state
            state.last_inserted_text = input_text

            # Copy text to clipboard for insertion
            pyperclip.copy(input_text)

            # Platform specific window activation (NO CLICKING)
            if PLATFORM == "windows":
                try:
                    # Try to activate the window
                    win32gui.SetForegroundWindow(self.last_active_window)
                    time.sleep(0.3)  # Allow time for focus
                except Exception as e:
                    self.log(f"Failed to focus window: {e}", error=True)

            elif PLATFORM == "mac":
                try:
                    # Try pygetwindow approach first
                    if hasattr(self.last_active_window, 'activate'):
                        self.last_active_window.activate()
                    else:
                        # Try AppleScript as fallback
                        os.system(f"osascript -e 'tell application \"{self.last_active_window_title}\" to activate'")
                    time.sleep(0.3)
                except Exception as e:
                    self.log(f"Failed to focus window: {e}", error=True)

            # Wait a moment for window to activate
            time.sleep(0.3)

            # STEP 1: Insert the text at the cursor position - NO CLICKING FIRST
            # Just paste directly (this preserves the last caret position)
            if sys.platform == 'darwin':
                pyautogui.hotkey('command', 'v')
            else:
                pyautogui.hotkey('ctrl', 'v')

            time.sleep(0.5)  # Wait for paste to complete

            self.log("Text inserted successfully")

            # STEP 2: Now select and process the code block (current position will be right after the inserted text)
            time.sleep(0.5)  # Wait a moment after insertion

            # Process the code block
            success = select_and_process_code_block(current_pos.x, current_pos.y, input_text)

            if success:
                self.log("Code processed successfully")
                self.update_status("Text inserted and code modified successfully")
            else:
                self.log("Failed to process code block", error=True)
                self.update_status("Text inserted but failed to process code", error=True)

            # Restore original clipboard after a brief pause
            time.sleep(0.3)
            pyperclip.copy(original_clipboard)

        except Exception as e:
            self.log(f"Error during text insertion and processing: {e}", error=True)
            self.show_error(f"Failed to insert text: {str(e)}")

            # Restore original clipboard
            pyperclip.copy(original_clipboard)

    def show_error(self, message):
        """Show error message in overlay"""
        self.root.deiconify()
        self.is_visible = True
        self.update_status(message, error=True)

    def update_status(self, message, error=False):
        """Update status label with message"""
        if not hasattr(self, 'status_label'):
            return

        color = "red" if error else "gray"
        try:
            self.status_label.config(text=message, fg=color)
            # Force UI update
            self.root.update_idletasks()
        except:
            pass  # Ignore if window is being destroyed

    def log(self, message, error=False):
        """Log message for debugging"""
        if self.debug_mode:
            prefix = "ERROR: " if error else "INFO: "
            print(f"{prefix}{message}")
            if error:
                traceback.print_exc()


def on_click(x, y, button, pressed):
    """Handle mouse clicks."""
    global state
    if pressed:
        # Always store the insertion point
        state.insertion_point = (x, y)
        print(f"Set insertion point at ({x}, {y})")


def main():
    """Main entry point"""
    print("-" * 60)
    print("Sequential Text Overlay with Code Processing")
    print("-" * 60)
    print(f"Platform detected: {sys.platform}")

    # Set up mouse listener
    mouse_listener = mouse.Listener(on_click=on_click)
    mouse_listener.start()

    try:
        # Create and run app
        root = tk.Tk()
        app = OverlayTextApp(root)

        # Show instructions
        print("\nInstructions:")
        print("1. Position your cursor where you want to insert text")
        print("2. Press Ctrl+Shift+T to show the overlay")
        print("3. Type the text you want to insert and any instructions")
        print("4. Press Enter or click 'Insert Text'")
        print("5. The app will insert the text, then select and process the code block")
        print("6. Press Escape or click 'Cancel' to hide the overlay")

        print("\nApplication running... (Press Ctrl+C in this console to exit)")

        # Start the Tkinter event loop
        root.mainloop()
    except KeyboardInterrupt:
        print("\nApplication terminated by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()