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

# Configuration
CONFIG = {
    'SERVER_HOST': '127.0.0.1',
    'SERVER_PORT': 66439,
    'API_KEY': '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',
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

    # Get all text first - use Ctrl+A, Ctrl+C
    pyautogui.hotkey('ctrl', 'a')
    time.sleep(CONFIG['COPY_PASTE_DELAY'])
    pyautogui.hotkey('ctrl', 'c')
    full_text = pyperclip.paste()

    # Return to the line we were on
    pyautogui.moveTo(x, y)
    pyautogui.click()
    time.sleep(CONFIG['CLICK_DELAY'])

    # Get current line text using Home + Shift+End instead of triple click
    pyautogui.press('home')
    time.sleep(0.05)
    pyautogui.keyDown('shift')
    pyautogui.press('end')
    pyautogui.keyUp('shift')
    time.sleep(0.05)

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

    # Return to original position without any additional clicks or selections
    pyautogui.moveTo(temp_pos.x, temp_pos.y)

    return full_text, line_number

def preview_code_block(x: int, y: int) -> str:
    """Preview the code block at the current position without selecting it."""
    try:
        # Store initial cursor position
        initial_pos = pyautogui.position()

        # Get all text in editor without moving cursor
        pyautogui.hotkey('ctrl', 'a')
        time.sleep(CONFIG['COPY_PASTE_DELAY'])
        pyautogui.hotkey('ctrl', 'c')
        full_text = pyperclip.paste()

        # Return to initial position immediately
        pyautogui.click(initial_pos.x, initial_pos.y)
        time.sleep(CONFIG['CLICK_DELAY'])

        # Get current line without moving cursor
        pyautogui.hotkey('shift', 'home')
        pyautogui.hotkey('shift', 'end')
        time.sleep(0.05)
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

        print(f"\nPreviewed block (starting at line {start}):")
        print("-" * 40)
        print(block_text)
        print("-" * 40)

        return block_text

    except Exception as e:
        print(f"\nError in preview_code_block: {str(e)}")
        return None


def process_and_update_text():
    """Process text with line-by-line replacement while maintaining exact spaces and handling line insertions."""
    global state
    try:
        if not all([state.insertion_point, state.starting_line_number, state.out, state.chat_app_position]):
            print("Error: Missing required state information")
            return

        # Save current cursor position and x-coordinate for vertical movement
        temp_pos = pyautogui.position()
        line_x_position = temp_pos.x  # Store x-coordinate for consistent vertical movement

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

        # Extract and clean response
        new_text = response.content[0].text.strip()
        if new_text.startswith('```'):
            lines = new_text.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].startswith('```'):
                lines = lines[:-1]
            while lines and not lines[-1].strip():
                lines.pop()
            new_text = '\n'.join(lines)

        # Split into lines while preserving whitespace
        new_lines = [line for line in new_text.split('\n') if line.strip() or any(c.isspace() for c in line)]
        state.llm_lines_count = len(new_lines)

        # Move to the insertion point maintaining x-coordinate
        pyautogui.moveTo(line_x_position, state.insertion_point[1])
        pyautogui.click()
        time.sleep(CONFIG['CLICK_DELAY'])

        # Calculate line offset
        _, current_line_num = get_text_and_line_number(line_x_position, state.insertion_point[1])
        lines_to_move = state.starting_line_number - current_line_num

        # Move to starting line maintaining x-coordinate
        if lines_to_move > 0:
            for _ in range(lines_to_move):
                current_y = pyautogui.position().y + CONFIG['LINE_HEIGHT']
                pyautogui.moveTo(line_x_position, current_y)
                pyautogui.click()
                time.sleep(0.05)
        elif lines_to_move < 0:
            for _ in range(abs(lines_to_move)):
                current_y = pyautogui.position().y - CONFIG['LINE_HEIGHT']
                pyautogui.moveTo(line_x_position, current_y)
                pyautogui.click()
                time.sleep(0.05)

        # Replace lines maintaining x-coordinate
        for i, new_line in enumerate(new_lines):
            if i < state.initial_line_count:
                # Select current line without moving cursor
                pyautogui.hotkey('shift', 'home')
                pyautogui.hotkey('shift', 'end')
                time.sleep(0.05)

                # Paste new line content
                pyperclip.copy(new_line)
                time.sleep(0.05)
                pyautogui.hotkey('ctrl', 'v')
                time.sleep(CONFIG['COPY_PASTE_DELAY'])

                # Move to next line maintaining x-coordinate
                if i < len(new_lines) - 1:
                    current_y = pyautogui.position().y + CONFIG['LINE_HEIGHT']
                    pyautogui.moveTo(line_x_position, current_y)
                    pyautogui.click()
                    time.sleep(0.05)
            else:
                # Handle additional lines
                pyautogui.press('end')
                pyautogui.press('enter')
                time.sleep(0.05)
                pyperclip.copy(new_line)
                pyautogui.hotkey('ctrl', 'v')
                time.sleep(CONFIG['COPY_PASTE_DELAY'])

        # Print line count information
        print(f"Sent Lines: {state.sent_lines_count}")
        print(f"LLM Lines: {state.llm_lines_count}")

        # The problematic section is likely in get_text_and_line_number or in some other function
        # that might be called after processing. We'll directly return to original position with no extra actions.
        pyautogui.moveTo(temp_pos.x, temp_pos.y)

        # Just perform a simple click with no extra selections
        pyautogui.click()
        time.sleep(CONFIG['CLICK_DELAY'])

        # Prevent any additional mouse actions for a short period
        time.sleep(0.5)

    except Exception as e:
        print(f"Error in process_and_update_text: {e}")
        import traceback
        traceback.print_exc()

# --------------------- Additional Helpers --------------------- #

def apply_changes_at_line(line_number: int, changes: List[Tuple[int, str]]):
    """Apply changes starting at a specific line number."""
    current_position = pyautogui.position()
    pyautogui.hotkey('ctrl', 'a')
    time.sleep(CONFIG['COPY_PASTE_DELAY'])
    pyautogui.press('home')

    for _ in range(line_number):
        pyautogui.press('down')
        time.sleep(0.05)

    for relative_line_num, new_text in sorted(changes, key=lambda x: x[0]):
        for _ in range(relative_line_num):
            pyautogui.press('down')
            time.sleep(0.05)

        pyautogui.keyDown('home')
        pyautogui.keyDown('shift')
        pyautogui.press('end')
        pyautogui.keyUp('shift')
        pyautogui.keyUp('home')

        pyperclip.copy(new_text)
        pyautogui.hotkey('ctrl', 'v')
        time.sleep(CONFIG['COPY_PASTE_DELAY'])
        print(f"Updated line {line_number + relative_line_num}: {new_text}")

    pyautogui.moveTo(current_position[0], current_position[1])


def select_line(x: int, y: int) -> None:
    """
    Selects the entire line at the specified coordinates using triple-click.

    Args:
        x: The x-coordinate for the click position
        y: The y-coordinate for the click position
    """
    # Store current mouse position
    original_pos = pyautogui.position()

    try:
        # Move to specified position
        pyautogui.moveTo(x, y)
        time.sleep(0.2)  # Increased delay for stability

        # Triple click to select the entire line
        pyautogui.click(clicks=3, interval=0.1)
        time.sleep(0.1)

        # Copy selection to check content
        pyautogui.hotkey('ctrl', 'c')  # Copy selection
        current_text = pyperclip.paste()

        # If there's no text, we've reached the end
        if not current_text.strip():
            pyautogui.moveTo(original_pos.x, original_pos.y)
            return

        # Return to original position
        pyautogui.moveTo(original_pos.x, original_pos.y)

    except Exception as e:
        print(f"Error selecting line: {str(e)}")
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
        # Store current cursor position without moving it
        current_pos = pyautogui.position()

        # Get text and line number without moving cursor
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

        # Update state without moving cursor
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