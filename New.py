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
import pytesseract
from PIL import Image
from difflib import SequenceMatcher

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
    'API_KEY': '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',
    'CLICK_DELAY': 0.1,
    'COPY_PASTE_DELAY': 0.2,
    'LINE_HEIGHT': 20,
    'OCR_CONFIDENCE': 70,
    'TESSERACT_PATH': r'C:\Program Files\Tesseract-OCR\tesseract.exe'
}

# Set path to Tesseract
pytesseract.pytesseract.tesseract_cmd = CONFIG['TESSERACT_PATH']


# Global state
class State:
    def __init__(self):
        self.last_active_window = None
        self.last_active_window_title = None
        self.insertion_point = None
        self.last_inserted_text = None
        self.starting_line_number = None
        self.out = None  # Current code block content
        self.exit_flag = False  # Flag to indicate if ESC has been pressed


state = State()


# --------------------- OCR Text Selection Class --------------------- #

class OCRTextSelector:
    """Class for OCR-based text selection functionality."""

    def __init__(self, debug=True):
        self.debug = debug

    def log(self, message):
        """Log message if debug is enabled."""
        if self.debug:
            print(message)

    def clean_text(self, text):
        """Clean text for better matching."""
        return re.sub(r'\s+', ' ', text).strip()

    def similarity_score(self, text1, text2):
        """Calculate similarity between two strings."""
        return SequenceMatcher(None, text1, text2).ratio()

    def get_screenshot(self, countdown=3):
        """Take a screenshot after a countdown."""
        self.log(f"Taking screenshot in {countdown} seconds...")
        for i in range(countdown, 0, -1):
            self.log(f"{i}...")
            time.sleep(1)

        self.log("Capturing screenshot now!")
        return pyautogui.screenshot()

    def find_text_on_screen(self, search_text, min_confidence=None, image_path=None):
        """Find text on screen using OCR."""
        if min_confidence is None:
            min_confidence = CONFIG['OCR_CONFIDENCE']

        # Get image from file or screenshot
        if image_path:
            try:
                image = Image.open(image_path)
            except Exception as e:
                self.log(f"Error opening image: {e}")
                self.log("Taking screenshot instead...")
                image = self.get_screenshot()
        else:
            image = self.get_screenshot()

        # Clean search text
        clean_search = self.clean_text(search_text)
        self.log(f"Looking for: '{clean_search}'")

        # Use Tesseract to get text with position data
        ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

        # Process results
        text_instances = []
        for i in range(len(ocr_data['text'])):
            # Skip empty text
            if not ocr_data['text'][i].strip():
                continue

            # Skip low confidence results
            try:
                # Convert confidence value to float first (Tesseract might return float values)
                confidence = float(ocr_data['conf'][i])
                if confidence < min_confidence:
                    continue
            except ValueError:
                # Skip items with invalid confidence values
                continue

            # Add to our list
            text_instances.append({
                'text': ocr_data['text'][i],
                'left': ocr_data['left'][i],
                'top': ocr_data['top'][i],
                'width': ocr_data['width'][i],
                'height': ocr_data['height'][i],
                'conf': ocr_data['conf'][i]
            })

        # Combine words on same line into phrases
        # Group by vertical position (with some tolerance)
        vertical_tolerance = 5
        y_positions = {}

        for item in text_instances:
            y_center = item['top'] + item['height'] // 2

            # Find closest existing y-position
            matched_y = None
            for y in y_positions.keys():
                if abs(y - y_center) <= vertical_tolerance:
                    matched_y = y
                    break

            if matched_y is None:
                y_positions[y_center] = []
            else:
                y_center = matched_y

            y_positions[y_center].append(item)

        # Process each horizontal line
        lines = []
        for y_pos in sorted(y_positions.keys()):
            words = y_positions[y_pos]
            # Sort words by x position
            words.sort(key=lambda w: w['left'])

            # Now check for words that should be grouped into phrases
            phrases = []
            current_phrase = [words[0]] if words else []

            for i in range(1, len(words)):
                prev_word = words[i - 1]
                curr_word = words[i]

                # If words are close, group them
                gap = curr_word['left'] - (prev_word['left'] + prev_word['width'])
                if gap <= 20:  # Threshold for same phrase
                    current_phrase.append(curr_word)
                else:
                    # Gap is large, add current phrase and start new one
                    phrases.append(current_phrase)
                    current_phrase = [curr_word]

            # Add final phrase
            if current_phrase:
                phrases.append(current_phrase)

            # Process each phrase
            for phrase in phrases:
                # Get phrase boundaries
                left = min(word['left'] for word in phrase)
                top = min(word['top'] for word in phrase)
                right = max(word['left'] + word['width'] for word in phrase)
                bottom = max(word['top'] + word['height'] for word in phrase)

                # Combine text
                text = ' '.join(word['text'] for word in phrase)

                # Add to lines
                lines.append({
                    'text': text,
                    'left': left,
                    'top': top,
                    'width': right - left,
                    'height': bottom - top,
                    'center_x': (left + right) // 2,
                    'center_y': (top + bottom) // 2,
                    'line_start_x': left,  # This is the start of the line
                    'line_start_y': (top + bottom) // 2  # Y-center at the start of the line
                })

        # Now find best match for our search text
        best_match = None
        best_score = 0.4  # Minimum threshold

        for line in lines:
            clean_line = self.clean_text(line['text'])

            # First check for exact match
            if clean_search.lower() in clean_line.lower():
                score = 0.9 + (0.1 * len(clean_search) / len(clean_line))
                if score > best_score:
                    best_score = score
                    best_match = line
                    best_match['match_score'] = score

            # Fuzzy match if no exact match found
            if best_score < 0.9:
                score = self.similarity_score(clean_search.lower(), clean_line.lower())
                if score > best_score:
                    best_score = score
                    best_match = line
                    best_match['match_score'] = score

        # Calculate line height estimation from OCR data
        if lines:
            avg_height = sum(line['height'] for line in lines) / len(lines)
            line_spacing = 0

            # If we have multiple lines, try to calculate average spacing
            if len(lines) > 1:
                sorted_lines = sorted(lines, key=lambda x: x['top'])
                spacings = []
                for i in range(1, len(sorted_lines)):
                    curr_top = sorted_lines[i]['top']
                    prev_bottom = sorted_lines[i - 1]['top'] + sorted_lines[i - 1]['height']
                    spacing = curr_top - prev_bottom
                    if spacing > 0:  # Only consider positive spacings
                        spacings.append(spacing)

                if spacings:
                    line_spacing = sum(spacings) / len(spacings)

            # Estimated line height including spacing
            line_height = avg_height + line_spacing
        else:
            # Default if we couldn't calculate
            line_height = CONFIG['LINE_HEIGHT']

        return best_match, line_height

    def count_lines(self, text):
        """Count the number of lines in a text string."""
        if not text:
            return 0
        return text.count('\n') + 1

    def ensure_beginning_of_line(self, x, y):
        """Move to what should be the absolute beginning of the line."""
        # First move to the left margin (with some buffer)
        margin_x = 10  # Assuming 10 pixels from left edge is safe
        pyautogui.moveTo(margin_x, y, duration=0.05)

        # Now move right to the provided x position
        pyautogui.moveTo(x, y, duration=0.05)

        # Click once to position cursor at the beginning of the line
        pyautogui.click()
        time.sleep(0.05)

    def select_text_lines(self, start_x, start_y, target_lines, line_height=None):
        """Select a specific number of lines starting from the given position."""
        if line_height is None:
            line_height = CONFIG['LINE_HEIGHT']

        # Ensure we're at the beginning of the line first
        self.ensure_beginning_of_line(start_x, start_y)

        # Double-click to select first word
        pyautogui.doubleClick()
        time.sleep(0.05)

        # Press and hold left mouse button
        pyautogui.mouseDown()

        # Initial estimation of where to move to select target_lines
        current_y = start_y + (line_height * target_lines * 0.8)
        pyautogui.moveTo(start_x, current_y, duration=0.05)

        # Copy text without releasing mouse button to check how many lines we've selected
        pyautogui.hotkey('ctrl', 'c')
        current_text = pyperclip.paste()
        current_line_count = self.count_lines(current_text)

        self.log(f"Initial selection: {current_line_count} lines, target: {target_lines}")

        # Variables for movement control
        max_iterations = 30
        iterations = 0
        adjustment = line_height // 2

        # Continue dragging until exact match or max iterations
        while current_line_count != target_lines and iterations < max_iterations and not state.exit_flag:
            # Calculate lines difference
            lines_difference = target_lines - current_line_count

            # Dynamic adjustment based on how far we are from target
            if abs(lines_difference) > 5:
                adjustment = line_height * 1.5
            elif abs(lines_difference) > 2:
                adjustment = line_height * 0.75
            else:
                adjustment = line_height * 0.3  # Fine-grained adjustment when close

            # Calculate movement direction and distance
            move_distance = adjustment * (1 if lines_difference > 0 else -1)

            # Get current position
            current_x, current_y = pyautogui.position()

            # Move to the new position, maintaining drag selection
            pyautogui.moveTo(current_x, current_y + move_distance, duration=0.03)

            # Copy text WITHOUT releasing mouse
            pyautogui.hotkey('ctrl', 'c')
            new_text = pyperclip.paste()
            new_line_count = self.count_lines(new_text)

            # Update log occasionally
            if iterations % 3 == 0 or abs(target_lines - new_line_count) <= 1:
                self.log(f"Selection: {new_line_count} lines (target: {target_lines}), diff: {lines_difference}")

            # Update current state
            current_text = new_text
            current_line_count = new_line_count
            iterations += 1

            # Very small delay to let system process
            time.sleep(0.01)

        # Release mouse button after selection is complete
        pyautogui.mouseUp()

        # Final copy to ensure we have the complete selection
        pyautogui.hotkey('ctrl', 'c')
        final_text = pyperclip.paste()
        final_line_count = self.count_lines(final_text)

        self.log(f"Final selection: {final_line_count} lines (target: {target_lines})")
        if final_line_count != target_lines:
            self.log(f"Warning: Could not achieve exact line count after {iterations} iterations")

        return final_text, pyautogui.position()[1]

    def select_code_block_with_ocr(self, search_text, target_lines, image_path=None):
        """Use OCR to find text, then select a specific number of lines."""
        # Find the text using OCR
        match, estimated_line_height = self.find_text_on_screen(search_text, image_path=image_path)

        if match:
            # Get the starting coordinates
            start_x = match['line_start_x']
            start_y = match['line_start_y']

            self.log(f"Found text: '{match['text']}' (match score: {match.get('match_score', 0):.2f})")
            self.log(f"Starting selection at position ({start_x}, {start_y})")
            self.log(f"Estimated line height: {estimated_line_height:.2f} pixels")

            # Select the text
            selected_text, end_y = self.select_text_lines(
                start_x,
                start_y,
                target_lines,
                line_height=max(estimated_line_height, 15)  # Use at least 15px as minimum
            )

            return selected_text, match
        else:
            self.log(f"Couldn't find text matching '{search_text}'")
            return None, None


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


# --------------------- Code Processing Class --------------------- #

class CodeProcessor:
    """Class that handles code block selection and processing using OCR."""

    def __init__(self, api_key=None, debug=True):
        self.debug = debug
        self.ocr_selector = OCRTextSelector(debug=debug)
        self.api_key = api_key if api_key else CONFIG['API_KEY']

    def log(self, message):
        """Log message if debug is enabled."""
        if self.debug:
            print(message)

    def get_text_and_line_number(self, x, y):
        """Get text from the entire editor and determine the line number at the current cursor position."""
        # Store original position
        temp_pos = pyautogui.position()

        # First, save the current line position
        pyautogui.press('home')  # Go to beginning of current line
        time.sleep(CONFIG['CLICK_DELAY'])
        pyautogui.keyDown('shift')
        pyautogui.press('end')  # Select to end of line
        pyautogui.keyUp('shift')
        time.sleep(CONFIG['COPY_PASTE_DELAY'])

        # Copy the current line
        pyautogui.hotkey('ctrl', 'c')
        current_line = pyperclip.paste()

        # Cancel the selection
        pyautogui.press('escape')
        time.sleep(CONFIG['CLICK_DELAY'])

        # Get all text content
        pyautogui.hotkey('ctrl', 'a')  # Select all
        time.sleep(CONFIG['COPY_PASTE_DELAY'])
        pyautogui.hotkey('ctrl', 'c')  # Copy
        full_text = pyperclip.paste()

        # Cancel the selection to return to where we were
        pyautogui.press('escape')
        time.sleep(CONFIG['CLICK_DELAY'])

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
            self.log("Warning: Could not determine line number accurately")
            line_number = 0

        # Return to the specific position within the line
        pyautogui.moveTo(temp_pos.x, temp_pos.y)

        return full_text, line_number, lines

    def preview_code_block(self, x, y):
        """Preview the code block at the current position without selecting it."""
        try:
            # Store initial cursor position
            initial_pos = pyautogui.position()

            # Get all text in editor
            pyautogui.hotkey('ctrl', 'a')
            time.sleep(CONFIG['COPY_PASTE_DELAY'])
            pyautogui.hotkey('ctrl', 'c')
            full_text = pyperclip.paste()

            # Return to where we were (press escape to deselect all)
            pyautogui.press('escape')
            time.sleep(CONFIG['CLICK_DELAY'])

            # Get current line with keyboard
            pyautogui.press('home')
            time.sleep(CONFIG['CLICK_DELAY'])
            pyautogui.keyDown('shift')
            pyautogui.press('end')
            pyautogui.keyUp('shift')
            time.sleep(CONFIG['CLICK_DELAY'])
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

            # Cancel selection to avoid disrupting the flow
            pyautogui.press('escape')
            time.sleep(CONFIG['CLICK_DELAY'])

            # Return to initial cursor position
            pyautogui.moveTo(initial_pos.x, initial_pos.y)

            self.log(f"\nPreviewed block (starting at line {start}):")
            self.log("-" * 40)
            self.log(block_text)
            self.log("-" * 40)

            return block_text, start, end, language, block_type

        except Exception as e:
            self.log(f"\nError in preview_code_block: {str(e)}")
            return None, None, None, None, None

    def process_code_with_llm(self, code_block, input_text, language, block_type):
        """Process code block with LLM using both the code and user input."""
        try:
            client = anthropic.Anthropic(api_key=self.api_key)
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
            self.log(f"Error processing with LLM: {e}")
            return code_block  # Return original code block on error

    def select_and_process_code_block_with_ocr(self, search_text, input_text, target_lines=None):
        """
        Use OCR to find the search_text, select lines, and process with LLM.

        Args:
            search_text: Text to search for on screen as starting point (will be ignored in favor of detected first line)
            input_text: User's instruction for code modification
            target_lines: Optional number of lines to select (if None, will try to detect)
        """
        try:
            # First try to detect the code block with traditional methods to get language
            block_text, start, end, language, block_type = self.preview_code_block(
                pyautogui.position().x,
                pyautogui.position().y
            )

            if not block_text:
                self.log("Could not detect code block using traditional methods")
                return False

            # Calculate the number of lines in the detected block
            lines_in_block = end - start + 1
            self.log(f"Detected code block with {lines_in_block} lines ({language}, {block_type})")

            # Always use the first line of the detected block for better OCR matching
            # This is the key change - we always use the first line of the code block detected
            # rather than using the passed-in search_text or the copied text
            lines = block_text.splitlines()
            if lines:
                # Override any passed-in search_text with the first non-empty line of the block
                for line in lines:
                    if line.strip():
                        search_text = line.strip()
                        self.log(f"Using first non-empty line of block as search text: '{search_text}'")
                        break

            if not search_text:
                self.log("Could not find suitable text in first line of code block")
                return False

            # Detect current cursor position
            current_line_idx = None
            cursor_pos = pyautogui.position()

            # Try to go to first line of the code block
            # First store current position
            current_pos = pyautogui.position()

            # Navigate to the start line using keyboard shortcuts
            # First go to beginning of current line
            pyautogui.press('home')
            time.sleep(0.1)

            # Determine how many lines up we need to go to reach the start
            # Get current line text
            pyautogui.keyDown('shift')
            pyautogui.press('end')
            pyautogui.keyUp('shift')
            pyautogui.hotkey('ctrl', 'c')
            time.sleep(0.2)
            current_line_text = pyperclip.paste()

            # Compare with our block lines to find current position in block
            current_line_idx = None
            for i, line in enumerate(lines):
                if line.strip() == current_line_text.strip():
                    current_line_idx = i
                    break

            # If we couldn't determine current position, assume we need to go to beginning
            lines_to_move_up = current_line_idx if current_line_idx is not None else 0

            # Go up to the first line of the block
            for _ in range(lines_to_move_up):
                pyautogui.press('up')
                time.sleep(0.05)

            # Now we should be at the first line, select the whole block
            pyautogui.press('home')  # Move to beginning of line
            time.sleep(0.2)

            # Use Shift+Down to select the required number of lines
            pyautogui.keyDown('shift')
            for _ in range(lines_in_block - 1):  # -1 because current line counts as 1
                pyautogui.press('down')
                time.sleep(0.05)
            pyautogui.press('end')  # Select to end of last line
            pyautogui.keyUp('shift')

            # Verify the selection by copying
            pyautogui.hotkey('ctrl', 'c')
            time.sleep(0.2)
            selected_text = pyperclip.paste()
            selected_lines = len(selected_text.splitlines())
            self.log(f"Selected {selected_lines} lines (expected {lines_in_block})")

            # If keyboard selection failed or didn't select the expected number of lines,
            # fall back to OCR-based selection
            if abs(selected_lines - lines_in_block) > 1:
                self.log("Traditional selection didn't match expected lines, falling back to OCR")
                selected_text, match = self.ocr_selector.select_code_block_with_ocr(
                    search_text,
                    lines_in_block
                )

                if not selected_text:
                    self.log(f"OCR couldn't find text matching '{search_text}'")
                    return False

            # Process the selected text with the LLM
            processed_text = self.process_code_with_llm(
                selected_text,
                input_text,
                language,
                block_type
            )
            # Paste the processed text
            pyperclip.copy(processed_text)
            time.sleep(0.2)
            pyautogui.hotkey('ctrl', 'v')
            time.sleep(CONFIG['COPY_PASTE_DELAY'])
            self.log(f"Replaced code block with processed text")

            return True

        except Exception as e:
            self.log(f"Error in select_and_process_code_block_with_ocr: {e}")
            traceback.print_exc()
            return False

# --------------------- Overlay Application --------------------- #

class OverlayTextApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Code Modifier")
        self.setup_ui()

        # Initialize state variables
        self.last_active_window = None
        self.last_active_window_title = None
        self.is_visible = False
        self.debug_mode = True  # Set to True to see debug information
        self.last_inserted_text = None

        # Initialize the code processor
        self.code_processor = CodeProcessor(debug=self.debug_mode)

        # Start monitors and set up hooks
        self.setup_keyboard_hook()
        self.start_monitoring_thread()

        # Log startup
        self.log("Application started")
        self.log(f"Running on: {PLATFORM} platform")

    # Modification 1: Update the setup_ui method to hide the lines entry field
    # Replace the original lines frame section with a label that shows auto-detection is enabled

    # Modification 1: Update the setup_ui method to remove both search text and lines entry fields
    # Make a simpler UI focused just on entering modification instructions

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
        window_width = 600  # Increased width
        window_height = 260  # Height for simpler UI
        position_x = (screen_width - window_width) // 2
        position_y = (screen_height - window_height) // 3
        self.root.geometry(f"{window_width}x{window_height}+{position_x}+{position_y}")

        # Frame for content
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Instructions label
        instructions = tk.Label(
            main_frame,
            text="Code Modifier",
            font=("Arial", 14, "bold")
        )
        instructions.pack(pady=(0, 5))

        # Auto-detection info as subtitle
        subtitle = tk.Label(
            main_frame,
            text="Automatically detects code block at cursor position",
            font=("Arial", 10),
            fg="blue"
        )
        subtitle.pack(pady=(0, 10))

        # Modification instructions - now the main focus of the UI
        mod_label = tk.Label(main_frame, text="Enter modification instructions:", anchor="w",
                             font=("Arial", 11, "bold"))
        mod_label.pack(fill=tk.X, pady=(10, 5), padx=5)

        self.text_entry = tk.Text(main_frame, height=5, width=40, font=("Arial", 11))
        self.text_entry.pack(fill=tk.BOTH, expand=True, pady=5)
        self.text_entry.bind("<Control-Return>", self.process_code)

        # Add hint text
        hint_label = tk.Label(main_frame, text="Position cursor in a code block, then enter instructions above",
                              fg="gray", font=("Arial", 9))
        hint_label.pack(fill=tk.X, pady=(0, 5), padx=5)

        # Buttons frame
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)

        # Process button
        self.process_button = tk.Button(
            button_frame,
            text="Process Code (Ctrl+Enter)",
            font=("Arial", 11),
            command=self.process_code
        )
        self.process_button.pack(side=tk.LEFT, padx=5)

        # Cancel button
        self.cancel_button = tk.Button(
            button_frame,
            text="Cancel (Esc)",
            font=("Arial", 11),
            command=self.hide_overlay
        )
        self.cancel_button.pack(side=tk.LEFT, padx=5)

        # Status frame
        status_frame = tk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=5)

        # Status label
        self.status_label = tk.Label(status_frame, text="Ready", fg="gray",
                                     anchor="w", font=("Arial", 9))
        self.status_label.pack(fill=tk.X)

        # Initially hide the window
        self.root.withdraw()

        # Add keyboard bindings
        self.root.bind("<Escape>", lambda e: self.hide_overlay())

    # Modification 2: Update the process_code method to automatically determine search text and line count
    # We'll use the preview_code_block functionality to detect the code block at cursor position

    def process_code(self, event=None):
        """Process the code block based on user inputs."""
        input_text = self.text_entry.get("1.0", tk.END).strip()

        if not input_text:
            self.update_status("Please enter modification instructions", error=True)
            return

        if not self.last_active_window_title:
            self.update_status("No previous window detected", error=True)
            return

        # Save original clipboard content
        original_clipboard = pyperclip.paste()

        try:
            # Hide our window first
            self.hide_overlay()
            time.sleep(0.3)  # Wait briefly

            # Update global state
            self.last_inserted_text = input_text
            global state
            state.last_inserted_text = input_text

            # Platform specific window activation
            if PLATFORM == "windows":
                try:
                    # Try to activate the window
                    win32gui.SetForegroundWindow(self.last_active_window)
                    time.sleep(0.3)  # Allow time for focus
                except Exception as e:
                    self.log(f"Failed to focus window: {e}", error=True)
                    return

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
                    return

            # Wait a moment for window to activate
            time.sleep(0.3)

            # First detect the code block to get the first line as search text
            current_x, current_y = pyautogui.position()
            block_text, start, end, language, block_type = self.code_processor.preview_code_block(current_x, current_y)

            if not block_text:
                self.log("Failed to detect code block", error=True)
                self.show_error("Could not detect code block. Please position cursor within a code block.")
                return

            # Extract the first line of the code block as search text
            lines = block_text.splitlines()
            if not lines:
                self.log("Detected empty code block", error=True)
                self.show_error("Detected code block has no content")
                return

            # Use the first line of the block as search text
            search_text = lines[0].strip()
            if not search_text:
                # If first line is empty, find the first non-empty line
                for line in lines:
                    if line.strip():
                        search_text = line.strip()
                        break

            if not search_text:
                self.log("Could not find suitable search text in code block", error=True)
                self.show_error("Could not find suitable text in code block to use as search anchor")
                return

            self.log(f"Auto-detected search text: '{search_text}'")
            self.log(f"Processing code block: lines {start}-{end}, type: {block_type}, language: {language}")
            self.log(f"Modification: '{input_text}'")

            # Process the code block with auto-detected search text and auto-detected line count
            success = self.code_processor.select_and_process_code_block_with_ocr(
                search_text,
                input_text,
                None  # Always pass None to force auto-detection of line count
            )

            if success:
                self.log("Code processed successfully")
                self.update_status("Code modified successfully")
            else:
                self.log("Failed to process code block", error=True)
                self.update_status("Failed to process code", error=True)

            # Restore original clipboard after a brief pause
            time.sleep(0.3)
            pyperclip.copy(original_clipboard)

        except Exception as e:
            self.log(f"Error during code processing: {e}", error=True)
            self.show_error(f"Failed to process code: {str(e)}")

            # Restore original clipboard
            pyperclip.copy(original_clipboard)
    def setup_keyboard_hook(self):
        """Set up keyboard hooks for activation"""
        try:
            # For global hotkey activation
            keyboard.add_hotkey('ctrl+shift+o', self.show_overlay)  # Changed from t to o

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
                        if window_title and window_title != "Code Modifier":
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
                        if active_window and active_window.title != "Code Modifier":
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
                    if wmname and "Code Modifier" not in wmname:
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
                self.update_status("Ready to process code")

                # Show the window
                self.root.deiconify()
                self.is_visible = True

                # Reset and focus search entry
                self.search_entry.delete(0, tk.END)
                self.text_entry.delete("1.0", tk.END)
                self.search_entry.focus_set()

                self.log("Overlay shown")
            except Exception as e:
                self.log(f"Error showing overlay: {e}", error=True)

    def hide_overlay(self):
        """Hide the overlay window"""
        if self.is_visible:
            self.root.withdraw()
            self.is_visible = False
            self.log("Overlay hidden")

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
    print("OCR-Guided Code Selection and Modification Tool")
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
        print("1. Position your cursor within a code block")
        print("2. Press Ctrl+Shift+O to show the overlay")
        print("3. Enter text to search for (start of code block)")
        print("4. Enter number of lines to select (optional)")
        print("5. Enter modification instructions for Claude")
        print("6. Press Ctrl+Enter or click 'Process Code'")
        print("7. The app will find the code, select it, and modify it")
        print("8. Press Escape or click 'Cancel' to hide the overlay")

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