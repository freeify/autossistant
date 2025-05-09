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
    'TESSERACT_PATH': r'C:\Program Files\Tesseract-OCR\tesseract.exe',
    'PARAGRAPH_CHAR_THRESHOLD': 100,  # Min characters for a paragraph
    'PARAGRAPH_DETECTION_LINES': 10  # Number of lines to analyze for paragraph detection
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
        self.content_type = "code"  # Default content type: "code" or "text"


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

    def select_block_with_ocr(self, search_text, target_lines, image_path=None):
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
        'text': {
            'indent_based': False,
            'block_start': r'^.*$',  # Any line can start a text block
            'keywords': []
        },
        'unknown': {
            'indent_based': False,
            'block_start': r'^.*[{:]$',
            'keywords': ['block']
        }
    }
    return patterns.get(language, patterns['unknown'])


def find_paragraph_boundaries(lines, cursor_line):
    """
    Find the start and end of the current paragraph with enhanced detection logic.
    A paragraph is defined as consecutive non-empty lines that share similar formatting.
    """
    if not lines or cursor_line >= len(lines):
        return 0, 0

    # Start from cursor position
    start = cursor_line
    end = cursor_line

    # Check if we're on an empty line
    if not lines[cursor_line].strip():
        # If we're on an empty line, find the nearest non-empty paragraph
        # First look forward
        forward_start = cursor_line
        while forward_start < len(lines) - 1:
            forward_start += 1
            if lines[forward_start].strip():
                break

        # Then look backward
        backward_start = cursor_line
        while backward_start > 0:
            backward_start -= 1
            if lines[backward_start].strip():
                break

        # Choose the closest non-empty line
        if forward_start >= len(lines) or not lines[forward_start].strip():
            forward_distance = float('inf')
        else:
            forward_distance = forward_start - cursor_line

        if backward_start < 0 or not lines[backward_start].strip():
            backward_distance = float('inf')
        else:
            backward_distance = cursor_line - backward_start

        # Set cursor_line to the closest non-empty line
        if forward_distance < backward_distance:
            cursor_line = forward_start
        else:
            cursor_line = backward_start

        # If both searches failed, return the original cursor line
        if cursor_line == forward_start and forward_distance == float('inf') or \
                cursor_line == backward_start and backward_distance == float('inf'):
            return start, end

        # Update start and end to the new cursor line
        start = cursor_line
        end = cursor_line

    # Detect indentation of the current line
    current_indent = len(lines[cursor_line]) - len(lines[cursor_line].lstrip())
    indent_threshold = 4  # Allow a small variation in indentation

    # Go backward to find paragraph start (empty line or significant indent change)
    while start > 0:
        prev_line = lines[start - 1]

        # Stop if we hit an empty line
        if not prev_line.strip():
            break

        # Check for significant indentation change
        if prev_line.strip():
            prev_indent = len(prev_line) - len(prev_line.lstrip())
            if abs(prev_indent - current_indent) > indent_threshold:
                # Indent change might indicate a new paragraph
                # But first check if this is a list item or special formatting
                if not (prev_line.lstrip().startswith(('-', '*', '•', '1.', '2.', '3.')) or
                        prev_line.lstrip().startswith(('>', '#'))):
                    break

        # Check for paragraph separators like headers or horizontal rules
        if prev_line.strip().startswith(('#', '---', '===', '***')):
            break

        start -= 1

    # Go forward to find paragraph end (empty line or significant indent change)
    while end < len(lines) - 1:
        next_line = lines[end + 1]

        # Stop if we hit an empty line
        if not next_line.strip():
            break

        # Check for significant indentation change
        if next_line.strip():
            next_indent = len(next_line) - len(next_line.lstrip())
            if abs(next_indent - current_indent) > indent_threshold:
                # Indent change might indicate a new paragraph
                # But first check if this is a list item or special formatting
                if not (next_line.lstrip().startswith(('-', '*', '•', '1.', '2.', '3.')) or
                        next_line.lstrip().startswith(('>', '#'))):
                    break

        # Check for paragraph separators like headers
        if next_line.strip().startswith(('#', '---', '===', '***')):
            break

        end += 1

    return start, end


# Enhanced function to detect if content looks like text rather than code
def is_text_content(content):
    """
    Determine if content is plain text (non-code).
    Returns True if the content appears to be prose text rather than code.
    """
    # Get a sample of the first few lines
    lines = content.split('\n')
    lines_to_check = min(CONFIG['PARAGRAPH_DETECTION_LINES'], len(lines))
    sample = '\n'.join(lines[:lines_to_check])

    # Indicators for code
    code_patterns = [
        r'^\s*(def|class|if|elif|else|for|while|try|except|with)\s+.*:',  # Python
        r'^\s*(function|class|if|else|for|while|try|catch)\s+.*{',  # JavaScript
        r'^\s*(public|private|protected|class|interface|void|static)',  # Java
        r'^\s*import\s+',  # Import statements
        r'^\s*#include',  # C/C++
        r'^\s*[<>{};]$',  # Common code syntax
        r'^\s*//|^\s*/\*|\*/$',  # Comments
        r'^\s*[a-zA-Z0-9_]+\s*\(',  # Function calls
    ]

    # Indicators for markdown/text
    text_patterns = [
        r'^\s*#{1,6}\s+\w',  # Markdown headers
        r'^\s*[-*+]\s+\w',  # Markdown lists
        r'^\s*\d+\.\s+\w',  # Numbered lists
        r'^\s*>\s+\w',  # Blockquotes
        r'^\s*\[.*\]\(.*\)',  # Markdown links
    ]

    # Count code indicators
    code_indicators = sum(1 for pattern in code_patterns if re.search(pattern, sample, re.MULTILINE))

    # Count text/markdown indicators
    text_indicators = sum(1 for pattern in text_patterns if re.search(pattern, sample, re.MULTILINE))

    # Count long lines that likely indicate prose paragraphs
    long_lines = sum(1 for line in lines[:lines_to_check] if len(line.strip()) > CONFIG['PARAGRAPH_CHAR_THRESHOLD'])

    # Check punctuation density
    punctuation_count = sum(1 for char in sample if char in '.,:;?!')
    char_count = len(sample)
    punctuation_ratio = punctuation_count / max(1, char_count)

    # Sentence-like structures indicate text
    sentences = re.findall(r'[.!?]\s+[A-Z]', sample)

    # Check for quotation marks which are common in text
    quotes = re.findall(r'["\'"].*?["\'"]', sample)

    # Calculate results - more sophisticated approach
    if code_indicators >= 3:
        # Strong evidence of code
        return False
    elif text_indicators >= 2 or long_lines >= 2 or len(sentences) >= 2 or len(quotes) >= 2:
        # Strong evidence of text
        return True
    elif punctuation_ratio > 0.05 and code_indicators < 2:
        # High punctuation density suggests text
        return True
    else:
        # Default to code if uncertain
        return False
def find_block_boundaries(lines, cursor_line, content_type="code"):
    """Find the start and end of the current block (code or text)."""
    print(f"Content type detected: {content_type}")

    # Handle empty file or invalid cursor position
    if not lines or cursor_line >= len(lines):
        return 0, 0, 'unknown'

    # For text content, find paragraph boundaries
    if content_type == "text":
        start, end = find_paragraph_boundaries(lines, cursor_line)
        return start, end, 'paragraph'

    # Otherwise, proceed with code block detection
    language = detect_language('\n'.join(lines))
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

class ContentProcessor:
    """Class that handles code/text block selection and processing using OCR."""

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
        pyautogui.press('escape')  # Cancel selection

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

    def preview_block(self, x, y):
        """Preview the content block (code or text) at the current position without selecting it."""
        try:
            # Store initial cursor position
            initial_pos = pyautogui.position()

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

            # Get all text in editor
            pyautogui.hotkey('ctrl', 'a')
            time.sleep(CONFIG['COPY_PASTE_DELAY'])
            pyautogui.hotkey('ctrl', 'c')
            full_text = pyperclip.paste()
            pyautogui.press('escape')  # Cancel selection

            # Find line number
            lines = full_text.splitlines()
            cursor_line = 0
            for i, line in enumerate(lines):
                if line.strip() == current_line.strip():
                    cursor_line = i
                    break

            # Determine if this is code or text content
            state.content_type = "text" if is_text_content(full_text) else "code"

            # Get block boundaries based on content type
            if state.content_type == "code":
                # Detect language and get block boundaries
                language = detect_language(full_text)
                start, end, block_type = find_block_boundaries(lines, cursor_line, "code")
            else:
                # For text, use paragraph detection
                start, end, block_type = find_block_boundaries(lines, cursor_line, "text")
                language = "text"

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

            self.log(f"\nPreviewed {state.content_type} block (starting at line {start}):")
            self.log("-" * 40)
            self.log(block_text)
            self.log("-" * 40)

            return block_text, start, end, language, block_type

        except Exception as e:
            self.log(f"\nError in preview_block: {str(e)}")
            traceback.print_exc()
            return None, None, None, None, None

    def process_content_with_llm(self, block_content, input_text, content_type, language="", block_type=""):
        """Process content block with LLM using both the content and user input."""
        try:
            client = anthropic.Anthropic(api_key=self.api_key)

            # Adjust prompt based on content type
            if content_type == "code":
                system_prompt = (
                    f"Language: {language}\n"
                    f"Block Type: {block_type}\n"
                    f"Current code block:\n{block_content}\n\n"
                    f"User requested modification: {input_text}\n\n"
                    "Please modify the code block according to the user's request. "
                    "Return only the modified code block with the same indentation, no explanations."
                )
            else:  # Text content
                system_prompt = (
                    f"Current text block:\n{block_content}\n\n"
                    f"User requested modification: {input_text}\n\n"
                    "Please modify the text block according to the user's request. "
                    "Return only the modified text with the same formatting and structure, no explanations."
                )

            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": system_prompt
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
            traceback.print_exc()
            return block_content  # Return original content block on error

    def select_and_process_block(self, search_text, input_text, target_lines=None):
        """
        Use OCR to find the search_text, select lines, and process with LLM.
        Works with both code and text content.

        Args:
            search_text: Text to search for on screen as starting point
            input_text: User's instruction for modification
            target_lines: Optional number of lines to select (if None, will try to detect)
        """
        try:
            # First try to detect the block with traditional methods
            block_text, start, end, language, block_type = self.preview_block(
                pyautogui.position().x,
                pyautogui.position().y
            )

            if not block_text:
                self.log("Could not detect any content block")
                return False

            # Calculate the number of lines in the detected block
            lines_in_block = end - start + 1
            self.log(f"Detected {state.content_type} block with {lines_in_block} lines")
            if state.content_type == "code":
                self.log(f"Language: {language}, Block type: {block_type}")

            # Find first non-empty line to use as search text
            lines = block_text.splitlines()
            if not lines:
                self.log("Empty block detected")
                return False

            # Find first non-empty line to use as search text
            first_non_empty_line = None
            for i, line in enumerate(lines):
                if line.strip():
                    first_non_empty_line = line.strip()
                    self.log(f"Using line {i} as search text: '{first_non_empty_line}'")
                    break

            if not first_non_empty_line:
                self.log("Could not find suitable text in block")
                return False

            # Override any passed-in search_text with the first non-empty line
            search_text = first_non_empty_line

            # Store the first line number for later reference
            first_line_number = start
            self.log(f"First line number of block: {first_line_number}")

            # Store current cursor position before doing any navigation
            original_cursor_position = pyautogui.position()
            self.log(f"Original cursor position: {original_cursor_position}")

            # First go to beginning of current line to get a reference point
            pyautogui.press('home')
            time.sleep(0.1)

            # Get current line text to find our position within the block
            pyautogui.keyDown('shift')
            pyautogui.press('end')
            pyautogui.keyUp('shift')
            pyautogui.hotkey('ctrl', 'c')
            time.sleep(0.2)
            current_line_text = pyperclip.paste()

            # Cancel the selection
            pyautogui.press('escape')
            time.sleep(0.1)

            # Compare with our block lines to find our current position in the block
            current_line_idx = None
            for i, line in enumerate(lines):
                if line.strip() == current_line_text.strip():
                    current_line_idx = i
                    self.log(f"Current position is at line {i} in the detected block")
                    break

            # Calculate how many lines we need to move to get to first line of block
            if current_line_idx is None:
                self.log("Couldn't match current line in block, using best guess")
                # Make a best guess - assume we're somewhere in the middle and need to go up
                # Try moving up a few lines to find the start
                test_moves = min(5, len(lines) // 2)  # Don't move too far
                for _ in range(test_moves):
                    pyautogui.press('up')
                    time.sleep(0.05)

                    # Check if we found the first line
                    pyautogui.press('home')
                    pyautogui.keyDown('shift')
                    pyautogui.press('end')
                    pyautogui.keyUp('shift')
                    pyautogui.hotkey('ctrl', 'c')
                    time.sleep(0.2)
                    test_line = pyperclip.paste()
                    pyautogui.press('escape')

                    if test_line.strip() == lines[0].strip():
                        self.log(f"Found first line after moving up {_ + 1} lines")
                        break
            else:
                # We know exactly where we are in the block
                lines_to_move = -current_line_idx  # Negative means move up

                # Move to the first line of the block
                for _ in range(abs(lines_to_move)):
                    if lines_to_move < 0:
                        pyautogui.press('up')
                    else:
                        pyautogui.press('down')
                    time.sleep(0.05)

            # Now we should be at the first line, position at beginning of line
            pyautogui.press('home')
            time.sleep(0.1)

            # Verify we're at the first line by getting the text
            pyautogui.keyDown('shift')
            pyautogui.press('end')
            pyautogui.keyUp('shift')
            pyautogui.hotkey('ctrl', 'c')
            time.sleep(0.2)
            first_line_check = pyperclip.paste()
            pyautogui.press('escape')  # Cancel selection

            # Log what we found
            self.log(f"First line check: '{first_line_check}'")
            if lines[0].strip() == first_line_check.strip():
                self.log("Successfully positioned at first line of block")
            else:
                self.log(f"WARNING: First line verification failed. Expected: '{lines[0]}', Got: '{first_line_check}'")

                # Try OCR-based positioning as a fallback
                self.log("Attempting OCR-based positioning as fallback")
                ocr_text, match = self.ocr_selector.select_block_with_ocr(
                    search_text,
                    lines_in_block
                )

                if ocr_text:
                    selected_text = ocr_text
                    self.log("OCR-based selection successful")
                else:
                    self.log("OCR-based selection failed, continuing with best effort")

                    # Try one more time to find the first line through relative navigation
                    self.log("Trying additional relative navigation")

                    # Move up a few more lines to see if we can find the beginning
                    for i in range(5):
                        pyautogui.press('up')
                        time.sleep(0.05)

                        # Check if we found the first line
                        pyautogui.press('home')
                        pyautogui.keyDown('shift')
                        pyautogui.press('end')
                        pyautogui.keyUp('shift')
                        pyautogui.hotkey('ctrl', 'c')
                        time.sleep(0.2)
                        test_line = pyperclip.paste()
                        pyautogui.press('escape')

                        if test_line.strip() == lines[0].strip():
                            self.log(f"Found first line after additional navigation")
                            break

            # Select the block now that we're at the beginning
            # IMPORTANT: This is now the only place we select the block
            pyautogui.press('home')  # Ensure we're at the beginning of line
            time.sleep(0.1)
            pyautogui.keyDown('shift')
            for _ in range(lines_in_block - 1):
                pyautogui.press('down')
                time.sleep(0.02)
            pyautogui.press('end')  # Select to end of last line
            pyautogui.keyUp('shift')
            time.sleep(0.1)

            # Get the selected text
            pyautogui.hotkey('ctrl', 'c')
            time.sleep(0.2)
            selected_text = pyperclip.paste()

            # Log the selection status
            selected_lines = len(selected_text.splitlines())
            self.log(f"Selected {selected_lines} lines (expected {lines_in_block})")

            # Process the selected text with the LLM
            processed_text = self.process_content_with_llm(
                selected_text,
                input_text,
                state.content_type,
                language,
                block_type
            )

            # Now paste the processed text over the selection
            # The block should still be selected from above
            pyperclip.copy(processed_text)
            time.sleep(0.2)
            pyautogui.hotkey('ctrl', 'v')
            time.sleep(CONFIG['COPY_PASTE_DELAY'])
            self.log(f"Replaced {state.content_type} block with processed text")

            return True

        except Exception as e:
            self.log(f"Error in select_and_process_block: {e}")
            traceback.print_exc()
            return False


# --------------------- Overlay Application --------------------- #

class OverlayTextApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Content Modifier")
        self.setup_ui()

        # Initialize state variables
        self.last_active_window = None
        self.last_active_window_title = None
        self.is_visible = False
        self.debug_mode = True  # Set to True to see debug information
        self.last_inserted_text = None

        # Initialize the content processor
        self.content_processor = ContentProcessor(debug=self.debug_mode)

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
        window_width = 600  # Increased width
        window_height = 280  # Slightly increased height for content type indicator
        position_x = (screen_width - window_width) // 2
        position_y = (screen_height - window_height) // 3
        self.root.geometry(f"{window_width}x{window_height}+{position_x}+{position_y}")

        # Frame for content
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Instructions label
        instructions = tk.Label(
            main_frame,
            text="Content Modifier",
            font=("Arial", 14, "bold")
        )
        instructions.pack(pady=(0, 5))

        # Auto-detection info as subtitle
        self.subtitle = tk.Label(
            main_frame,
            text="Automatically detects code or text at cursor position",
            font=("Arial", 10),
            fg="blue"
        )
        self.subtitle.pack(pady=(0, 10))

        # Content type indicator
        self.content_type_label = tk.Label(
            main_frame,
            text="Current content type: Detecting...",
            font=("Arial", 10, "italic"),
            fg="purple"
        )
        self.content_type_label.pack(pady=(0, 5))

        # Modification instructions - main focus of the UI
        mod_label = tk.Label(main_frame, text="Enter modification instructions:", anchor="w",
                             font=("Arial", 11, "bold"))
        mod_label.pack(fill=tk.X, pady=(10, 5), padx=5)

        self.text_entry = tk.Text(main_frame, height=5, width=40, font=("Arial", 11))
        self.text_entry.pack(fill=tk.BOTH, expand=True, pady=5)
        self.text_entry.bind("<Control-Return>", self.process_content)

        # Add hint text
        hint_label = tk.Label(main_frame,
                              text="Position cursor in a paragraph or code block, then enter instructions above",
                              fg="gray", font=("Arial", 9))
        hint_label.pack(fill=tk.X, pady=(0, 5), padx=5)

        # Buttons frame
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)

        # Process button
        self.process_button = tk.Button(
            button_frame,
            text="Process Content (Ctrl+Enter)",
            font=("Arial", 11),
            command=self.process_content
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

    def process_content(self, event=None):
        """Process the content block based on user inputs."""
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

            # First detect the block to get the first line as search text
            current_x, current_y = pyautogui.position()
            block_text, start, end, language, block_type = self.content_processor.preview_block(current_x, current_y)

            if not block_text:
                self.log("Failed to detect content block", error=True)
                self.show_error("Could not detect content. Please position cursor within text or code.")
                return

            # Extract the first line of the block as search text
            lines = block_text.splitlines()
            if not lines:
                self.log("Detected empty block", error=True)
                self.show_error("Detected block has no content")
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
                self.log("Could not find suitable search text in block", error=True)
                self.show_error("Could not find suitable text in block to use as search anchor")
                return

            self.log(f"Auto-detected search text: '{search_text}'")

            if state.content_type == "code":
                self.log(f"Processing code block: lines {start}-{end}, type: {block_type}, language: {language}")
            else:
                self.log(f"Processing text block: lines {start}-{end}")

            self.log(f"Modification: '{input_text}'")

            # Process the content block with auto-detected properties
            success = self.content_processor.select_and_process_block(
                search_text,
                input_text,
                None  # Always pass None to force auto-detection of line count
            )

            if success:
                self.log("Content processed successfully")
                self.update_status("Content modified successfully")
            else:
                self.log("Failed to process content block", error=True)
                self.update_status("Failed to process content", error=True)

            # Restore original clipboard after a brief pause
            time.sleep(0.3)
            pyperclip.copy(original_clipboard)

        except Exception as e:
            self.log(f"Error during content processing: {e}", error=True)
            self.show_error(f"Failed to process content: {str(e)}")

            # Restore original clipboard
            pyperclip.copy(original_clipboard)

    def setup_keyboard_hook(self):
        """Set up keyboard hooks for activation"""
        try:
            # For global hotkey activation
            keyboard.add_hotkey('ctrl+shift+m', self.show_overlay)  # Changed from O to M for "Modifier"

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
                        if window_title and window_title != "Content Modifier":
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
                        if active_window and active_window.title != "Content Modifier":
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
                    if wmname and "Content Modifier" not in wmname:
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
        """Show the overlay window and prepare the UI"""
        if not self.is_visible:
            try:
                # Try to detect content type before showing UI
                try:
                    # Get current position
                    x, y = pyautogui.position()

                    # Preview block to detect content type
                    block_text, _, _, _, _ = self.content_processor.preview_block(x, y)

                    # Update UI based on detected content type
                    if state.content_type == "code":
                        self.content_type_label.config(text="Current content type: Code", fg="blue")
                        self.subtitle.config(text="Automatically detects code blocks at cursor position")
                    else:
                        self.content_type_label.config(text="Current content type: Text", fg="green")
                        self.subtitle.config(text="Automatically detects text paragraphs at cursor position")
                except:
                    # If detection fails, show neutral message
                    self.content_type_label.config(text="Current content type: Detecting...", fg="purple")
                    self.subtitle.config(text="Automatically detects code or text at cursor position")

                # Save the current active window before we take focus
                self.update_status("Ready to process content")

                # Show the window
                self.root.deiconify()
                self.is_visible = True

                # Reset and focus text entry
                self.text_entry.delete("1.0", tk.END)
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
    print("Content Modification Tool - For Code and Text")
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
        print("1. Position your cursor within a code block or text paragraph")
        print("2. Press Ctrl+Shift+M to show the overlay")
        print("3. Enter modification instructions for Claude")
        print("4. Press Ctrl+Enter or click 'Process Content'")
        print("5. The app will find the text/code, select it, and modify it")
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