import pytesseract
from PIL import Image
import pyautogui
import time
import re
import numpy as np
from difflib import SequenceMatcher

# Point to where you installed Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Configure Tesseract for better code recognition
# These settings help optimize for monospaced text in code editors
TESSERACT_CONFIG = '--psm 6 --oem 3 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789()[]{}:;,.\"\'`~!@#$%^&*-_=+/<>?\\|"'


def get_image(image_path=None):
    """Get image either from file path or by taking a screenshot"""
    if image_path:
        try:
            return Image.open(image_path)
        except Exception as e:
            print(f"Error opening image file: {e}")
            print("Falling back to screenshot...")

    # Add a countdown before taking the screenshot
    print("Taking screenshot in:")
    for i in range(10, 0, -1):
        print(f"{i}...")
        time.sleep(1)

    print("Capturing screenshot now!")
    # Take a screenshot if no image provided or error loading file
    return pyautogui.screenshot()


def is_likely_code_line(text):
    """Determine if a line is likely to be code based on common code patterns"""
    # Common programming keywords and syntax patterns
    code_patterns = [
        # Common syntax patterns
        r'[{}\[\]()<>]',  # Brackets and braces
        r'(if|else|for|while|def|class|return|import|from)\b',  # Keywords
        r'(==|!=|<=|>=|=>|->|\.\.\.|\+=|-=|\*=|/=)',  # Operators
        r'(["\']{1,3})',  # String quotes
        r'(#|//|/\*|\*/)',  # Comments
        r'^[ \t]*\w+[ \t]*=',  # Variable assignments
        r'^[ \t]*\w+\(',  # Function calls
        r'^[ \t]*@\w+',  # Decorators
        r'\w+\.\w+\(',  # Method calls
        r'\w+\:\:?\w+',  # C++/Java class methods or Python type hints
        r'^[ \t]*(public|private|protected|static|final)',  # Java/C# modifiers
        r'^\s*(\d+\.|\*|\-|\+)\s+\w+',  # Markdown list items (to exclude)
        r'^\s*>',  # Markdown blockquotes (to exclude)
    ]

    # Check for any code pattern
    for pattern in code_patterns:
        if re.search(pattern, text):
            return True

    # Check for excessive indentation (common in code)
    if re.match(r'^ {4,}|\t+', text):
        return True

    # Check for consistent indentation patterns
    if re.match(r'^(\s{2,4}|\t)+\w+', text):
        return True

    return False


def find_code_lines_in_image(image):
    """Find lines of code in the image using OCR with improved horizontal position separation"""
    # Use image_to_data with detailed output
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config=TESSERACT_CONFIG)

    # Group text elements by their vertical position with tolerance
    # This allows for slight vertical misalignment within the same line
    vertical_tolerance = 5  # pixels
    horizontal_gap_threshold = 20  # pixels - to identify separate word groups

    # Create a mapping of y-positions to words
    y_positions = {}
    for i in range(len(data['text'])):
        if not data['text'][i].strip():
            continue

        # Use the center of the text box for y-position
        y_center = data['top'][i] + data['height'][i] // 2

        # Find the closest existing y-position within tolerance
        matched_y = None
        for y in y_positions.keys():
            if abs(y - y_center) <= vertical_tolerance:
                matched_y = y
                break

        if matched_y is None:
            y_positions[y_center] = []
        else:
            y_center = matched_y

        # Store word with its horizontal position data
        y_positions[y_center].append({
            'text': data['text'][i],
            'left': data['left'][i],
            'width': data['width'][i],
            'right': data['left'][i] + data['width'][i],
            'top': data['top'][i],
            'height': data['height'][i],
            'conf': float(data['conf'][i]) if data['conf'][i] != '-1' else 0.0
        })

    # Sort y-positions vertically (top to bottom)
    sorted_y_positions = sorted(y_positions.keys())

    # Process each horizontal line to separate words by proximity
    all_lines = []
    for line_num, y_pos in enumerate(sorted_y_positions):
        words = y_positions[y_pos]

        # Sort words by horizontal position (left to right)
        words.sort(key=lambda w: w['left'])

        # Group words by horizontal proximity
        word_groups = []
        current_group = [words[0]] if words else []

        for i in range(1, len(words)):
            prev_word = words[i - 1]
            curr_word = words[i]

            # Calculate gap between current word and previous word
            gap = curr_word['left'] - prev_word['right']

            if gap <= horizontal_gap_threshold:
                # Words are close, add to current group
                current_group.append(curr_word)
            else:
                # Gap is large, start a new group
                if current_group:
                    word_groups.append(current_group)
                current_group = [curr_word]

        # Add the last group if it exists
        if current_group:
            word_groups.append(current_group)

        # Create a separate line entry for each word group
        for group_idx, word_group in enumerate(word_groups):
            # Calculate group boundaries
            left = min(word['left'] for word in word_group)
            top = min(word['top'] for word in word_group)
            right = max(word['right'] for word in word_group)
            width = right - left
            height = max(word['height'] for word in word_group)
            bottom = max(word['top'] + word['height'] for word in word_group)

            # Combine texts with proper spacing
            text = ' '.join(word['text'] for word in word_group)

            # Calculate average confidence
            conf_values = [word['conf'] for word in word_group if word['conf'] > 0]
            avg_conf = sum(conf_values) / len(conf_values) if conf_values else 0.0

            # Add to results
            all_lines.append({
                'text': text,
                'left': left,
                'top': top,
                'width': width,
                'height': height,
                'line_num': line_num,
                'group_idx': group_idx,
                'y_pos': y_pos,
                'conf': avg_conf
            })

    # Sort all line segments in reading order (top to bottom, left to right)
    all_lines.sort(key=lambda x: (x['line_num'], x['group_idx']))

    # Now find potential text editor regions based on line consistency
    # Filter lines to only include those that likely contain code
    code_lines = [line for line in all_lines if is_likely_code_line(line['text'])]

    # No code lines found
    if not code_lines:
        return all_lines  # Return all lines if no code detected

    # Find regions of consistent line height and spacing
    regions = []
    current_region = [code_lines[0]] if code_lines else []

    # Parameters for determining consistent regions
    height_variance_threshold = 3  # pixels
    spacing_variance_threshold = 3  # pixels

    for i in range(1, len(code_lines)):
        current_line = code_lines[i]
        prev_line = code_lines[i - 1]

        # Vertical proximity check (same criteria as before)
        height_diff = abs(current_line['height'] - prev_line['height'])
        vertical_spacing = abs(current_line['y_pos'] - prev_line['y_pos'])

        if (height_diff <= height_variance_threshold and
                vertical_spacing < 50):  # Reasonable line spacing
            current_region.append(current_line)
        else:
            # Start a new region if this is a big enough region
            if len(current_region) >= 2:  # At least 2 lines to be considered a block
                regions.append(current_region)
            current_region = [current_line]

    # Add the last region if it's big enough
    if len(current_region) >= 2:
        regions.append(current_region)

    # If no regions found, return all code lines
    if not regions:
        return code_lines

    # Score regions based on code-like characteristics
    best_region = []
    best_score = -1

    for region in regions:
        # Calculate region score (similar to original)
        line_count_score = min(len(region) / 5.0, 5.0)

        # Score based on code pattern matches
        code_pattern_matches = sum(1 for line in region if is_likely_code_line(line['text']))
        code_score = code_pattern_matches / len(region) * 5.0

        # Calculate consistency of line heights
        heights = [line['height'] for line in region]
        height_mean = sum(heights) / len(heights)
        height_variance = sum((h - height_mean) ** 2 for h in heights) / len(heights)
        height_score = max(0, 3.0 - (height_variance ** 0.5 / height_mean) * 10) if height_mean > 0 else 0

        # Calculate OCR confidence
        conf_values = [line['conf'] for line in region if line['conf'] > 0]
        conf_score = 0
        if conf_values:
            avg_conf = sum(conf_values) / len(conf_values)
            conf_score = avg_conf / 20.0  # Scale from 0-100 to 0-5

        # Total score
        total_score = line_count_score + code_score + height_score + conf_score

        if total_score > best_score:
            best_score = total_score
            best_region = region

    # Return the best region or all code lines if no good region
    return best_region if best_region else code_lines


def clean_and_normalize_text(text):
    """Clean and normalize text for better matching"""
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove common OCR errors for programming symbols
    text = text.replace('{ ', '{').replace(' }', '}')
    text = text.replace('[ ', '[').replace(' ]', ']')
    text = text.replace('( ', '(').replace(' )', ')')
    text = text.replace(' :', ':').replace(' ;', ';')
    text = text.replace(' =', '=').replace('= ', '=')
    # Fix common OCR errors with operators
    text = text.replace('< =', '<=').replace('> =', '>=')
    text = text.replace('= =', '==').replace('! =', '!=')
    text = text.replace('+ =', '+=').replace('- =', '-=')
    text = text.replace('* =', '*=').replace('/ =', '/=')
    # Fix common OCR errors with quotes and brackets
    text = re.sub(r'(?<!\w)\'\'(?!\w)', '"', text)  # '' -> "
    text = re.sub(r'(?<!\w)\"\'(?!\w)', '"', text)  # "' -> "
    text = re.sub(r'(?<!\w)\'\"(?!\w)', '"', text)  # '" -> "
    return text


def similarity_score(text1, text2):
    """Calculate similarity between two strings using sequence matcher"""
    return SequenceMatcher(None, text1, text2).ratio()


def find_best_matching_line(lines_data, search_text):
    """Find the line that best matches the search text"""
    search_text_clean = clean_and_normalize_text(search_text)
    best_match = None
    best_score = 0

    # Print detected lines for debugging
    print(f"\nFound {len(lines_data)} code lines in the text editor:")
    for i, line in enumerate(lines_data):
        clean_line = clean_and_normalize_text(line['text'])
        print(f"{i + 1}: '{clean_line}'")

        # Look for exact substring match first (higher priority)
        if search_text_clean in clean_line:
            # Calculate position score - prefer exact matches near the beginning of the line
            position = clean_line.find(search_text_clean)
            position_factor = 1.0 - (position / len(clean_line)) * 0.3  # Small penalty for matches later in the line

            line['match_type'] = 'exact'
            line['match_score'] = 0.9 + (position_factor * 0.1)  # Score between 0.9-1.0 for exact matches

            # If it's a perfect match for the whole line, return immediately
            if search_text_clean == clean_line:
                line['match_score'] = 1.0
                return line

            # Very good match, but keep checking for potentially better matches
            if line['match_score'] > best_score:
                best_score = line['match_score']
                best_match = line

    # If we already found a very good match, return it
    if best_match and best_match['match_score'] >= 0.95:
        return best_match

    # Calculate fuzzy similarity scores for all lines
    for line in lines_data:
        clean_line = clean_and_normalize_text(line['text'])

        # Skip lines that are much longer or shorter than the search text
        # (prevents matching tiny search strings against huge lines)
        length_ratio = min(len(clean_line), len(search_text_clean)) / max(len(clean_line), len(search_text_clean))
        if length_ratio < 0.3:  # Skip if lengths are too different
            continue

        # Get similarity score for the whole line
        full_score = similarity_score(search_text_clean, clean_line)

        # Check if key words from search are in line (more robust to OCR errors)
        # Focus on words with 4+ characters as they're more distinctive
        search_words = [w for w in search_text_clean.lower().split() if len(w) >= 4]
        if not search_words:  # If no long words, use all words
            search_words = search_text_clean.lower().split()

        line_words = clean_line.lower().split()

        # Count matches using substring matching for better OCR error tolerance
        matched_words = 0
        for sw in search_words:
            for lw in line_words:
                # Look for substantial matches even with OCR errors
                if (sw in lw) or (lw in sw) or similarity_score(sw, lw) > 0.7:
                    matched_words += 1
                    break

        word_match_ratio = matched_words / len(search_words) if search_words else 0

        # Check for special code symbols - these are important in code
        code_symbols = set("()[]{}=+-*/%<>!&|^;:")
        search_symbols = set(c for c in search_text_clean if c in code_symbols)
        line_symbols = set(c for c in clean_line if c in code_symbols)

        symbol_match_ratio = len(search_symbols.intersection(line_symbols)) / len(
            search_symbols) if search_symbols else 1.0

        # Combined score with emphasis on word matches and code symbols
        combined_score = (full_score * 0.3) + (word_match_ratio * 0.5) + (symbol_match_ratio * 0.2)

        if combined_score > best_score:
            best_score = combined_score
            best_match = line
            best_match['match_score'] = combined_score
            best_match['match_type'] = 'fuzzy'

    # Only return matches above a reasonable threshold
    if best_match and best_score > 0.4:
        return best_match

    return None


def click_on_code_line(search_text, image_path=None):
    """Find and click on a specific line of code"""
    # Get the image
    print(f"Looking for text: '{search_text}'")
    image = get_image(image_path)

    # Find all code lines
    lines_data = find_code_lines_in_image(image)

    if not lines_data:
        print("No code lines detected in the image.")
        return False

    # Find the best matching line
    target_line = find_best_matching_line(lines_data, search_text)

    if target_line:
        # Click at the beginning of the line
        x = target_line['left']
        y = target_line['top'] + (target_line['height'] / 2)

        match_type = target_line.get('match_type', 'unknown')
        score = target_line.get('match_score', 1.0)
        print(f"Found line: '{target_line['text']}' (match type: {match_type}, score: {score:.2f})")
        print(f"Clicking at position ({x}, {y})")

        pyautogui.moveTo(x, y)
        time.sleep(0.5)  # Small delay
        pyautogui.click()
        return True
    else:
        print(f"Couldn't find a line matching '{search_text}'")
        return False


def click_and_drag_code_line(search_text, select_whole_line=True, image_path=None):
    """Find a specific line of code in a text editor and click and drag to select it"""
    # Get the image
    print(f"Looking for code: '{search_text}'")
    image = get_image(image_path)

    # Enhanced OCR for text editor content
    try:
        print("Analyzing screenshot for text editor content...")
        # Try to enhance text editor regions by adjusting contrast
        enhanced = image.copy()
        # Apply light contrast enhancement to make text more readable
        try:
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(1.2)  # Slight contrast boost
        except Exception as e:
            print(f"Image enhancement failed: {e}, using original image")
            enhanced = image

        # Find all code lines using the enhanced image
        lines_data = find_code_lines_in_image(enhanced)

        # If few lines found, try with original image
        if len(lines_data) < 3:
            print("Few code lines detected with enhanced image, trying original...")
            lines_data = find_code_lines_in_image(image)
    except Exception as e:
        print(f"Error during image processing: {e}")
        # Fall back to original image processing with minimal processing
        try:
            # Use a simpler approach with the original image
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

            # Process horizontal text elements with new approach
            # Group by vertical position with tolerance
            vertical_tolerance = 5
            y_positions = {}

            for i in range(len(data['text'])):
                if not data['text'][i].strip():
                    continue

                y_center = data['top'][i] + data['height'][i] // 2

                # Find closest y-position
                matched_y = None
                for y in y_positions.keys():
                    if abs(y - y_center) <= vertical_tolerance:
                        matched_y = y
                        break

                if matched_y is None:
                    y_positions[y_center] = []
                else:
                    y_center = matched_y

                y_positions[y_center].append({
                    'text': data['text'][i],
                    'left': data['left'][i],
                    'width': data['width'][i],
                    'right': data['left'][i] + data['width'][i],
                    'top': data['top'][i],
                    'height': data['height'][i]
                })

            # Process each horizontal line
            lines_data = []
            for y_pos in sorted(y_positions.keys()):
                words = y_positions[y_pos]
                words.sort(key=lambda w: w['left'])

                # Basic line data from word group
                left = min(word['left'] for word in words)
                top = min(word['top'] for word in words)
                width = max(word['right'] for word in words) - left
                height = max(word['height'] for word in words)

                # Combine text
                text = ' '.join(word['text'] for word in words)

                lines_data.append({
                    'text': text,
                    'left': left,
                    'top': top,
                    'width': width,
                    'height': height,
                    'y_pos': y_pos,
                    'conf': 0.0  # Default confidence
                })

        except Exception as e2:
            print(f"Fallback text recognition also failed: {e2}")
            return False

    if not lines_data:
        print("No code lines detected in the text editor region.")
        return False

    # Find the best matching line
    target_line = find_best_matching_line(lines_data, search_text)

    if target_line:
        # Get line position
        start_x = target_line['left']
        y = target_line['top'] + (target_line['height'] / 2)

        if select_whole_line:
            # Make sure we select the entire line, with a small margin
            end_x = start_x + target_line['width'] + 5  # Small extra margin
        else:
            clean_line = clean_and_normalize_text(target_line['text'])
            clean_search = clean_and_normalize_text(search_text)

            # Try to find position of search text in line
            start_index = clean_line.lower().find(clean_search.lower())
            if start_index >= 0:
                # Estimate character width based on monospace font assumption
                char_count = len(clean_line)
                if char_count > 0:
                    avg_char_width = target_line['width'] / char_count
                    start_x = target_line['left'] + (start_index * avg_char_width)
                    end_x = start_x + (len(clean_search) * avg_char_width)
                else:
                    # Fall back to selecting whole line
                    end_x = start_x + target_line['width']
            else:
                # Try fuzzy position estimation for partial matches
                matched_portions = []
                search_words = clean_search.split()
                for word in search_words:
                    word_pos = clean_line.lower().find(word.lower())
                    if word_pos >= 0:
                        matched_portions.append((word_pos, word_pos + len(word)))

                if matched_portions:
                    # Get the full range of all matched portions
                    first_pos = min(pos[0] for pos in matched_portions)
                    last_pos = max(pos[1] for pos in matched_portions)

                    # Estimate position based on character width
                    avg_char_width = target_line['width'] / len(clean_line)
                    start_x = target_line['left'] + (first_pos * avg_char_width)
                    end_x = target_line['left'] + (last_pos * avg_char_width)
                else:
                    # Fall back to selecting whole line
                    end_x = start_x + target_line['width']

        match_type = target_line.get('match_type', 'unknown')
        score = target_line.get('match_score', 1.0)
        print(f"Found code line: '{target_line['text']}' (match type: {match_type}, score: {score:.2f})")
        print(f"Selecting from position ({start_x}, {y}) to ({end_x}, {y})")

        # Triple-click method for selecting lines in many text editors
        if select_whole_line:
            try:
                # Method 1: Triple click (works in many text editors)
                pyautogui.moveTo(start_x + 5, y)  # Small offset to ensure we're in text
                time.sleep(0.2)
                pyautogui.tripleClick()
                print("Selected line using triple-click method")
                return True
            except Exception as e:
                print(f"Triple-click failed: {e}, falling back to click and drag")
                # Fall back to click and drag

        # Standard click and drag method
        pyautogui.moveTo(start_x, y)
        time.sleep(0.3)  # Small delay
        pyautogui.mouseDown()
        pyautogui.moveTo(end_x, y, duration=0.4)  # Slightly slower for better selection
        pyautogui.mouseUp()
        return True
    else:
        print(f"Couldn't find a code line matching '{search_text}'")
        return False


def main():
    """Main function to handle user input and execute operations"""
    print("===== Code Selection Tool =====")
    print("This tool will automatically select a code line in your text editor")

    search_text = input("Enter the code line to find and select: ")

    use_file = input("Use image file instead of screenshot? (y/n): ").lower() == 'y'
    image_path = None

    if use_file:
        image_path = input("Enter the path to the image file: ")
        if not image_path:
            print("No path entered, falling back to screenshot.")
            image_path = None

    # Only implement option 2 (select code line)
    click_and_drag_code_line(search_text, select_whole_line=True, image_path=image_path)


if __name__ == "__main__":
    main()