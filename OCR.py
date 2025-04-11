import pytesseract
from PIL import Image
import pyautogui
import time
import re
from difflib import SequenceMatcher

# Set path to Tesseract - update this to your installation path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def get_screenshot(countdown=3):
    """Take a screenshot after a countdown"""
    print(f"Taking screenshot in {countdown} seconds...")
    for i in range(countdown, 0, -1):
        print(f"{i}...")
        time.sleep(1)

    print("Capturing screenshot now!")
    return pyautogui.screenshot()


def clean_text(text):
    """Clean text for better matching"""
    # Remove extra spaces and normalize whitespace
    return re.sub(r'\s+', ' ', text).strip()


def similarity_score(text1, text2):
    """Calculate similarity between two strings"""
    return SequenceMatcher(None, text1, text2).ratio()


def find_text_on_screen(search_text, min_confidence=70, image_path=None):
    """Find text on screen using OCR"""
    # Get image from file or screenshot
    if image_path:
        try:
            image = Image.open(image_path)
        except Exception as e:
            print(f"Error opening image: {e}")
            print("Taking screenshot instead...")
            image = get_screenshot()
    else:
        image = get_screenshot()

    # Clean search text
    clean_search = clean_text(search_text)
    print(f"Looking for: '{clean_search}'")

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
        clean_line = clean_text(line['text'])

        # First check for exact match
        if clean_search.lower() in clean_line.lower():
            score = 0.9 + (0.1 * len(clean_search) / len(clean_line))
            if score > best_score:
                best_score = score
                best_match = line
                best_match['match_score'] = score

        # Fuzzy match if no exact match found
        if best_score < 0.9:
            score = similarity_score(clean_search.lower(), clean_line.lower())
            if score > best_score:
                best_score = score
                best_match = line
                best_match['match_score'] = score

    return best_match


def click_on_text(search_text, image_path=None):
    """Find and click on text anywhere on screen"""
    # Find the text
    match = find_text_on_screen(search_text, image_path=image_path)

    if match:
        # Get coordinates (center of the text)
        x = match['center_x']
        y = match['center_y']

        print(f"Found text: '{match['text']}' (match score: {match.get('match_score', 0):.2f})")
        print(f"Clicking at position ({x}, {y})")

        # Move mouse and click
        pyautogui.moveTo(x, y)
        time.sleep(0.2)  # Small delay
        pyautogui.click()
        return True, match
    else:
        print(f"Couldn't find text matching '{search_text}'")
        return False, None


def click_at_line_start_and_double_click(search_text, image_path=None):
    """Find text on screen, move to the start of its line and double-click"""
    # Find the text
    match = find_text_on_screen(search_text, image_path=image_path)

    if match:
        # Get line start coordinates
        x = match['line_start_x']
        y = match['line_start_y']

        print(f"Found text: '{match['text']}' (match score: {match.get('match_score', 0):.2f})")
        print(f"Moving to line start at position ({x}, {y}) and double-clicking")

        # Move mouse to the start of the line
        pyautogui.moveTo(x, y)
        time.sleep(0.3)  # Small delay before clicking

        # Perform double click
        pyautogui.doubleClick()
        return True
    else:
        print(f"Couldn't find text matching '{search_text}'")
        return False


def main():
    """Main function to handle user input"""
    print("===== Text Selection Tool =====")
    print("This tool will find text on your screen and can:")
    print("1. Click on the text")
    print("2. Move to the start of the line and double-click (for text selection)")

    search_text = input("Enter the text to find: ")

    action = input("Choose action (1=Click on text, 2=Line selection): ")

    use_file = input("Use image file instead of taking screenshot? (y/n): ").lower() == 'y'
    image_path = None

    if use_file:
        image_path = input("Enter the path to the image file: ")
        if not image_path:
            print("No path entered, falling back to screenshot.")
            image_path = None

    # Execute selected action
    if action == "2":
        click_at_line_start_and_double_click(search_text, image_path=image_path)
    else:
        click_on_text(search_text, image_path=image_path)


if __name__ == "__main__":
    main()