import pytesseract
from PIL import Image
import pyautogui
import time
import re
from difflib import SequenceMatcher
import pyperclip
import keyboard
import threading
import sys
import random

# Set path to Tesseract - update this to your installation path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Global flag to indicate if ESC has been pressed
exit_flag = False


def monitor_esc_key():
    """Monitor for ESC key press and set exit flag when detected"""
    global exit_flag
    keyboard.wait('esc')
    print("\nESC pressed. Exiting program...")
    exit_flag = True
    sys.exit(0)
print

def get_screenshot(countdown=10):
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
        line_height = 20

    return best_match, line_height


def count_lines(text):
    """Count the number of lines in a text string"""
    if not text:
        return 0
    return text.count('\n') + 1


def ensure_beginning_of_line(x, y):
    """
    Move to what should be the absolute beginning of the line
    by moving to the far left, then right until text is found.

    This is particularly useful for documents where the text
    might not start exactly at the left margin.
    """
    # First move to the left margin (with some buffer)
    margin_x = 10  # Assuming 10 pixels from left edge is safe
    pyautogui.moveTo(margin_x, y, duration=0.05)

    # Now move right until we find text or reach the original position
    # This advanced approach would require more complex image analysis
    # For now, we'll simply use the detected line_start_x position
    pyautogui.moveTo(x, y, duration=0.05)

    # Click once to position cursor at the beginning of the line
    pyautogui.click()
    time.sleep(0.05)


def fast_drag_select(start_x, start_y, target_lines, line_height=20):
    """
    Improved text selection that continuously drags until reaching the target line count.
    Never stops dragging until the exact line count is reached or max iterations exceeded.
    """
    global exit_flag

    # Ensure we're at the beginning of the line first
    ensure_beginning_of_line(start_x, start_y)

    # Double-click to select first word
    pyautogui.doubleClick()
    time.sleep(0.05)  # Minimal delay

    # Press and hold left mouse button
    pyautogui.mouseDown()

    # Initial estimation
    current_y = start_y + (line_height * target_lines * 0.8)  # Start slightly shorter to avoid overshooting
    pyautogui.moveTo(start_x, current_y, duration=0.05)

    # Copy text without releasing mouse button
    pyautogui.hotkey('ctrl', 'c')
    current_text = pyperclip.paste()
    current_line_count = count_lines(current_text)

    print(f"Initial selection: {current_line_count} lines, target: {target_lines}")

    # Variables for movement control
    max_iterations = 50  # Increased from 30 to allow more attempts
    iterations = 0
    adjustment = line_height // 2
    last_different_line_count = current_line_count
    stuck_counter = 0

    # Maintain a history of positions tried to avoid oscillation
    position_history = []

    # Continue dragging until exact match or max iterations
    while current_line_count != target_lines and iterations < max_iterations and not exit_flag:
        # Calculate lines difference
        lines_difference = target_lines - current_line_count

        # Dynamic adjustment based on how far we are from target
        if abs(lines_difference) > 10:
            adjustment = line_height * 2.5
        elif abs(lines_difference) > 5:
            adjustment = line_height * 1.5
        elif abs(lines_difference) > 2:
            adjustment = line_height * 0.75
        else:
            adjustment = line_height * 0.3  # Fine-grained adjustment when close

        # Add some randomness to break patterns when stuck
        if stuck_counter > 2:
            adjustment *= (0.8 + random.random() * 0.4)  # Randomize by ±20%

        # Calculate movement direction and distance
        move_distance = adjustment * (1 if lines_difference > 0 else -1)

        # Get current position
        current_x, current_y = pyautogui.position()

        # Check if we've been to this position before (within a small tolerance)
        new_position = current_y + move_distance
        position_conflict = False
        tolerance = 3

        for pos in position_history[-10:]:  # Check last 10 positions
            if abs(pos - new_position) < tolerance:
                position_conflict = True
                break

        # If position conflict, add random offset
        if position_conflict:
            move_distance += random.randint(-10, 10)
            new_position = current_y + move_distance

        # Add to position history
        position_history.append(new_position)

        # Move to the new position, maintaining drag selection
        pyautogui.moveTo(current_x, new_position, duration=0.03)  # Faster movement

        # Copy text WITHOUT releasing mouse
        pyautogui.hotkey('ctrl', 'c')
        new_text = pyperclip.paste()
        new_line_count = count_lines(new_text)

        # Only print occasionally to avoid slowing down
        if iterations % 3 == 0 or abs(target_lines - new_line_count) <= 1:
            print(
                f"Selection: {new_line_count} lines (target: {target_lines}), y-pos: {new_position}, diff: {lines_difference}")

        # Handle getting stuck at the same line count
        if new_line_count == current_line_count:
            stuck_counter += 1

            # If stuck for too long, try more aggressive movement
            if stuck_counter >= 5:
                # Try a completely different approach - move to a calculated position
                # based on current line count ratio
                if current_line_count > 0:  # Avoid division by zero
                    ratio = target_lines / current_line_count
                    estimated_position = start_y + (current_y - start_y) * ratio
                    # Cap the jump to prevent moving off-screen
                    jump_distance = min(abs(estimated_position - current_y), 300)
                    jump_direction = 1 if estimated_position > current_y else -1

                    # Apply the jump
                    jump_y = current_y + (jump_distance * jump_direction)
                    pyautogui.moveTo(current_x, jump_y, duration=0.05)

                    # Try scrolling if we might be at edge of viewable area
                    if abs(lines_difference) > 10:
                        # Scroll in the direction we need more text
                        scroll_amount = 5 if lines_difference > 0 else -5
                        pyautogui.scroll(scroll_amount)
                        time.sleep(0.05)

                    # Reset stuck counter
                    stuck_counter = 0
                else:
                    # Desperate measure - large random move
                    desperate_move = random.randint(20, 100) * (1 if lines_difference > 0 else -1)
                    pyautogui.moveRel(0, desperate_move, duration=0.05)
        else:
            # We're making progress, reset stuck counter
            if new_line_count != current_line_count:
                last_different_line_count = current_line_count
                stuck_counter = 0

        # Update current state
        current_text = new_text
        current_line_count = new_line_count
        iterations += 1

        # Very small delay to let system process
        time.sleep(0.01)

    # Only release mouse button after we've achieved target or hit max iterations
    pyautogui.mouseUp()

    # Final copy to ensure we have the complete selection
    pyautogui.hotkey('ctrl', 'c')
    final_text = pyperclip.paste()
    final_line_count = count_lines(final_text)

    print(f"Final selection: {final_line_count} lines (target: {target_lines})")
    if final_line_count != target_lines:
        print(f"Warning: Could not achieve exact line count after {iterations} iterations")

    return final_text, pyautogui.position()[1]


def ocr_guided_text_selection(search_text, target_lines, image_path=None):
    """
    Use OCR to find the starting position of text, then select a specific number of lines.

    Parameters:
    search_text: Text to search for as starting point
    target_lines: Number of lines to select
    image_path: Optional path to image file instead of taking screenshot

    Returns:
    The selected text
    """
    # Find the text using OCR
    match, estimated_line_height = find_text_on_screen(search_text, image_path=image_path)

    if match:
        # Get the starting coordinates (beginning of the line containing the text)
        start_x = match['line_start_x']
        start_y = match['line_start_y']

        print(f"Found text: '{match['text']}' (match score: {match.get('match_score', 0):.2f})")
        print(f"Starting selection at position ({start_x}, {start_y})")
        print(f"Estimated line height: {estimated_line_height:.2f} pixels")

        # Use the fast drag select function with OCR-estimated line height
        selected_text, end_y = fast_drag_select(
            start_x,
            start_y,
            target_lines,
            line_height=max(estimated_line_height, 15)  # Use at least 15px as minimum
        )

        return selected_text, match
    else:
        print(f"Couldn't find text matching '{search_text}'")
        return None, None


def process_multiple_fragments_with_ocr(search_texts, lines_per_fragment, image_path=None):
    """
    Processes multiple text fragments by using OCR to locate each starting point.

    Parameters:
    search_texts: List of text strings to search for (one for each fragment)
    lines_per_fragment: Number of lines to select in each fragment
    image_path: Optional path to image file instead of taking screenshot
    """
    global exit_flag
    all_fragments = []

    # Start from the first search text
    for i, search_text in enumerate(search_texts):
        if exit_flag:
            print("Exiting due to ESC key press")
            break

        print(f"\nProcessing fragment {i + 1}/{len(search_texts)} searching for: '{search_text}'")

        # Use OCR to find and select text
        selected_text, match = ocr_guided_text_selection(
            search_text,
            lines_per_fragment,
            image_path=image_path
        )

        if selected_text:
            # Store the fragment
            all_fragments.append(selected_text)
            current_line_count = count_lines(selected_text)

            # Print preview of selected text
            preview = selected_text[:100] + "..." if len(selected_text) > 100 else selected_text
            print(f"Fragment {i + 1} text ({current_line_count} lines):\n{preview}")

            # Allow time for screen to update before next fragment
            if i < len(search_texts) - 1 and not exit_flag:
                time.sleep(1.5)

    return all_fragments


def main():
    """Main function to handle user input"""
    global exit_flag

    # Start a thread to monitor for ESC key
    esc_thread = threading.Thread(target=monitor_esc_key)
    esc_thread.daemon = True
    esc_thread.start()

    print("===== High-Speed OCR Text Selection Tool =====")
    print("This tool combines OCR text detection with rapid selection adjustments")
    print("for maximum speed and accuracy when selecting specific numbers of lines.")
    print("Press ESC at any time to exit the program.")

    # Choose selection mode
    print("\nSelection modes:")
    print("1. Select text by providing search text and number of lines")
    print("2. Select multiple fragments with different starting points")

    mode = input("Choose mode (1-2): ").strip()

    use_file = input("Use image file instead of taking screenshot? (y/n): ").lower() == 'y'
    image_path = None

    if use_file:
        image_path = input("Enter the path to the image file: ")
        if not image_path:
            print("No path entered, falling back to screenshot.")
            image_path = None

    if mode == "2":
        # Multiple fragments mode
        try:
            num_fragments = int(input("Enter the number of fragments to select: "))
            lines_per_fragment = int(input("Enter the number of lines to select per fragment: "))

            search_texts = []
            for i in range(num_fragments):
                search_text = input(f"Enter search text for fragment {i + 1}: ")
                search_texts.append(search_text)

            # Process multiple fragments
            if not exit_flag:
                fragments = process_multiple_fragments_with_ocr(
                    search_texts,
                    lines_per_fragment,
                    image_path=image_path
                )

                # Print a summary of all fragments
                print("\nAll fragments processed:")
                for i, fragment in enumerate(fragments):
                    lines = count_lines(fragment)
                    print(f"\nFragment {i + 1} ({lines} lines):")
                    preview = fragment[:200] + "..." if len(fragment) > 200 else fragment
                    print(preview)

                # Calculate total lines copied
                total_lines = sum(count_lines(fragment) for fragment in fragments)
                expected_lines = lines_per_fragment * len(search_texts)
                print(f"\nTotal lines copied: {total_lines} (Expected: {expected_lines})")
        except ValueError:
            print("Please enter valid numbers.")
    else:
        # Single selection mode
        search_text = input("Enter the text to search for as starting point: ")
        try:
            target_lines = int(input("Enter the number of lines to select: "))
        except ValueError:
            print("Please enter a valid number. Using default of 5 lines.")
            target_lines = 5

        # Wait before starting to give time to prepare
        print(f"Preparing to select {target_lines} lines starting from text '{search_text}'...")
        print("Starting in 5 seconds...")
        for i in range(5, 0, -1):
            if exit_flag:
                break
            print(f"{i}...")
            time.sleep(1)

        if not exit_flag:
            # Perform OCR-guided selection
            selected_text, match = ocr_guided_text_selection(
                search_text,
                target_lines,
                image_path=image_path
            )

            if selected_text:
                print("\nSelected text:")
                print(selected_text)
                print(f"\nTotal lines selected: {count_lines(selected_text)}")

    print("\nText selection and copying completed")
    print("Press ESC to exit if not already pressed.")


if __name__ == "__main__":
    # Set pyautogui failsafe toggle
    pyautogui.FAILSAFE = True  # Set to False to disable failsafe
    main()