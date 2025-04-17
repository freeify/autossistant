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


def auto_detect_text_boundaries(ocr_data, screen_size):
    """
    Automatically detect text boundaries based on OCR data distribution

    Returns a tuple of (left, top, right, bottom) boundaries
    """
    screen_width, screen_height = screen_size

    # Default boundaries (conservative) in case detection fails
    default_bounds = (50, 100, screen_width - 50, screen_height - 100)

    # Check if we have valid OCR data to work with
    if not ocr_data or 'text' not in ocr_data or not ocr_data['text'] or len(ocr_data['text']) < 5:
        print("Not enough OCR data for boundary detection, using defaults")
        return default_bounds

    try:
        # Filter out low confidence and empty texts
        valid_indices = []
        for i in range(len(ocr_data['text'])):
            if (ocr_data['text'][i].strip() and
                    float(ocr_data['conf'][i]) >= 60):  # Only items with decent confidence
                valid_indices.append(i)

        if not valid_indices:
            print("No valid text elements found, using default boundaries")
            return default_bounds

        # Find extremes
        left_positions = [ocr_data['left'][i] for i in valid_indices]
        right_positions = [ocr_data['left'][i] + ocr_data['width'][i] for i in valid_indices]
        top_positions = [ocr_data['top'][i] for i in valid_indices]
        bottom_positions = [ocr_data['top'][i] + ocr_data['height'][i] for i in valid_indices]

        # Find the primary text column by looking at text distribution
        # First, get a histogram of x-positions
        x_positions = sorted(left_positions)

        # Find clusters of x positions that are likely to be text columns
        clusters = []
        if x_positions:
            current_cluster = [x_positions[0]]

            for i in range(1, len(x_positions)):
                if x_positions[i] - x_positions[i - 1] <= 50:  # Within same column
                    current_cluster.append(x_positions[i])
                else:
                    clusters.append(current_cluster)
                    current_cluster = [x_positions[i]]

            if current_cluster:
                clusters.append(current_cluster)

        # Find the largest cluster (main text area)
        main_cluster = max(clusters, key=len) if clusters else x_positions

        # Get the bounds of the main text area with margins
        main_left = min(main_cluster) - 20 if main_cluster else min(left_positions)
        main_right = max(right_positions) + 20 if right_positions else screen_width - 50

        # For vertical bounds, we can be more careful about outliers
        # Use 5th and 95th percentiles to avoid extreme outliers
        if len(top_positions) >= 20:
            top_positions_sorted = sorted(top_positions)
            bottom_positions_sorted = sorted(bottom_positions)

            idx_5pct = max(0, int(len(top_positions) * 0.05))
            idx_95pct = min(len(top_positions) - 1, int(len(top_positions) * 0.95))

            top = top_positions_sorted[idx_5pct] - 30
            bottom = bottom_positions_sorted[idx_95pct] + 30
        else:
            # When we have limited data, use all points but with bigger margins
            top = min(top_positions) - 50 if top_positions else 100
            bottom = max(bottom_positions) + 50 if bottom_positions else screen_height - 100

        # Enforce minimum size for text area
        width = max(300, main_right - main_left)
        height = max(200, bottom - top)

        # Make sure boundaries are within screen
        left = max(0, main_left)
        top = max(0, top)
        right = min(screen_width, main_right)
        bottom = min(screen_height, bottom)

        # Final sanity check to ensure we have reasonable boundary size
        if right - left < 200:  # Too narrow
            center = (left + right) / 2
            left = max(0, center - 200)
            right = min(screen_width, center + 200)

        if bottom - top < 150:  # Too short
            center = (top + bottom) / 2
            top = max(0, center - 150)
            bottom = min(screen_height, center + 150)

        return (int(left), int(top), int(right), int(bottom))

    except Exception as e:
        print(f"Error during boundary detection: {e}")
        return default_bounds


def find_text_on_screen(search_text, min_confidence=70, image_path=None):
    """Find text on screen using OCR and automatically detect text boundaries"""
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

    # Get screen dimensions
    screen_size = pyautogui.size()

    # Clean search text
    clean_search = clean_text(search_text)
    print(f"Looking for: '{clean_search}'")

    # Use Tesseract to get text with position data
    try:
        ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

        # Automatically detect text boundaries
        text_area_bounds = auto_detect_text_boundaries(ocr_data, screen_size)
        print(f"Auto-detected text area bounds: {text_area_bounds}")
    except Exception as e:
        print(f"OCR error: {e}")
        # Fallback to default boundaries
        screen_width, screen_height = screen_size
        text_area_bounds = (50, 100, screen_width - 50, screen_height - 100)
        print(f"Using default text area bounds: {text_area_bounds}")
        ocr_data = {"text": [], "conf": [], "left": [], "top": [], "width": [], "height": []}

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
        except (ValueError, TypeError):
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
        if words:  # Check if there are any words
            current_phrase = [words[0]]

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

    return best_match, line_height, text_area_bounds


def count_lines(text):
    """Count the number of lines in a text string"""
    if not text:
        return 0
    return text.count('\n') + 1


def adaptive_drag_select(start_x, start_y, target_lines, line_height=20, text_area_bounds=None):
    """
    Improved text selection that uses adaptive techniques to reach exact target line count.
    Features adaptive movement, automatic error recovery, and intelligent boundary handling.

    Parameters:
    start_x, start_y: Starting coordinates for selection
    target_lines: Number of lines to select
    line_height: Estimated line height in pixels
    text_area_bounds: Text area boundaries (left, top, right, bottom)
    """
    global exit_flag

    # Ensure we have boundaries
    if text_area_bounds is None:
        # Use screen dimensions with margins as default boundaries
        screen_width, screen_height = pyautogui.size()
        text_area_bounds = (50, 100, screen_width - 50, screen_height - 100)

    # Extract boundaries
    left_bound, top_bound, right_bound, bottom_bound = text_area_bounds

    try:
        # Move to the starting position and double-click to start selection
        pyautogui.moveTo(start_x, start_y, duration=0.05)
        pyautogui.doubleClick()
        time.sleep(0.1)  # Slightly longer initial delay for reliable selection start

        # Press and hold left mouse button
        pyautogui.mouseDown()

        # Initial smart estimation - calculate based on line height and target
        # Start conservatively to avoid overshooting
        current_y = start_y + (line_height * target_lines * 0.7)
        # Constrain to text area boundaries
        current_y = max(top_bound + 10, min(current_y, bottom_bound - 10))

        # Move to initial position with easing for smoother movement
        pyautogui.moveTo(start_x, current_y, duration=0.05)

        # Copy text immediately and check
        pyautogui.hotkey('ctrl', 'c')
        time.sleep(0.05)  # Small delay to ensure copy completes
        current_text = pyperclip.paste()
        current_line_count = count_lines(current_text)

        print(f"Initial selection: {current_line_count} lines, target: {target_lines}")

        # Advanced variables for adaptive selection
        max_iterations = 60  # Increased for more thorough attempts
        iterations = 0
        adjustment_factor = line_height // 2

        # Track recent positions and results to detect patterns and stalls
        position_history = []
        line_count_history = []

        # Learning variables
        learning_rate = 0.8  # Controls how aggressively we adjust (starts high)
        successful_moves = 0
        failed_moves = 0

        # Reset movement when stuck
        consecutive_failures = 0
        last_successful_y = current_y

        # For exponential backoff when stuck
        backoff_factor = 1.0

        # Success threshold counter - require multiple consecutive successes
        success_count = 0
        required_success_count = 2  # Number of consecutive successes required

        # Main adaptive loop
        while (
                current_line_count != target_lines or success_count < required_success_count) and iterations < max_iterations and not exit_flag:
            # Calculate lines difference
            lines_difference = target_lines - current_line_count

            # Update histories for pattern detection
            position_history.append(current_y)
            line_count_history.append(current_line_count)
            if len(position_history) > 10:
                position_history = position_history[-10:]
                line_count_history = line_count_history[-10:]

            # Check if we've reached the target
            if current_line_count == target_lines:
                success_count += 1
                print(f"Target reached ({success_count}/{required_success_count} confirmations)")
            else:
                success_count = 0  # Reset on any failure

            # Early exit check
            if current_line_count == target_lines and success_count >= required_success_count:
                break

            # Pattern detection
            pattern_detected = False
            if len(line_count_history) >= 6:
                # Check for oscillation (same values repeating)
                last_four = line_count_history[-4:]
                if len(set(last_four)) <= 2 and last_four[0] != target_lines:
                    pattern_detected = True
                    print("Pattern detected: oscillation - adjusting strategy")

                    # If oscillating, reduce learning rate dramatically
                    learning_rate *= 0.5

                    # Try a random micro-adjustment to break out of the pattern
                    jitter = random.randint(1, max(2, int(line_height / 5))) * (1 if random.random() > 0.5 else -1)
                    current_x, current_y = pyautogui.position()
                    new_y = min(bottom_bound - 10, max(top_bound + 10, current_y + jitter))
                    pyautogui.moveTo(current_x, new_y, duration=0.02)

            # If multiple consecutive failures with no progress, try more aggressive measures
            if not pattern_detected and current_line_count != target_lines:
                # Calculate adaptive adjustment size based on distance from target
                if abs(lines_difference) > 10:
                    adjustment = adjustment_factor * 0.8 * learning_rate  # Reduced to prevent overshooting
                elif abs(lines_difference) > 5:
                    adjustment = adjustment_factor * 0.6 * learning_rate
                elif abs(lines_difference) > 2:
                    adjustment = adjustment_factor * 0.4 * learning_rate
                else:
                    adjustment = adjustment_factor * 0.2 * learning_rate  # Very fine for close distances

                # Dynamic back-off when we're stuck
                if consecutive_failures >= 3:
                    # Increase backoff factor to try different step sizes
                    backoff_factor = min(3.0, backoff_factor * 1.5)
                    adjustment *= backoff_factor
                    print(f"Stuck detected: increasing adjustment with backoff={backoff_factor:.2f}")

                    # After several failures, try going back to last known good position
                    if consecutive_failures >= 5 and consecutive_failures % 2 == 1:
                        print("Multiple failures: reverting to last successful position")
                        new_y = last_successful_y
                        pyautogui.moveTo(start_x, new_y, duration=0.05)

                    # Reset learning rate occasionally when stuck to try fresh approach
                    if consecutive_failures >= 7:
                        learning_rate = min(0.9, learning_rate * 1.5)  # Increase learning rate to escape local minimum
                        print(f"Multiple failures: resetting learning rate to {learning_rate:.2f}")

                # Calculate move direction and distance with sign
                move_distance = adjustment * (1 if lines_difference > 0 else -1)

                # Apply the movement
                current_x, current_y = pyautogui.position()
                new_y = min(bottom_bound - 10, max(top_bound + 10, current_y + move_distance))

                # Check if we're too close to bounds, if so adjust strategy
                if abs(new_y - top_bound) < 20 or abs(new_y - bottom_bound) < 20:
                    print("Near boundary - adjusting movement strategy")
                    # When near boundaries, use smaller movements
                    new_y = current_y + (move_distance * 0.5)
                    new_y = min(bottom_bound - 15, max(top_bound + 15, new_y))

                # Apply the move
                pyautogui.moveTo(current_x, new_y, duration=0.02)

            # Copy and check result after movement
            pyautogui.hotkey('ctrl', 'c')
            time.sleep(0.03)  # Small delay for copy
            new_text = pyperclip.paste()
            new_line_count = count_lines(new_text)

            # Print status occasionally or when close to target
            if iterations % 3 == 0 or abs(target_lines - new_line_count) <= 3:
                print(f"Selection: {new_line_count} lines (target: {target_lines})")

            # Check if we made progress
            if new_line_count == current_line_count:
                consecutive_failures += 1
                failed_moves += 1

                # Try emergency measures if stuck in the same place
                if consecutive_failures == 1:
                    # First try: small random adjustment
                    random_adjustment = random.randint(1, 5) * (1 if lines_difference > 0 else -1)
                    current_x, current_y = pyautogui.position()
                    new_y = min(bottom_bound - 10, max(top_bound + 10, current_y + random_adjustment))
                    pyautogui.moveTo(current_x, new_y, duration=0.02)
                elif consecutive_failures >= 2:
                    # Try more significant adjustments with increasing magnitude
                    emergency_factor = consecutive_failures * 0.5
                    emergency_move = line_height * 0.3 * emergency_factor * (1 if lines_difference > 0 else -1)
                    current_x, current_y = pyautogui.position()
                    new_y = min(bottom_bound - 10, max(top_bound + 10, current_y + emergency_move))
                    pyautogui.moveTo(current_x, new_y, duration=0.02)
                    print(f"Emergency adjustment: {emergency_move:.1f}px")

                # Try another copy after emergency moves
                pyautogui.hotkey('ctrl', 'c')
                time.sleep(0.03)
                new_text = pyperclip.paste()
                new_line_count = count_lines(new_text)
            else:
                # Success! We're making progress
                consecutive_failures = 0
                backoff_factor = 1.0  # Reset backoff
                successful_moves += 1

                # If we made progress in right direction, this is a good reference point
                if (lines_difference > 0 and new_line_count > current_line_count) or \
                        (lines_difference < 0 and new_line_count < current_line_count):
                    last_successful_y = pyautogui.position()[1]

                # Dynamic learning rate adjustment based on success/failure ratio
                if iterations > 5 and (successful_moves + failed_moves) > 0:
                    success_ratio = successful_moves / (successful_moves + failed_moves)
                    # Adjust learning rate - higher for more successes, lower for more failures
                    if success_ratio > 0.7:
                        learning_rate = min(0.9, learning_rate * 1.1)  # Increase confidence
                    elif success_ratio < 0.3:
                        learning_rate = max(0.1, learning_rate * 0.9)  # Reduce confidence

            # Update state for next iteration
            current_text = new_text
            current_line_count = new_line_count
            iterations += 1

            # Slow down slightly if we're very close to prevent overshooting
            if abs(target_lines - current_line_count) <= 1:
                time.sleep(0.02)  # Extra tiny delay for precision

        # Release mouse button once done
        pyautogui.mouseUp()

        # Final copy to ensure we have the complete selection
        pyautogui.hotkey('ctrl', 'c')
        time.sleep(0.05)
        final_text = pyperclip.paste()
        final_line_count = count_lines(final_text)

        # Verify we got exactly what we wanted
        if final_line_count == target_lines:
            print(f"SUCCESS! Final selection: {final_line_count} lines (exact match)")
        else:
            # Last attempt for precision if we're close
            if abs(final_line_count - target_lines) <= 3 and not exit_flag:
                print(f"Final adjustment needed: current={final_line_count}, target={target_lines}")

                # Final precision selection attempt
                pyautogui.mouseDown()
                lines_difference = target_lines - final_line_count

                # Super fine adjustments for final precision
                for attempt in range(abs(lines_difference) * 2):  # Extra attempts for precision
                    if exit_flag:
                        break

                    # Calculate micro-adjustment
                    fine_adjustment = max(1, line_height // 8) * (1 if lines_difference > 0 else -1)

                    # Apply micro-adjustment
                    current_x, current_y = pyautogui.position()
                    new_y = min(bottom_bound - 5, max(top_bound + 5, current_y + fine_adjustment))
                    pyautogui.moveTo(current_x, new_y, duration=0.02)

                    # Check result
                    pyautogui.hotkey('ctrl', 'c')
                    time.sleep(0.03)
                    check_text = pyperclip.paste()
                    check_lines = count_lines(check_text)

                    # If we reached target, success!
                    if check_lines == target_lines:
                        final_text = check_text
                        final_line_count = check_lines
                        print(f"Final adjustment successful: {final_line_count} lines")
                        break

                    # If we overshot, reverse direction with smaller step
                    if (lines_difference > 0 and check_lines > target_lines) or \
                            (lines_difference < 0 and check_lines < target_lines):
                        lines_difference *= -0.5  # Reverse with half magnitude

                pyautogui.mouseUp()

            print(f"Final selection: {final_line_count} lines (target was {target_lines})")

        return final_text, pyautogui.position()[1]

    except Exception as e:
        print(f"Error during drag selection: {e}")
        # Make sure to release the mouse button in case of error
        try:
            pyautogui.mouseUp()
        except:
            pass
        return "", start_y


def ocr_guided_text_selection(search_text, target_lines, image_path=None):
    """
    Use OCR to find the starting position of text, then select a specific number of lines.
    Automatically detects text area boundaries.

    Parameters:
    search_text: Text to search for as starting point
    target_lines: Number of lines to select
    image_path: Optional path to image file instead of taking screenshot

    Returns:
    The selected text and match information
    """
    try:
        # Find the text using OCR
        match, estimated_line_height, text_area_bounds = find_text_on_screen(search_text, image_path=image_path)

        if match:
            # Get the starting coordinates (beginning of the line containing the text)
            start_x = match['line_start_x']
            start_y = match['line_start_y']

            print(f"Found text: '{match['text']}' (match score: {match.get('match_score', 0):.2f})")
            print(f"Starting selection at position ({start_x}, {start_y})")
            print(f"Estimated line height: {estimated_line_height:.2f} pixels")
            print(f"Using text area boundaries: {text_area_bounds}")

            # Use the improved adaptive drag select function
            selected_text, end_y = adaptive_drag_select(
                start_x,
                start_y,
                target_lines,
                line_height=max(estimated_line_height, 15),  # Use at least 15px as minimum
                text_area_bounds=text_area_bounds
            )

            return selected_text, match
        else:
            print(f"Couldn't find text matching '{search_text}'")
            return None, None
    except Exception as e:
        print(f"Error in OCR-guided selection: {e}")
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

    try:
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
            else:
                print(f"Failed to select text for fragment {i + 1}")

        return all_fragments
    except Exception as e:
        print(f"Error processing multiple fragments: {e}")
        return all_fragments


def main():
    """Main function to handle user input"""
    global exit_flag

    try:
        # Start a thread to monitor for ESC key
        esc_thread = threading.Thread(target=monitor_esc_key)
        esc_thread.daemon = True
        esc_thread.start()

        print("===== Advanced OCR Text Selection Tool =====")
        print("This tool automatically detects text boundaries and selects text with high precision.")
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
            print("Starting in 3 seconds...")
            for i in range(3, 0, -1):
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

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Press ESC to exit.")


if __name__ == "__main__":
    # Set pyautogui failsafe toggle
    pyautogui.FAILSAFE = True  # Moving cursor to screen corner will abort program as safety measure
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}")
        print("Program terminated. Press Enter to exit.")
        input()