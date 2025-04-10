import cv2
import numpy as np
import os
import pyautogui
import time
from datetime import datetime


def capture_screenshot(delay=3, output_dir="screenshots", region=None):
    """
    Capture a screenshot after a specified delay.
    Can limit capture to a specific region if provided.

    Args:
        delay: Number of seconds to wait before capturing
        output_dir: Directory to save the screenshot
        region: Optional (left, top, width, height) tuple to capture only a region

    Returns:
        Path to the saved screenshot and cursor position
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate a timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    screenshot_path = os.path.join(output_dir, f"screenshot_{timestamp}.png")

    # Print countdown for user
    print(f"Taking screenshot in {delay} seconds...")
    print("Switch to your code editor and ensure your text is selected")

    for i in range(delay, 0, -1):
        print(f"{i}...")
        time.sleep(1)

    # Get cursor position before capturing screenshot
    cursor_pos = pyautogui.position()

    # Capture the screenshot, possibly limited to a region
    if region:
        screenshot = pyautogui.screenshot(region=region)
        # Adjust cursor position relative to the region
        rel_cursor_pos = (cursor_pos[0] - region[0], cursor_pos[1] - region[1])
        # Only use relative cursor if it's within the captured region
        if 0 <= rel_cursor_pos[0] < region[2] and 0 <= rel_cursor_pos[1] < region[3]:
            cursor_pos = rel_cursor_pos
        else:
            # Cursor is outside the region, set to center of region as fallback
            cursor_pos = (region[2] // 2, region[3] // 2)
    else:
        screenshot = pyautogui.screenshot()

    screenshot.save(screenshot_path)

    print(f"Screenshot saved to: {screenshot_path}")
    print(f"Cursor position: x={cursor_pos[0]}, y={cursor_pos[1]}")

    return screenshot_path, cursor_pos


def detect_text_selection(image_path, cursor_pos=None, output_dir="selection_detection", proximity_threshold=200):
    """
    Detect the text selection (highlighted text) in a screenshot.
    Prioritizes the area with the same color as the cursor position and only considers
    regions near the cursor (within proximity_threshold pixels).

    Args:
        image_path: Path to the screenshot
        cursor_pos: Tuple of (x, y) cursor position
        output_dir: Directory to save results
        proximity_threshold: Maximum distance in pixels to consider from cursor (default: 200px)

    Returns:
        Dictionary with selection boundary coordinates or None if not found
    """
    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return None

    # Get image dimensions
    img_height, img_width = image.shape[:2]

    # Create copies for visualization
    visualization = image.copy()

    # Add cursor position to visualization if provided
    if cursor_pos:
        cursor_x, cursor_y = cursor_pos
        # Draw crosshair at cursor position
        cv2.drawMarker(visualization, (cursor_x, cursor_y),
                       (0, 0, 255), markerType=cv2.MARKER_CROSS,
                       markerSize=20, thickness=2)
        cv2.putText(visualization, "Cursor", (cursor_x + 10, cursor_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Convert to HSV color space to better detect highlights
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color range for common text selection highlights
    # This covers blue, light blue, and gray highlights (common in code editors)

    # Method 1: Try blue highlight detection (many editors use blue highlighting)
    lower_blue = np.array([90, 50, 120])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Method 2: Try light blue highlight detection
    lower_light_blue = np.array([80, 30, 180])
    upper_light_blue = np.array([110, 120, 255])
    light_blue_mask = cv2.inRange(hsv, lower_light_blue, upper_light_blue)

    # Method 3: Try gray highlight detection
    lower_gray = np.array([0, 0, 120])
    upper_gray = np.array([180, 40, 200])
    gray_mask = cv2.inRange(hsv, lower_gray, upper_gray)

    # Method 4: Try detecting any non-white, non-black color that's consistent
    # This is a more general approach for editors with custom highlight colors

    # First create a mask for white and black (text and background in most editors)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])

    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    black_mask = cv2.inRange(hsv, lower_black, upper_black)
    text_bg_mask = cv2.bitwise_or(white_mask, black_mask)

    # Invert to get non-text, non-background pixels
    other_mask = cv2.bitwise_not(text_bg_mask)

    # Combine all masks
    combined_mask = cv2.bitwise_or(blue_mask, light_blue_mask)
    combined_mask = cv2.bitwise_or(combined_mask, gray_mask)
    combined_mask = cv2.bitwise_or(combined_mask, other_mask)

    # Apply morphological operations to enhance detection
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Save the mask for debugging
    mask_path = os.path.join(output_dir, "selection_mask.png")
    cv2.imwrite(mask_path, mask)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by area and ensure only rectangular regions are considered
    min_area = 500  # Minimum area to consider
    valid_contours = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        # Calculate how rectangular the contour is
        x, y, w, h = cv2.boundingRect(cnt)
        rect_area = w * h
        area_ratio = area / rect_area if rect_area > 0 else 0

        # Only allow contours that are very close to rectangular (area_ratio close to 1)
        # For perfect rectangles, the contour area equals the bounding rectangle area
        if area_ratio > 0.85:  # 85% or more similar to a perfect rectangle
            valid_contours.append(cnt)

    # If no valid contours found, try a backup method
    if not valid_contours:
        print("No selection detected with color-based method. Trying alternate method...")

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Use edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Save edges for debugging
        edges_path = os.path.join(output_dir, "edges.png")
        cv2.imwrite(edges_path, edges)

        # Find contours in the edges
        edge_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter by size and shape, ensuring only rectangular regions
        valid_contours = []
        for cnt in edge_contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            # Ensure the contour is actually rectangular
            x, y, w, h = cv2.boundingRect(cnt)
            rect_area = w * h
            area_ratio = area / rect_area if rect_area > 0 else 0
            aspect_ratio = w / h if h > 0 else 0

            # Look for rectangular shapes with appropriate aspect ratio
            # and high similarity to a perfect rectangle
            if 0.5 < aspect_ratio < 5.0 and area_ratio > 0.85:
                valid_contours.append(cnt)

    # Process the contours to find the text selection with same color near cursor
    if valid_contours and cursor_pos:
        cursor_x, cursor_y = cursor_pos

        # Check if cursor is within the image bounds
        if 0 <= cursor_x < img_width and 0 <= cursor_y < img_height:
            # Get the color at cursor position (in HSV for better color comparison)
            cursor_color = hsv[cursor_y, cursor_x]

            # Create a debug visualization showing cursor color
            color_viz = visualization.copy()
            # Draw a rectangle with the exact color at cursor (in BGR)
            cursor_bgr = image[cursor_y, cursor_x]
            cv2.rectangle(color_viz, (10, 10), (50, 50), (int(cursor_bgr[0]), int(cursor_bgr[1]), int(cursor_bgr[2])),
                          -1)
            cv2.putText(color_viz, "Cursor Color", (60, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imwrite(os.path.join(output_dir, "cursor_color.png"), color_viz)

            print(f"Cursor color (HSV): H={cursor_color[0]}, S={cursor_color[1]}, V={cursor_color[2]}")

            # Filter contours by proximity to cursor
            near_contours = []
            for cnt in valid_contours:
                x, y, w, h = cv2.boundingRect(cnt)
                center_x = x + w / 2
                center_y = y + h / 2
                distance = np.sqrt((center_x - cursor_x) ** 2 + (center_y - cursor_y) ** 2)

                # Only consider contours within the proximity threshold
                if distance <= proximity_threshold:
                    near_contours.append({
                        'contour': cnt,
                        'distance': distance,
                        'bounds': (x, y, w, h)
                    })

            print(f"Found {len(near_contours)} contours near cursor (within {proximity_threshold}px)")

            # If no near contours, expand search area
            if not near_contours:
                print(f"No contours found near cursor, expanding search radius to {proximity_threshold * 2}px")
                for cnt in valid_contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    center_x = x + w / 2
                    center_y = y + h / 2
                    distance = np.sqrt((center_x - cursor_x) ** 2 + (center_y - cursor_y) ** 2)

                    # Use expanded threshold
                    if distance <= proximity_threshold * 2:
                        near_contours.append({
                            'contour': cnt,
                            'distance': distance,
                            'bounds': (x, y, w, h)
                        })

            # Process near contours to find the one with most similar color
            if near_contours:
                # For each near contour, analyze its color similarity to cursor
                for i, c_info in enumerate(near_contours):
                    x, y, w, h = c_info['bounds']

                    # Extract region for color analysis
                    region = hsv[y:y + h, x:x + w]

                    # Skip empty regions
                    if region.size == 0:
                        continue

                    # Create a mask for non-background pixels
                    # Assuming background is usually white or very light
                    lower_bg = np.array([0, 0, 200])
                    upper_bg = np.array([180, 30, 255])
                    bg_mask = cv2.inRange(region, lower_bg, upper_bg)
                    non_bg_mask = cv2.bitwise_not(bg_mask)

                    # If all pixels are background, use all pixels
                    if cv2.countNonZero(non_bg_mask) == 0:
                        non_bg_mask = np.ones_like(bg_mask)

                    # Calculate dominant color (mode) of non-background pixels
                    # This focuses on the highlight color, not text or bg
                    h_values = region[:, :, 0][non_bg_mask > 0]
                    s_values = region[:, :, 1][non_bg_mask > 0]
                    v_values = region[:, :, 2][non_bg_mask > 0]

                    if len(h_values) > 0:
                        # Use histogram to find most common hue (dominant color)
                        h_hist = np.bincount(h_values)
                        dominant_h = np.argmax(h_hist)

                        # Calculate color similarity - focus on hue (color) not saturation/value
                        h_diff = min(abs(cursor_color[0] - dominant_h), 180 - abs(cursor_color[0] - dominant_h))
                        color_similarity = 1.0 - min(h_diff / 90.0, 1.0)  # 0-1 scale, higher is more similar

                        # Check cursor inside
                        cursor_inside = (x <= cursor_x <= x + w) and (y <= cursor_y <= y + h)

                        # Store values
                        c_info['color_similarity'] = color_similarity
                        c_info['dominant_hue'] = dominant_h
                        c_info['cursor_inside'] = cursor_inside
                        c_info['area'] = w * h

                        # Debug visualization for this contour
                        region_viz = visualization.copy()
                        cv2.rectangle(region_viz, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        # Convert HSV to BGR for display
                        dominant_color_hsv = np.uint8([[[dominant_h, 255, 255]]])
                        dominant_color_bgr = cv2.cvtColor(dominant_color_hsv, cv2.COLOR_HSV2BGR)[0][0]
                        # Draw color sample rectangle
                        cv2.rectangle(region_viz, (x + w + 10, y), (x + w + 40, y + 20),
                                      (int(dominant_color_bgr[0]), int(dominant_color_bgr[1]),
                                       int(dominant_color_bgr[2])), -1)
                        cv2.putText(region_viz, f"Similarity: {color_similarity:.2f}", (x + w + 50, y + 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        # Save region-specific visualization
                        cv2.imwrite(os.path.join(output_dir, f"region_{i}_color.png"), region_viz)

                        print(f"Region #{i}: dominant hue={dominant_h}, similarity={color_similarity:.2f}, "
                              f"area={c_info['area']}, distance={c_info['distance']:.1f}px, cursor_inside={cursor_inside}")
                    else:
                        # Empty or invalid region
                        c_info['color_similarity'] = 0
                        c_info['cursor_inside'] = False

                # Sort near contours by: 1. cursor inside, 2. color similarity, 3. proximity
                near_contours.sort(key=lambda c: (
                    not c.get('cursor_inside', False),  # First priority: cursor inside
                    -c.get('color_similarity', 0),  # Second: highest color similarity
                    c.get('distance', float('inf'))  # Third: closest to cursor
                ))

                # Get best contour
                if near_contours and 'color_similarity' in near_contours[0]:
                    best_contour = near_contours[0]['contour']
                    x, y, w, h = near_contours[0]['bounds']

                    # Print selected highlight info
                    selected = near_contours[0]
                    print(f"Selected highlight: Color similarity: {selected.get('color_similarity', 0):.2f}, "
                          f"Distance: {selected.get('distance', 0):.0f}px, "
                          f"Cursor inside: {selected.get('cursor_inside', False)}")

                    # Draw all considered contours with color similarity
                    for i, c_info in enumerate(near_contours[1:], 2):
                        if 'bounds' in c_info:
                            cx, cy, cw, ch = c_info['bounds']
                            # Use color to indicate similarity (green=similar, red=different)
                            sim = c_info.get('color_similarity', 0)
                            color = (0, int(255 * sim), int(255 * (1 - sim)))
                            cv2.rectangle(visualization, (cx, cy), (cx + cw, cy + ch), color, 1)
                            # Display similarity score
                            cv2.putText(visualization, f"{sim:.2f}", (cx, cy - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                else:
                    # Fallback to closest contour if color analysis failed
                    near_contours.sort(key=lambda c: c['distance'])
                    best_contour = near_contours[0]['contour']
                    x, y, w, h = near_contours[0]['bounds']
            else:
                # No contours near cursor, use largest overall
                print("No contours found near cursor after expanding search. Using largest contour.")
                valid_contours.sort(key=cv2.contourArea, reverse=True)
                best_contour = valid_contours[0]
                x, y, w, h = cv2.boundingRect(best_contour)
        else:
            # Cursor outside image bounds, fallback to largest contour
            valid_contours.sort(key=cv2.contourArea, reverse=True)
            best_contour = valid_contours[0]
            x, y, w, h = cv2.boundingRect(best_contour)
    elif valid_contours:
        # No cursor position provided, use largest contour
        valid_contours.sort(key=cv2.contourArea, reverse=True)
        best_contour = valid_contours[0]
        x, y, w, h = cv2.boundingRect(best_contour)
    else:
        # No valid contours found
        print("No valid contours found in the image.")
        return None

        # Draw the detected selection on the visualization
        cv2.rectangle(visualization, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(visualization, "Selected Text", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Save the visualization
        viz_path = os.path.join(output_dir, "detected_selection.png")
        cv2.imwrite(viz_path, visualization)

        selection_boundary = {
            'x': x,
            'y': y,
            'width': w,
            'height': h
        }

        return {
            'boundary': selection_boundary,
            'visualization': visualization,
            'visualization_path': viz_path
        }

    print("No text selection detected in the screenshot.")
    return None


def extract_lines_from_selection(image_path, selection_boundary,
                                 output_dir=r"C:\Users\EphraimMataranyika\Pictures\Screenshots\Omni Parser\line_detection"):
    """
    Extract individual lines of text from a selection boundary.

    Args:
        image_path: Path to the screenshot
        selection_boundary: Dictionary with selection coordinates
        output_dir: Directory to save results (default is the specified Omni Parser directory)

    Returns:
        Dictionary with extracted lines and visualization
    """
    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return None

    # Extract the selection region
    x = selection_boundary['x']
    y = selection_boundary['y']
    w = selection_boundary['width']
    h = selection_boundary['height']

    selection_img = image[y:y + h, x:x + w]

    # Create a visualization image
    visualization = image.copy()
    cv2.rectangle(visualization, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Convert selection to grayscale
    gray = cv2.cvtColor(selection_img, cv2.COLOR_BGR2GRAY)

    # Save the selection image
    selection_path = os.path.join(output_dir, "selection.png")
    cv2.imwrite(selection_path, selection_img)

    # Extract line regions directly from the selection
    # Instead of using projection profiles, we'll look for text rows with line breaks

    # Create a binary image with text as white pixels
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Alternative approach using adaptive thresholding
    adaptive_binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 11, 2)

    # Save binary images for debugging
    cv2.imwrite(os.path.join(output_dir, "binary.png"), binary)
    cv2.imwrite(os.path.join(output_dir, "adaptive_binary.png"), adaptive_binary)

    # Save a color distribution visualization of the selection
    # This helps understand the color patterns in the highlight
    if selection_img.shape[0] > 0 and selection_img.shape[1] > 0:
        # Convert to HSV for better color analysis
        selection_hsv = cv2.cvtColor(selection_img, cv2.COLOR_BGR2HSV)

        # Create histograms for Hue and Saturation channels
        h_hist = cv2.calcHist([selection_hsv], [0], None, [180], [0, 180])
        s_hist = cv2.calcHist([selection_hsv], [1], None, [256], [0, 256])

        # Normalize histograms
        h_hist = cv2.normalize(h_hist, h_hist, 0, 255, cv2.NORM_MINMAX)
        s_hist = cv2.normalize(s_hist, s_hist, 0, 255, cv2.NORM_MINMAX)

        # Create histogram visualizations
        h_hist_img = np.zeros((200, 180, 3), np.uint8)
        s_hist_img = np.zeros((200, 256, 3), np.uint8)

        # Draw the histograms
        for i in range(180):
            h_val = int(h_hist[i])
            cv2.line(h_hist_img, (i, 200), (i, 200 - h_val), (255, 0, 0), 1)

        for i in range(256):
            s_val = int(s_hist[i])
            cv2.line(s_hist_img, (i, 200), (i, 200 - s_val), (0, 255, 0), 1)

        # Add labels
        cv2.putText(h_hist_img, "Hue Distribution", (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(s_hist_img, "Saturation Distribution", (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Save the histograms
        cv2.imwrite(os.path.join(output_dir, "hue_histogram.png"), h_hist_img)
        cv2.imwrite(os.path.join(output_dir, "saturation_histogram.png"), s_hist_img)

    # Use whichever binary image gives better results
    binary_img = adaptive_binary

    # Compute horizontal projection (sum pixel values for each row)
    h_projection = np.sum(binary_img, axis=1)

    # Normalize for visualization
    if np.max(h_projection) > 0:
        h_projection_norm = h_projection / np.max(h_projection) * 255
    else:
        h_projection_norm = h_projection

    # Create visualization of horizontal projection
    projection_image = np.zeros((h, 256), dtype=np.uint8)
    for i in range(h):
        projection_image[i, 0:int(h_projection_norm[i])] = 255

    # Save projection visualization
    cv2.imwrite(os.path.join(output_dir, "projection.png"), projection_image)

    # Find line boundaries using the projection
    line_regions = []
    in_line = False
    start_y = 0
    min_line_height = 5  # Minimum height for a line
    threshold = np.max(h_projection) * 0.1  # Threshold for line detection

    for i in range(len(h_projection)):
        if not in_line and h_projection[i] > threshold:
            # Start of a new line
            in_line = True
            start_y = i
        elif in_line and h_projection[i] <= threshold:
            # End of the current line
            in_line = False
            line_height = i - start_y
            if line_height >= min_line_height:
                line_regions.append((start_y, i))

    # Add the last line if we were still in a line at the end
    if in_line and len(h_projection) - start_y >= min_line_height:
        line_regions.append((start_y, len(h_projection)))

    # Extract and save each line
    line_images = []
    for i, (start_y, end_y) in enumerate(line_regions):
        # Calculate absolute coordinates (relative to original image)
        abs_start_y = y + start_y
        abs_end_y = y + end_y

        # Draw rectangle around this line in the visualization
        cv2.rectangle(visualization,
                      (x, abs_start_y),
                      (x + w, abs_end_y),
                      (0, 0, 255), 1)

        # Add line number
        cv2.putText(visualization, f"Line {i + 1}",
                    (x - 70, abs_start_y + (abs_end_y - abs_start_y) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Extract the line image from the selection
        line_img = selection_img[start_y:end_y, :]

        # Save the line image with a naming convention that matches the original code
        line_path = os.path.join(output_dir, f"hline_{i + 1}.png")
        cv2.imwrite(line_path, line_img)

        # Also save a highlighted version
        highlighted_img = line_img.copy()
        # Add a subtle highlight border
        if highlighted_img.shape[0] > 2 and highlighted_img.shape[1] > 2:
            cv2.rectangle(highlighted_img, (0, 0), (highlighted_img.shape[1] - 1, highlighted_img.shape[0] - 1),
                          (0, 255, 0), 1)
        highlighted_path = os.path.join(output_dir, f"hline_{i + 1}_highlighted.png")
        cv2.imwrite(highlighted_path, highlighted_img)

        line_images.append({
            'id': i + 1,
            'image': line_img,
            'position': (x, abs_start_y),
            'size': (w, end_y - start_y),
            'path': line_path
        })

    # Save the visualization
    viz_path = os.path.join(output_dir, "lines_visualization.png")
    cv2.imwrite(viz_path, visualization)

    # Create and save a collage of all lines
    if line_images:
        collage = create_lines_collage(line_images)
        collage_path = os.path.join(output_dir, "all_lines_collage.png")
        cv2.imwrite(collage_path, collage)

    return {
        'selection_boundary': selection_boundary,
        'line_count': len(line_images),
        'line_images': line_images,
        'visualization': visualization,
        'visualization_path': viz_path,
        'projection': projection_image
    }


def create_lines_collage(line_images, max_width=1200):
    """
    Create a collage of all detected text lines.

    Args:
        line_images: List of dictionaries with line images
        max_width: Maximum width for the collage

    Returns:
        Collage image
    """
    if not line_images:
        return None

    # Get maximum dimensions
    max_line_width = max(img['image'].shape[1] for img in line_images)
    max_line_width = min(max_line_width, max_width)  # Cap width

    # Add some padding between lines
    padding = 10

    # Calculate total height needed
    total_height = sum(img['image'].shape[0] for img in line_images) + padding * (len(line_images) - 1)

    # Create blank canvas for collage
    collage = np.ones((total_height, max_line_width, 3), dtype=np.uint8) * 240  # Light gray background

    # Place each line image in the collage
    y_offset = 0
    for line in line_images:
        img = line['image']
        line_id = line['id']
        h, w = img.shape[:2]

        # Resize if too wide
        if w > max_width:
            scale = max_width / w
            new_h = int(h * scale)
            img = cv2.resize(img, (max_width, new_h))
            h, w = img.shape[:2]

        # Place image in collage
        collage[y_offset:y_offset + h, 0:w] = img

        # Add line number
        cv2.putText(collage, f"Line {line_id}",
                    (5, y_offset + h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Add a separator line
        if y_offset + h < total_height:
            cv2.line(collage,
                     (0, y_offset + h + padding // 2),
                     (max_line_width, y_offset + h + padding // 2),
                     (200, 200, 200), 1)

        y_offset += h + padding

    return collage


def simulate_ctrl_a():
    """
    Simulate pressing Ctrl+A to select all text.
    """
    print("Simulating Ctrl+A key press...")
    pyautogui.hotkey('ctrl', 'a')
    # Small delay to ensure selection is complete
    time.sleep(0.5)


def detect_manual_selection(output_dir=r"C:\Users\EphraimMataranyika\Pictures\Screenshots\Omni Parser\line_detection",
                            proximity_threshold=200):
    """
    Detect a manual selection without simulating Ctrl+A.
    Useful when the user has already made a specific selection in their IDE.

    Args:
        output_dir: Directory to save results
        proximity_threshold: Maximum distance in pixels to consider from cursor

    Returns:
        Tuple of (screenshot_path, cursor_pos, selection_info)
    """
    print("Looking for manual text selection...")
    print("Please make sure your text is already selected in your code editor")

    # Get active window information - focus on just the editor area
    try:
        # Get cursor position to identify the active window area
        cursor_pos = pyautogui.position()
        print(f"Current cursor position: x={cursor_pos[0]}, y={cursor_pos[1]}")

        # Option to capture only around the cursor area to avoid processing the entire screen
        # This creates a region centered on the cursor with reasonable size for a code editor
        window_width, window_height = 800, 600  # Smaller region around cursor for more precise detection
        region_left = max(0, cursor_pos[0] - window_width // 2)
        region_top = max(0, cursor_pos[1] - window_height // 2)
        capture_region = (region_left, region_top, window_width, window_height)

        print(f"Capturing region around cursor: {capture_region}")

        # Capture screenshot with the current selection, focusing only on the editor area
        screenshot_path, cursor_pos = capture_screenshot(delay=1, output_dir=output_dir, region=capture_region)
    except Exception as e:
        print(f"Error getting window info: {e}. Falling back to full screen capture.")
        screenshot_path, cursor_pos = capture_screenshot(delay=1, output_dir=output_dir)

    # Detect the text selection with cursor position and specified proximity
    selection_info = detect_text_selection(screenshot_path, cursor_pos, output_dir, proximity_threshold)

    return screenshot_path, cursor_pos, selection_info


def main():
    # Use the specified output directory
    output_dir = r"C:\Users\EphraimMataranyika\Pictures\Screenshots\Omni Parser\line_detection"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Ask user for selection mode and proximity threshold
    print("Selection modes:")
    print("1. Auto select all text (Ctrl+A)")
    print("2. Use my current manual selection")
    print("3. Select specific screen region")
    selection_mode = input("Enter selection mode (1/2/3): ").strip()

    # Ask for proximity threshold
    try:
        proximity = int(input("Enter proximity threshold in pixels (50-500, default 200): ").strip() or "200")
        proximity = max(50, min(500, proximity))  # Limit to reasonable range
    except ValueError:
        proximity = 200
        print("Invalid input. Using default proximity threshold of 200 pixels.")

    print(f"Using proximity threshold of {proximity} pixels")

    if selection_mode == "1":
        # Countdown to give user time to switch to the editor
        print("Please switch to your code editor window")
        print("Waiting 5 seconds before selecting all text...")
        time.sleep(5)

        # Simulate Ctrl+A to select all text
        simulate_ctrl_a()

        # Wait a moment to ensure selection is complete
        time.sleep(1)

        # Capture screenshot with selection
        screenshot_path, cursor_pos = capture_screenshot(delay=1, output_dir=output_dir)

        # Detect the text selection with specified proximity threshold
        selection_info = detect_text_selection(screenshot_path, cursor_pos, output_dir, proximity)
    elif selection_mode == "2":
        # Use the manual selection mode
        screenshot_path, cursor_pos, selection_info = detect_manual_selection(output_dir)

        # Re-run detection with specified proximity if first attempt failed
        if not selection_info:
            print("Re-running detection with specified proximity threshold...")
            selection_info = detect_text_selection(screenshot_path, cursor_pos, output_dir, proximity)
    else:
        # Mode 3: Define a specific rectangular region to capture
        try:
            print("You'll need to specify the coordinates for capture.")
            print("Please position your cursor at the top-left corner of the region to capture.")
            print("Waiting 3 seconds for you to position cursor...")
            time.sleep(3)

            start_pos = pyautogui.position()
            print(f"Top-left recorded at: x={start_pos[0]}, y={start_pos[1]}")

            print("Now move your cursor to the bottom-right corner of the region to capture.")
            print("Waiting 3 seconds for you to position cursor...")
            time.sleep(3)

            end_pos = pyautogui.position()
            print(f"Bottom-right recorded at: x={end_pos[0]}, y={end_pos[1]}")

            # Calculate region dimensions
            region_left = min(start_pos[0], end_pos[0])
            region_top = min(start_pos[1], end_pos[1])
            region_width = abs(end_pos[0] - start_pos[0])
            region_height = abs(end_pos[1] - start_pos[1])

            capture_region = (region_left, region_top, region_width, region_height)
            print(f"Capturing rectangular region: {capture_region}")

            # Capture the specified rectangular region
            screenshot_path, cursor_pos = capture_screenshot(delay=1, output_dir=output_dir, region=capture_region)

            # Detect the text selection with specified proximity threshold
            selection_info = detect_text_selection(screenshot_path, cursor_pos, output_dir, proximity)
        except Exception as e:
            print(f"Error defining capture region: {e}. Falling back to full screen capture.")
            screenshot_path, cursor_pos = capture_screenshot(delay=1, output_dir=output_dir)
            selection_info = detect_text_selection(screenshot_path, cursor_pos, output_dir, proximity)

    if selection_info:
        print(f"Selection detected with dimensions: "
              f"{selection_info['boundary']['width']}x{selection_info['boundary']['height']}")

        # Extract individual lines from the selection and save directly to the specified path
        lines_result = extract_lines_from_selection(
            screenshot_path,
            selection_info['boundary'],
            output_dir
        )

        if lines_result and lines_result['line_count'] > 0:
            print(f"Successfully extracted {lines_result['line_count']} lines from the selection")
            print(f"Results saved to {output_dir}")

            # Display the visualizations
            cv2.imshow("Text Selection", selection_info['visualization'])
            cv2.imshow("Extracted Lines", lines_result['visualization'])
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            print(f"\nLines saved to: {output_dir}")
            print(f"Total lines detected and saved: {lines_result['line_count']}")
            print(f"Individual line images saved with prefix 'hline_' to match your original code naming")
        else:
            print("Failed to extract lines from the selection.")
    else:
        print("No text selection detected. Please ensure your text is selected before running.")


if __name__ == "__main__":
    main()