# line_detection.py
import cv2
import numpy as np


def group_horizontal_lines(icons_info, img_height, y_tolerance_factor=0.02):
    """
    Group icons that are on the same horizontal line based on their y-coordinates.

    Args:
        icons_info: List of icon dictionaries
        img_height: Height of the original image
        y_tolerance_factor: Percentage of image height to use as tolerance for line grouping

    Returns:
        List of line groups, where each group contains icons on the same horizontal line
    """
    if not icons_info:
        return []

    # Calculate adaptive y-tolerance based on image height
    y_tolerance = img_height * y_tolerance_factor

    # Sort icons by y-coordinate (vertical position)
    sorted_icons = sorted(icons_info, key=lambda x: x['center_y'])

    horizontal_lines = []
    current_line = [sorted_icons[0]]
    reference_y = sorted_icons[0]['center_y']

    # Group icons into horizontal lines
    for icon in sorted_icons[1:]:
        # If this icon is within tolerance of the current line's reference y-coordinate
        if abs(icon['center_y'] - reference_y) <= y_tolerance:
            current_line.append(icon)
        else:
            # Start a new line
            horizontal_lines.append(current_line)
            current_line = [icon]
            reference_y = icon['center_y']

    # Add the last line if it has icons
    if current_line:
        horizontal_lines.append(current_line)

    # Sort icons within each line by x-coordinate (horizontal position)
    for line in horizontal_lines:
        line.sort(key=lambda x: x['center_x'])

    return horizontal_lines


def create_horizontal_line_visualization(horizontal_lines, image, img_width, img_height):
    """
    Create visualization with one bounding box per line of icons.

    Args:
        horizontal_lines: List of line groups
        image: Original image to draw on
        img_width: Width of the original image
        img_height: Height of the original image

    Returns:
        Image with bounding boxes for horizontal lines
    """
    # Create a copy of the original image
    visualization = image.copy()

    # Draw a single box around each horizontal line of icons
    for i, line in enumerate(horizontal_lines):
        if len(line) > 0:
            # Find the boundaries of the entire line
            min_x = min(icon['x'] for icon in line)
            max_x = max(icon['x'] + icon['w'] for icon in line)
            min_y = min(icon['y'] for icon in line)
            max_y = max(icon['y'] + icon['h'] for icon in line)

            # Add some padding
            padding = 10
            min_x = max(0, min_x - padding)
            min_y = max(0, min_y - padding)
            max_x = min(img_width, max_x + padding)
            max_y = min(img_height, max_y + padding)

            # Draw a rectangle around the entire line
            cv2.rectangle(visualization,
                          (min_x, min_y),
                          (max_x, max_y),
                          (0, 255, 0), 2)  # Green color for lines

            # Add a line identifier (optional)
            cv2.putText(visualization, f"Line {i + 1}",
                        (min_x, min_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return visualization


def save_line_groups(horizontal_lines, original_image, output_directory):
    """
    Extract and save each detected horizontal line as a separate image.

    Args:
        horizontal_lines: List of line groups
        original_image: Original image to extract from
        output_directory: Base directory to save line images

    Returns:
        List of saved file paths
    """
    import os

    # Create a directory for line groups if it doesn't exist
    lines_directory = os.path.join(output_directory, "line_groups")
    if not os.path.exists(lines_directory):
        os.makedirs(lines_directory)

    saved_paths = []

    # Save each line as a separate image
    for i, line in enumerate(horizontal_lines):
        if len(line) > 0:
            # Find the boundaries of the entire line
            min_x = min(icon['x'] for icon in line)
            max_x = max(icon['x'] + icon['w'] for icon in line)
            min_y = min(icon['y'] for icon in line)
            max_y = max(icon['y'] + icon['h'] for icon in line)

            # Add some padding
            padding = 10
            min_x = max(0, min_x - padding)
            min_y = max(0, min_y - padding)
            max_x = min(original_image.shape[1], max_x + padding)
            max_y = min(original_image.shape[0], max_y + padding)

            # Extract the line region from the original image
            line_img = original_image[min_y:max_y, min_x:max_x]

            # Save the line image
            line_path = os.path.join(lines_directory, f"line_{i + 1}.png")
            cv2.imwrite(line_path, line_img)
            saved_paths.append(line_path)

    return saved_paths


def detect_vertical_lines(icons_info, img_width, x_tolerance_factor=0.02):
    """
    Group icons that are in the same vertical line based on their x-coordinates.

    Args:
        icons_info: List of icon dictionaries
        img_width: Width of the original image
        x_tolerance_factor: Percentage of image width to use as tolerance for line grouping

    Returns:
        List of vertical line groups, where each group contains icons in the same vertical column
    """
    if not icons_info:
        return []

    # Calculate adaptive x-tolerance based on image width
    x_tolerance = img_width * x_tolerance_factor

    # Sort icons by x-coordinate (horizontal position)
    sorted_icons = sorted(icons_info, key=lambda x: x['center_x'])

    vertical_lines = []
    current_line = [sorted_icons[0]]
    reference_x = sorted_icons[0]['center_x']

    # Group icons into vertical lines
    for icon in sorted_icons[1:]:
        # If this icon is within tolerance of the current line's reference x-coordinate
        if abs(icon['center_x'] - reference_x) <= x_tolerance:
            current_line.append(icon)
        else:
            # Start a new line
            vertical_lines.append(current_line)
            current_line = [icon]
            reference_x = icon['center_x']

    # Add the last line if it has icons
    if current_line:
        vertical_lines.append(current_line)

    # Sort icons within each line by y-coordinate (vertical position)
    for line in vertical_lines:
        line.sort(key=lambda x: x['center_y'])

    return vertical_lines


def display_and_save_lines(horizontal_lines, vertical_lines, original_image, lines_directory):
    """
    Display and save each individual horizontal and vertical line as a separate image.

    Args:
        horizontal_lines: List of horizontal line groups
        vertical_lines: List of vertical line groups
        original_image: Original image to extract from
        lines_directory: Directory to save individual line images

    Returns:
        None
    """
    import os
    import cv2

    # Create subdirectories for horizontal and vertical lines
    horizontal_dir = os.path.join(lines_directory, "horizontal_lines")
    vertical_dir = os.path.join(lines_directory, "vertical_lines")

    for dir_path in [horizontal_dir, vertical_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created directory: {dir_path}")

    # Save and display horizontal lines
    for i, line in enumerate(horizontal_lines):
        if len(line) > 0:
            # Find the boundaries of the entire line
            min_x = min(icon['x'] for icon in line)
            max_x = max(icon['x'] + icon['w'] for icon in line)
            min_y = min(icon['y'] for icon in line)
            max_y = max(icon['y'] + icon['h'] for icon in line)

            # Add some padding
            padding = 10
            min_x = max(0, min_x - padding)
            min_y = max(0, min_y - padding)
            max_x = min(original_image.shape[1], max_x + padding)
            max_y = min(original_image.shape[0], max_y + padding)

            # Extract the line region from the original image
            line_img = original_image[min_y:max_y, min_x:max_x]

            # Add a label
            label = f"H{i + 1}"
            cv2.putText(line_img, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Save the line image
            line_path = os.path.join(horizontal_dir, f"hline_{i + 1}.png")
            cv2.imwrite(line_path, line_img)

            # Display the line image
            window_name = f"Horizontal Line {i + 1}"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, line_img)
            cv2.resizeWindow(window_name, min(800, line_img.shape[1]), min(200, line_img.shape[0]))

    # Save and display vertical lines
    for i, line in enumerate(vertical_lines):
        if len(line) > 0:
            # Find the boundaries of the entire line
            min_x = min(icon['x'] for icon in line)
            max_x = max(icon['x'] + icon['w'] for icon in line)
            min_y = min(icon['y'] for icon in line)
            max_y = max(icon['y'] + icon['h'] for icon in line)

            # Add some padding
            padding = 10
            min_x = max(0, min_x - padding)
            min_y = max(0, min_y - padding)
            max_x = min(original_image.shape[1], max_x + padding)
            max_y = min(original_image.shape[0], max_y + padding)

            # Extract the line region from the original image
            line_img = original_image[min_y:max_y, min_x:max_x]

            # Add a label
            label = f"V{i + 1}"
            cv2.putText(line_img, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Save the line image
            line_path = os.path.join(vertical_dir, f"vline_{i + 1}.png")
            cv2.imwrite(line_path, line_img)

            # Display the line image
            window_name = f"Vertical Line {i + 1}"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, line_img)
            cv2.resizeWindow(window_name, min(200, line_img.shape[1]), min(800, line_img.shape[0]))

    # Wait for a key press to close all windows
    print(f"\nDisplaying individual line images. Press any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Print summary
    print(f"Saved {len(horizontal_lines)} horizontal lines to: {horizontal_dir}")
    print(f"Saved {len(vertical_lines)} vertical lines to: {vertical_dir}")

def create_vertical_line_visualization(vertical_lines, image, img_width, img_height):
    """
    Create visualization with one bounding box per vertical line of icons.

    Args:
        vertical_lines: List of vertical line groups
        image: Original image to draw on
        img_width: Width of the original image
        img_height: Height of the original image

    Returns:
        Image with bounding boxes for vertical lines
    """
    # Create a copy of the original image
    visualization = image.copy()

    # Draw a single box around each vertical line of icons
    for i, line in enumerate(vertical_lines):
        if len(line) > 0:
            # Find the boundaries of the entire line
            min_x = min(icon['x'] for icon in line)
            max_x = max(icon['x'] + icon['w'] for icon in line)
            min_y = min(icon['y'] for icon in line)
            max_y = max(icon['y'] + icon['h'] for icon in line)

            # Add some padding
            padding = 10
            min_x = max(0, min_x - padding)
            min_y = max(0, min_y - padding)
            max_x = min(img_width, max_x + padding)
            max_y = min(img_height, max_y + padding)

            # Draw a rectangle around the entire line
            cv2.rectangle(visualization,
                          (min_x, min_y),
                          (max_x, max_y),
                          (255, 0, 0), 2)  # Red color for vertical lines

            # Add a line identifier
            cv2.putText(visualization, f"VLine {i + 1}",
                        (min_x, min_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return visualization


def display_and_save_lines(horizontal_lines, vertical_lines, original_image, lines_directory):
    """
    Display and save each individual horizontal and vertical line as a separate image
    with improved visibility and guaranteed display.

    Args:
        horizontal_lines: List of horizontal line groups
        vertical_lines: List of vertical line groups
        original_image: Original image to extract from
        lines_directory: Directory to save individual line images

    Returns:
        None
    """
    import os
    import cv2
    import numpy as np
    import time

    # Create subdirectories for horizontal and vertical lines
    horizontal_dir = os.path.join(lines_directory, "horizontal_lines")
    vertical_dir = os.path.join(lines_directory, "vertical_lines")

    for dir_path in [horizontal_dir, vertical_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created directory: {dir_path}")

    # Create a composite image to show all horizontal lines
    h_lines_count = len([line for line in horizontal_lines if len(line) > 0])
    if h_lines_count > 0:
        # Determine max dimensions for organizing the grid
        max_h_width = 0
        max_h_height = 0
        valid_h_lines = []

        for i, line in enumerate(horizontal_lines):
            if len(line) > 0:
                # Find the boundaries of the entire line
                min_x = min(icon['x'] for icon in line)
                max_x = max(icon['x'] + icon['w'] for icon in line)
                min_y = min(icon['y'] for icon in line)
                max_y = max(icon['y'] + icon['h'] for icon in line)

                # Add padding
                padding = 10
                min_x = max(0, min_x - padding)
                min_y = max(0, min_y - padding)
                max_x = min(original_image.shape[1], max_x + padding)
                max_y = min(original_image.shape[0], max_y + padding)

                width = max_x - min_x
                height = max_y - min_y

                max_h_width = max(max_h_width, width)
                max_h_height = max(max_h_height, height)

                valid_h_lines.append({
                    'index': i,
                    'min_x': min_x,
                    'min_y': min_y,
                    'max_x': max_x,
                    'max_y': max_y,
                    'width': width,
                    'height': height
                })

        # Create a composite image for all horizontal lines
        rows = min(5, len(valid_h_lines))  # Max 5 rows
        cols = (len(valid_h_lines) + rows - 1) // rows  # Calculate needed columns

        # Ensure reasonable size for display
        display_width = min(300, max_h_width)
        display_height = min(150, max_h_height)

        # Create composite image
        h_composite = np.zeros((rows * display_height + rows * 10,
                                cols * display_width + cols * 10, 3),
                               dtype=np.uint8)

        # Process horizontal lines
        for idx, line_info in enumerate(valid_h_lines):
            i = line_info['index']
            min_x = line_info['min_x']
            min_y = line_info['min_y']
            max_x = line_info['max_x']
            max_y = line_info['max_y']

            # Extract the line region from the original image
            line_img = original_image[min_y:max_y, min_x:max_x].copy()

            # Highlight the icons in the line with green rectangles
            for icon in horizontal_lines[i]:
                # Calculate relative position within cropped image
                rel_x = icon['x'] - min_x
                rel_y = icon['y'] - min_y
                cv2.rectangle(line_img,
                              (rel_x, rel_y),
                              (rel_x + icon['w'], rel_y + icon['h']),
                              (0, 255, 0), 2)

            # Add a label
            label = f"H{i + 1}"
            cv2.putText(line_img, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Save the line image with a clear filename
            line_path = os.path.join(horizontal_dir, f"hline_{i + 1}.png")
            cv2.imwrite(line_path, line_img)

            # Resize for composite display
            display_img = cv2.resize(line_img, (display_width, display_height))

            # Calculate position in composite
            row = idx % rows
            col = idx // rows
            y_offset = row * (display_height + 10)
            x_offset = col * (display_width + 10)

            # Place in composite
            h_composite[y_offset:y_offset + display_height,
            x_offset:x_offset + display_width] = display_img

        # Save and display the composite
        h_composite_path = os.path.join(lines_directory, "all_horizontal_lines.png")
        cv2.imwrite(h_composite_path, h_composite)

        # Display with forced foreground
        cv2.namedWindow("All Horizontal Lines", cv2.WINDOW_NORMAL)
        cv2.imshow("All Horizontal Lines", h_composite)
        cv2.setWindowProperty("All Horizontal Lines", cv2.WND_PROP_TOPMOST, 1)
        cv2.resizeWindow("All Horizontal Lines",
                         min(1200, h_composite.shape[1]),
                         min(800, h_composite.shape[0]))

    # Create a composite image to show all vertical lines
    v_lines_count = len([line for line in vertical_lines if len(line) > 0])
    if v_lines_count > 0:
        # Determine max dimensions for organizing the grid
        max_v_width = 0
        max_v_height = 0
        valid_v_lines = []

        for i, line in enumerate(vertical_lines):
            if len(line) > 0:
                # Find the boundaries of the entire line
                min_x = min(icon['x'] for icon in line)
                max_x = max(icon['x'] + icon['w'] for icon in line)
                min_y = min(icon['y'] for icon in line)
                max_y = max(icon['y'] + icon['h'] for icon in line)

                # Add padding
                padding = 10
                min_x = max(0, min_x - padding)
                min_y = max(0, min_y - padding)
                max_x = min(original_image.shape[1], max_x + padding)
                max_y = min(original_image.shape[0], max_y + padding)

                width = max_x - min_x
                height = max_y - min_y

                max_v_width = max(max_v_width, width)
                max_v_height = max(max_v_height, height)

                valid_v_lines.append({
                    'index': i,
                    'min_x': min_x,
                    'min_y': min_y,
                    'max_x': max_x,
                    'max_y': max_y,
                    'width': width,
                    'height': height
                })

        # Create a composite image for all vertical lines
        cols = min(5, len(valid_v_lines))  # Max 5 columns
        rows = (len(valid_v_lines) + cols - 1) // cols  # Calculate needed rows

        # Ensure reasonable size for display
        display_width = min(150, max_v_width)
        display_height = min(300, max_v_height)

        # Create composite image
        v_composite = np.zeros((rows * display_height + rows * 10,
                                cols * display_width + cols * 10, 3),
                               dtype=np.uint8)

        # Process vertical lines
        for idx, line_info in enumerate(valid_v_lines):
            i = line_info['index']
            min_x = line_info['min_x']
            min_y = line_info['min_y']
            max_x = line_info['max_x']
            max_y = line_info['max_y']

            # Extract the line region from the original image
            line_img = original_image[min_y:max_y, min_x:max_x].copy()

            # Highlight the icons in the line with red rectangles
            for icon in vertical_lines[i]:
                # Calculate relative position within cropped image
                rel_x = icon['x'] - min_x
                rel_y = icon['y'] - min_y
                cv2.rectangle(line_img,
                              (rel_x, rel_y),
                              (rel_x + icon['w'], rel_y + icon['h']),
                              (0, 0, 255), 2)

            # Add a label
            label = f"V{i + 1}"
            cv2.putText(line_img, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Save the line image with a clear filename
            line_path = os.path.join(vertical_dir, f"vline_{i + 1}.png")
            cv2.imwrite(line_path, line_img)

            # Resize for composite display
            display_img = cv2.resize(line_img, (display_width, display_height))

            # Calculate position in composite
            row = idx // cols
            col = idx % cols
            y_offset = row * (display_height + 10)
            x_offset = col * (display_width + 10)

            # Place in composite
            v_composite[y_offset:y_offset + display_height,
            x_offset:x_offset + display_width] = display_img

        # Save and display the composite
        v_composite_path = os.path.join(lines_directory, "all_vertical_lines.png")
        cv2.imwrite(v_composite_path, v_composite)

        # Display with forced foreground
        cv2.namedWindow("All Vertical Lines", cv2.WINDOW_NORMAL)
        cv2.imshow("All Vertical Lines", v_composite)
        cv2.setWindowProperty("All Vertical Lines", cv2.WND_PROP_TOPMOST, 1)
        cv2.resizeWindow("All Vertical Lines",
                         min(1200, v_composite.shape[1]),
                         min(800, v_composite.shape[0]))

    # Force the windows to update and wait for user input
    print("\nDisplaying line images. Press any key to continue...")
    cv2.waitKey(1)  # Refresh display
    time.sleep(0.5)  # Give windows time to appear
    cv2.waitKey(0)  # Wait for key press
    cv2.destroyAllWindows()

    # Print summary
    print(f"Saved {h_lines_count} horizontal lines to: {horizontal_dir}")
    print(f"Saved {v_lines_count} vertical lines to: {vertical_dir}")
    print(f"Composite images saved to:")
    print(f"  - {os.path.join(lines_directory, 'all_horizontal_lines.png')}")
    print(f"  - {os.path.join(lines_directory, 'all_vertical_lines.png')}")


def create_non_overlapping_lines(horizontal_lines, vertical_lines, img_width, img_height):
    h_rectangles = []  # (min_x, min_y, max_x, max_y, line_index)
    v_rectangles = []  # (min_x, min_y, max_x, max_y, line_index)

    # Reduce padding for tighter line detection
    h_padding = 2  # Reduced from 5
    v_padding = 1  # Reduced from 3

    # Process horizontal lines first
    for i, line in enumerate(horizontal_lines):
        if len(line) > 0:
            # Get text heights to stay closer to text
            min_x = max(0, min(icon['x'] for icon in line) - h_padding)
            max_x = min(img_width, max(icon['x'] + icon['w'] for icon in line) + h_padding)

            # Calculate the average height of elements to stay closer to text
            heights = [icon['h'] for icon in line]
            avg_height = sum(heights) / len(heights)

            # Use a fraction of the average height for vertical padding
            text_padding = max(1, int(avg_height * 0.1))  # 10% of text height

            min_y = max(0, min(icon['y'] for icon in line) - text_padding)
            max_y = min(img_height, max(icon['y'] + icon['h'] for icon in line) + text_padding)

            h_rectangles.append((min_x, min_y, max_x, max_y, i))

    # Process vertical lines with more precise splitting
    for i, line in enumerate(vertical_lines):
        if len(line) > 0:
            # Initial vertical rectangle
            min_x = max(0, min(icon['x'] for icon in line) - v_padding)
            max_x = min(img_width, max(icon['x'] + icon['w'] for icon in line) + v_padding)
            min_y = max(0, min(icon['y'] for icon in line) - v_padding)
            max_y = min(img_height, max(icon['y'] + icon['h'] for icon in line) + v_padding)

            # Check for overlaps with horizontal rectangles and adjust
            segments = [(min_y, max_y)]  # Initial vertical segment

            for h_rect in h_rectangles:
                h_min_x, h_min_y, h_max_x, h_max_y = h_rect[:4]

                # Only consider horizontal lines that actually intersect with this vertical line
                if not (max_x <= h_min_x or min_x >= h_max_x):
                    # Process each existing segment and split if needed
                    new_segments = []
                    for seg_min_y, seg_max_y in segments:
                        # If segment overlaps with horizontal line
                        if not (seg_max_y <= h_min_y or seg_min_y >= h_max_y):
                            # Add segment above the horizontal line if exists
                            if seg_min_y < h_min_y:
                                new_segments.append((seg_min_y, h_min_y))

                            # Add segment below the horizontal line if exists
                            if seg_max_y > h_max_y:
                                new_segments.append((h_max_y, seg_max_y))
                        else:
                            # No overlap, keep the segment as is
                            new_segments.append((seg_min_y, seg_max_y))

                    segments = new_segments

            # Create rectangle for each remaining segment
            for j, (seg_min_y, seg_max_y) in enumerate(segments):
                if seg_max_y - seg_min_y > 5:  # Only keep segments that are at least 5px tall
                    v_rectangles.append((min_x, seg_min_y, max_x, seg_max_y, i))

    return h_rectangles, v_rectangles

def generate_line_colors(num_lines, saturation=0.9, value=0.8):
    import colorsys
    colors = []
    for i in range(num_lines):
        # Use golden ratio for better color distribution
        h = (i * 0.618033988749895) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, saturation, value)
        # Convert to OpenCV BGR format (0-255)
        color = (int(b * 255), int(g * 255), int(r * 255))
        colors.append(color)
    return colors


def improved_horizontal_line_detection(icons_info, img_height, y_tolerance_factor=0.01):
    """
    Group icons that are on the same horizontal line based on their y-coordinates.
    Uses a more strict approach to avoid detecting separate lines as one.

    Args:
        icons_info: List of icon dictionaries
        img_height: Height of the original image
        y_tolerance_factor: Percentage of image height to use as tolerance for line grouping
                           (lowered to be more strict)

    Returns:
        List of line groups, where each group contains icons on the same horizontal line
    """
    if not icons_info:
        return []

    # Calculate adaptive y-tolerance based on image height
    # Using a smaller tolerance factor for more precise line detection
    y_tolerance = img_height * y_tolerance_factor

    # Sort icons by y-coordinate (vertical position)
    sorted_icons = sorted(icons_info, key=lambda x: x['center_y'])

    horizontal_lines = []
    current_line = [sorted_icons[0]]

    # Instead of using a fixed reference_y, we'll calculate the average y-coordinate
    # of the current line for more accurate line detection
    def get_line_avg_y(line):
        return sum(icon['center_y'] for icon in line) / len(line)

    reference_y = sorted_icons[0]['center_y']

    # Group icons into horizontal lines
    for icon in sorted_icons[1:]:
        # Calculate the average y-coordinate of the current line
        avg_y = get_line_avg_y(current_line)

        # If this icon is within tolerance of the current line's average y-coordinate
        if abs(icon['center_y'] - avg_y) <= y_tolerance:
            current_line.append(icon)
            # Update the reference_y with the average
            reference_y = get_line_avg_y(current_line)
        else:
            # Before starting a new line, check if this icon would be better grouped
            # with the next icon (avoid premature line breaks)
            if icon != sorted_icons[-1]:
                next_index = sorted_icons.index(icon) + 1
                if next_index < len(sorted_icons):
                    next_icon = sorted_icons[next_index]
                    # If the next icon is closer to this icon than the current line
                    if abs(next_icon['center_y'] - icon['center_y']) < abs(icon['center_y'] - avg_y):
                        # Start a new line
                        horizontal_lines.append(current_line)
                        current_line = [icon]
                        reference_y = icon['center_y']
                        continue

            # Start a new line
            horizontal_lines.append(current_line)
            current_line = [icon]
            reference_y = icon['center_y']

    # Add the last line if it has icons
    if current_line:
        horizontal_lines.append(current_line)

    # Sort icons within each line by x-coordinate (horizontal position)
    for line in horizontal_lines:
        line.sort(key=lambda x: x['center_x'])

    # Additional filtering step: merge lines that are very close to each other
    # This helps avoid splitting a single logical line into multiple lines
    i = 0
    while i < len(horizontal_lines) - 1:
        line1 = horizontal_lines[i]
        line2 = horizontal_lines[i + 1]

        line1_avg_y = get_line_avg_y(line1)
        line2_avg_y = get_line_avg_y(line2)

        # If the lines are extremely close, merge them
        if abs(line1_avg_y - line2_avg_y) < y_tolerance / 2:
            horizontal_lines[i] = line1 + line2
            horizontal_lines[i].sort(key=lambda x: x['center_x'])
            horizontal_lines.pop(i + 1)
        else:
            i += 1

    # Second pass filtering: verify that each line is truly horizontal
    # by checking the y-coordinate standard deviation
    import numpy as np
    filtered_lines = []
    for line in horizontal_lines:
        if len(line) < 2:
            filtered_lines.append(line)
            continue

        y_coords = [icon['center_y'] for icon in line]
        y_std = np.std(y_coords)
        y_mean = np.mean(y_coords)

        # If the standard deviation is too high relative to the mean,
        # this might not be a true horizontal line
        if y_std / y_mean > 0.05:  # Threshold for acceptable deviation
            # Try to split the line into more coherent sub-lines
            from sklearn.cluster import KMeans

            # Use K-means to find potential sub-lines based on y-coordinates
            if len(line) > 2:
                # Estimate optimal k using silhouette score
                from sklearn.metrics import silhouette_score

                max_k = min(5, len(line))  # Set an upper limit for k
                best_k = 2
                best_score = -1

                y_array = np.array(y_coords).reshape(-1, 1)

                for k in range(2, max_k + 1):
                    kmeans = KMeans(n_clusters=k, random_state=0).fit(y_array)
                    labels = kmeans.labels_

                    # Skip if any cluster has only one point
                    if min(np.bincount(labels)) < 2:
                        continue

                    # Calculate silhouette score
                    score = silhouette_score(y_array, labels)

                    if score > best_score:
                        best_score = score
                        best_k = k

                # Only split if the silhouette score indicates distinct clusters
                if best_score > 0.5:
                    kmeans = KMeans(n_clusters=best_k, random_state=0).fit(y_array)
                    labels = kmeans.labels_

                    # Create sub-lines based on cluster labels
                    sub_lines = [[] for _ in range(best_k)]
                    for i, icon in enumerate(line):
                        sub_lines[labels[i]].append(icon)

                    # Sort and add sub-lines
                    for sub_line in sub_lines:
                        if sub_line:
                            sub_line.sort(key=lambda x: x['center_x'])
                            filtered_lines.append(sub_line)
                else:
                    filtered_lines.append(line)
            else:
                filtered_lines.append(line)
        else:
            filtered_lines.append(line)

    # Remove any empty lines
    filtered_lines = [line for line in filtered_lines if len(line) > 0]

    # Final sorting of lines by average y-coordinate
    filtered_lines.sort(key=lambda line: get_line_avg_y(line))

    return filtered_lines


def create_non_overlapping_horizontal_visualization(horizontal_lines, image, img_width, img_height):
    """
    Create visualization with one bounding box per line of icons,
    ensuring that the boxes do not overlap.

    Args:
        horizontal_lines: List of line groups
        image: Original image to draw on
        img_width: Width of the original image
        img_height: Height of the original image

    Returns:
        Image with bounding boxes for horizontal lines
    """
    import cv2

    # Create a copy of the original image
    visualization = image.copy()

    # Generate distinct colors for each line
    import colorsys
    colors = []
    for i in range(len(horizontal_lines)):
        # Use HSV color space to generate evenly distributed colors
        h = i / len(horizontal_lines)
        r, g, b = colorsys.hsv_to_rgb(h, 0.8, 0.9)
        # Convert to OpenCV BGR format (0-255)
        color = (int(b * 255), int(g * 255), int(r * 255))
        colors.append(color)

    # Calculate bounding boxes for each line with minimal padding
    horizontal_boxes = []

    for i, line in enumerate(horizontal_lines):
        if len(line) > 0:
            # Find the boundaries of the entire line
            min_x = min(icon['x'] for icon in line)
            max_x = max(icon['x'] + icon['w'] for icon in line)
            min_y = min(icon['y'] for icon in line)
            max_y = max(icon['y'] + icon['h'] for icon in line)

            # Add minimal padding
            h_padding = 2
            v_padding = 1
            min_x = max(0, min_x - h_padding)
            min_y = max(0, min_y - v_padding)
            max_x = min(img_width, max_x + h_padding)
            max_y = min(img_height, max_y + v_padding)

            horizontal_boxes.append({
                'index': i,
                'min_x': min_x,
                'min_y': min_y,
                'max_x': max_x,
                'max_y': max_y,
                'color': colors[i % len(colors)]
            })

    # Sort boxes by y-coordinate for top-to-bottom processing
    horizontal_boxes.sort(key=lambda box: box['min_y'])

    # Draw each horizontal line box
    for box in horizontal_boxes:
        i = box['index']
        min_x = box['min_x']
        min_y = box['min_y']
        max_x = box['max_x']
        max_y = box['max_y']
        color = box['color']

        # Draw a rectangle around the entire line
        cv2.rectangle(visualization,
                      (min_x, min_y),
                      (max_x, max_y),
                      color, 2)

        # Add a line identifier with black outline for better visibility
        text = f"Line {i + 1}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

        # Ensure the text fits within the box
        text_x = min(min_x + 5, max_x - text_size[0] - 5)
        text_y = min_y + text_size[1] + 5

        # Draw text outline for better visibility
        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            cv2.putText(visualization, text,
                        (text_x + dx, text_y + dy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Draw the text in the line's color
        cv2.putText(visualization, text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return visualization, horizontal_boxes


def save_horizontal_line_snapshots(horizontal_lines, original_image, output_dir, boxes=None):
    """
    Extract and save a snapshot of each horizontal line from the original image.

    Args:
        horizontal_lines: List of horizontal line groups
        original_image: Original image to extract from
        output_dir: Directory to save line snapshots
        boxes: Optional pre-calculated bounding boxes

    Returns:
        List of saved file paths
    """
    import os
    import cv2

    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    saved_paths = []

    # Create bounding boxes if not provided
    if boxes is None:
        boxes = []
        for i, line in enumerate(horizontal_lines):
            if len(line) > 0:
                # Find the boundaries of the entire line
                min_x = min(icon['x'] for icon in line)
                max_x = max(icon['x'] + icon['w'] for icon in line)
                min_y = min(icon['y'] for icon in line)
                max_y = max(icon['y'] + icon['h'] for icon in line)

                # Add minimal padding
                h_padding = 2
                v_padding = 1
                min_x = max(0, min_x - h_padding)
                min_y = max(0, min_y - v_padding)
                max_x = min(original_image.shape[1], max_x + h_padding)
                max_y = min(original_image.shape[0], max_y + v_padding)

                boxes.append({
                    'index': i,
                    'min_x': min_x,
                    'min_y': min_y,
                    'max_x': max_x,
                    'max_y': max_y
                })

    # Save each line snapshot
    for box in boxes:
        i = box['index']
        min_x = box['min_x']
        min_y = box['min_y']
        max_x = box['max_x']
        max_y = box['max_y']

        # Extract the line region from the original image
        line_img = original_image[min_y:max_y, min_x:max_x].copy()

        # Create a copy for adding highlights
        highlighted = line_img.copy()

        # Highlight each icon in the line
        line = horizontal_lines[i]
        for icon in line:
            # Calculate relative position within cropped image
            rel_x = icon['x'] - min_x
            rel_y = icon['y'] - min_y
            cv2.rectangle(highlighted,
                          (rel_x, rel_y),
                          (rel_x + icon['w'], rel_y + icon['h']),
                          (0, 255, 0), 1)

        # Save both the original and highlighted versions
        line_path = os.path.join(output_dir, f"hline_{i + 1}.png")
        highlighted_path = os.path.join(output_dir, f"hline_{i + 1}_highlighted.png")

        cv2.imwrite(line_path, line_img)
        cv2.imwrite(highlighted_path, highlighted)

        saved_paths.append(line_path)

        # Print info about the saved line
        print(f"Saved line {i + 1} with {len(line)} icons: {line_path}")

    return saved_paths