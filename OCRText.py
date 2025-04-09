import cv2
import numpy as np
import os
import shutil
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from collections import defaultdict

# Import the line detection module
from line_detection import (
    group_horizontal_lines,
    create_horizontal_line_visualization,
    save_line_groups,
    detect_vertical_lines,
    create_vertical_line_visualization, display_and_save_lines
)

# Define the input image path
image_path = r"C:\Users\EphraimMataranyika\Pictures\Screenshots\Omni Parser\Screenshot 2025-04-02 142655.png"

# Load the image
image = cv2.imread(image_path)

# Check if image was loaded successfully
if image is None:
    print("Error: Could not load image. Please check the file path.")
    exit()

# Define the output directory and filename
output_directory = r"C:\Users\EphraimMataranyika\Pictures\Screenshots\Omni Parser"
output_filename = "detected_icons.png"
output_path = os.path.join(output_directory, output_filename)

# Create organized directory structure for icon groups
base_icons_directory = os.path.join(output_directory, "organized_icons")
if not os.path.exists(base_icons_directory):
    os.makedirs(base_icons_directory)
    print(f"Created directory: {base_icons_directory}")
else:
    # Clean the directory if it exists
    for file in os.listdir(base_icons_directory):
        file_path = os.path.join(base_icons_directory, file)
        if os.path.isfile(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    print(f"Cleaned directory: {base_icons_directory}")

# Create category directories reflecting common UI organization
categories = [
    "top_navigation", "side_navigation", "bottom_navigation",
    "toolbar", "content_main", "content_secondary",
    "dialog_elements", "grouped_controls", "standalone"
]

category_dirs = {}
for category in categories:
    dir_path = os.path.join(base_icons_directory, category)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    category_dirs[category] = dir_path

# Create a copy of the original image for results
original = image.copy()
img_height, img_width = original.shape[:2]

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply adaptive thresholding
adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)

# Apply Canny edge detection
edges = cv2.Canny(blurred, 30, 150)

# Dilate edges to connect broken contours
kernel = np.ones((3, 3), np.uint8)
dilated_edges = cv2.dilate(edges, kernel, iterations=1)

# Find contours
contours_thresh, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_edges, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours by size
min_area = 100
filtered_contours_thresh = [cnt for cnt in contours_thresh if cv2.contourArea(cnt) > min_area]
filtered_contours_edges = [cnt for cnt in contours_edges if cv2.contourArea(cnt) > min_area]

# Combine both methods and remove duplicates based on IOU
all_contours = filtered_contours_thresh + filtered_contours_edges

# Extract bounding rectangle information for all contours
icons_info = []
for i, cnt in enumerate(all_contours):
    x, y, w, h = cv2.boundingRect(cnt)
    icons_info.append({
        'id': i,
        'x': x,
        'y': y,
        'w': w,
        'h': h,
        'center_x': x + w / 2,
        'center_y': y + h / 2,
        'area': w * h,
        'aspect_ratio': w / h if h > 0 else 0
    })


# Remove duplicates based on high overlap
def calculate_iou(box1, box2):
    # Calculate intersection over union of two bounding boxes
    x1_1, y1_1, w1, h1 = box1['x'], box1['y'], box1['w'], box1['h']
    x2_1, y2_1, w2, h2 = box2['x'], box2['y'], box2['w'], box2['h']

    x1_2, y1_2 = x1_1 + w1, y1_1 + h1
    x2_2, y2_2 = x2_1 + w2, y2_1 + h2

    # Calculate intersection area
    x_left = max(x1_1, x2_1)
    y_top = max(y1_1, y2_1)
    x_right = min(x1_2, x2_2)
    y_bottom = min(y1_2, y2_2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate union area
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area if union_area > 0 else 0.0


# Remove duplicates based on high IOU
non_duplicate_icons = []
iou_threshold = 0.7

for icon in icons_info:
    is_duplicate = False
    for unique_icon in non_duplicate_icons:
        if calculate_iou(icon, unique_icon) > iou_threshold:
            is_duplicate = True
            break

    if not is_duplicate:
        non_duplicate_icons.append(icon)

icons_info = non_duplicate_icons


# Function to detect position-based UI sections
def categorize_by_position(icon, img_width, img_height):
    x, y, w, h = icon['x'], icon['y'], icon['w'], icon['h']
    center_x, center_y = icon['center_x'], icon['center_y']

    # Define region thresholds
    top_threshold = img_height * 0.15
    bottom_threshold = img_height * 0.85
    left_threshold = img_width * 0.15
    right_threshold = img_width * 0.85

    # Top navigation bar
    if y < top_threshold:
        return "top_navigation"

    # Bottom navigation bar
    if y + h > bottom_threshold:
        return "bottom_navigation"

    # Left side navigation
    if x < left_threshold and h > w:
        return "side_navigation"

    # Right side controls/navigation
    if x + w > right_threshold and h > w:
        return "side_navigation"

    # Default to content area
    return None


# Initialize position_category for all icons
for icon in icons_info:
    icon['position_category'] = None

# Identify baseline categories by position
for icon in icons_info:
    position_category = categorize_by_position(icon, img_width, img_height)
    if position_category:
        icon['position_category'] = position_category
    else:
        icon['position_category'] = None

# Group icons using DBSCAN clustering based on spatial proximity
# Extract feature vectors (x, y coordinates with appropriate weighting)
feature_vectors = []
for icon in icons_info:
    # We may want to weight x and y differently depending on the expected layout
    feature_vectors.append([
        icon['center_x'] / img_width,  # Normalize to [0, 1]
        icon['center_y'] / img_height  # Normalize to [0, 1]
    ])

# Apply DBSCAN clustering
eps = 0.05  # Adjust based on your expected icon spacing (normalized)
min_samples = 2  # Minimum icons to form a cluster
db = DBSCAN(eps=eps, min_samples=min_samples).fit(feature_vectors)
labels = db.labels_

# Add cluster labels to icons_info
for i, icon in enumerate(icons_info):
    icon['spatial_cluster'] = int(labels[i]) if labels[i] >= 0 else -1

# Analyze clusters for row/column arrangements and consistent sizes/shapes
clusters = defaultdict(list)
for i, icon in enumerate(icons_info):
    cluster_id = icon['spatial_cluster']
    if cluster_id >= 0:  # Ignore noise points (-1)
        clusters[cluster_id].append(icon)

# Analyze each cluster for arrangement patterns
for cluster_id, cluster_icons in clusters.items():
    if len(cluster_icons) < 2:
        continue

    # Check if the cluster forms a horizontal row
    y_coords = [icon['center_y'] for icon in cluster_icons]
    y_std = np.std(y_coords)
    y_range = max(y_coords) - min(y_coords)

    # Check if the cluster forms a vertical column
    x_coords = [icon['center_x'] for icon in cluster_icons]
    x_std = np.std(x_coords)
    x_range = max(x_coords) - min(x_coords)

    # Calculate size and shape consistency
    widths = [icon['w'] for icon in cluster_icons]
    heights = [icon['h'] for icon in cluster_icons]
    aspect_ratios = [icon['aspect_ratio'] for icon in cluster_icons]

    size_consistency = np.std(widths) / np.mean(widths) + np.std(heights) / np.mean(heights)
    shape_consistency = np.std(aspect_ratios) / np.mean(aspect_ratios) if np.mean(aspect_ratios) > 0 else 0

    # Determine cluster type based on arrangement
    y_tolerance = img_height * 0.03
    x_tolerance = img_width * 0.03

    for icon in cluster_icons:
        # Existing position-based category takes precedence
        if icon['position_category']:
            continue

        # Assign to toolbar if icons are similar size and in a row
        if y_std < y_tolerance and size_consistency < 0.3 and shape_consistency < 0.3:
            icon['position_category'] = "toolbar"
        # Group of controls (buttons, etc.) if consistent in size/shape
        elif size_consistency < 0.3 and shape_consistency < 0.3:
            icon['position_category'] = "grouped_controls"
        # Dialog elements if medium-sized and grouped
        elif 0.01 * img_width * img_height < icon['area'] < 0.05 * img_width * img_height:
            icon['position_category'] = "dialog_elements"
        # Main content elements if larger
        elif icon['area'] > 0.05 * img_width * img_height:
            icon['position_category'] = "content_main"
        # Secondary content elements
        else:
            icon['position_category'] = "content_secondary"

# Assign any remaining uncategorized icons (ensure this runs)
for icon in icons_info:
    if not icon['position_category']:
        icon['position_category'] = "standalone"

# Create a visualization with colors based on categories
category_colors = {
    "top_navigation": (255, 0, 0),  # Red
    "side_navigation": (0, 255, 0),  # Green
    "bottom_navigation": (0, 0, 255),  # Blue
    "toolbar": (255, 255, 0),  # Yellow
    "content_main": (255, 0, 255),  # Magenta
    "content_secondary": (0, 255, 255),  # Cyan
    "dialog_elements": (128, 0, 128),  # Purple
    "grouped_controls": (255, 128, 0),  # Orange
    "standalone": (128, 128, 128)  # Gray
}

# Draw bounding boxes with category colors BUT NO TEXT LABELS
visualization = original.copy()
for icon in icons_info:
    category = icon['position_category']
    color = category_colors.get(category, (255, 255, 255))  # White for unknown

    x, y, w, h = icon['x'], icon['y'], icon['w'], icon['h']
    cv2.rectangle(visualization, (x, y), (x + w, y + h), color, 2)

    # No category label text is added here

# Save visualization results
category_viz_path = os.path.join(output_directory, "ui_categorized_icons.png")
cv2.imwrite(category_viz_path, visualization)

# Extract and save icons by category
for icon in icons_info:
    category = icon['position_category']
    icon_id = icon['id']
    x, y, w, h = icon['x'], icon['y'], icon['w'], icon['h']

    # Extract the icon
    icon_img = original[y:y + h, x:x + w]

    # Save to corresponding category directory
    save_dir = category_dirs.get(category, category_dirs['standalone'])
    icon_path = os.path.join(save_dir, f"icon_{icon_id}_{category}.png")
    cv2.imwrite(icon_path, icon_img)

# Create visual cluster map using different approach
plt.figure(figsize=(10, 8))

# Plot all icons with their cluster colors
colors = plt.cm.rainbow(np.linspace(0, 1, len(set(labels)) + 1))
color_map = {label: colors[i] for i, label in enumerate(sorted(set(labels)))}

for i, icon in enumerate(icons_info):
    cluster_id = icon['spatial_cluster']
    color = color_map.get(cluster_id, colors[0])  # Use first color for noise
    category = icon['position_category']

    plt.scatter(icon['center_x'], icon['center_y'], c=[color], s=50, alpha=0.7)
    plt.text(icon['center_x'], icon['center_y'], f"{icon['id']}", fontsize=8)  # Only show ID, not category

# Add bounding boxes around clusters
for cluster_id, cluster_icons in clusters.items():
    if len(cluster_icons) < 2:
        continue

    # Get cluster bounds
    x_min = min(icon['x'] for icon in cluster_icons)
    y_min = min(icon['y'] for icon in cluster_icons)
    x_max = max(icon['x'] + icon['w'] for icon in cluster_icons)
    y_max = max(icon['y'] + icon['h'] for icon in cluster_icons)

    # Add rectangle with some padding
    padding = 10
    rect = plt.Rectangle((x_min - padding, y_min - padding),
                         (x_max - x_min) + 2 * padding, (y_max - y_min) + 2 * padding,
                         linewidth=1, edgecolor=color_map.get(cluster_id),
                         facecolor='none', alpha=0.5)
    plt.gca().add_patch(rect)

plt.gca().invert_yaxis()  # Invert Y-axis to match image coordinates
plt.title("Icon Clusters")
plt.xlabel("X position")
plt.ylabel("Y position")
plt.tight_layout()

# Save the cluster visualization
cluster_viz_path = os.path.join(output_directory, "icon_clusters.png")
plt.savefig(cluster_viz_path)

# Summary
print(f"Total icons detected: {len(icons_info)}")
print(f"Number of spatial clusters: {len(clusters)}")

# Print category statistics
category_counts = defaultdict(int)
for icon in icons_info:
    category_counts[icon['position_category']] += 1

print("\nIcon categories summary:")
for category, count in category_counts.items():
    print(f"  {category}: {count} icons")

print(f"\nResults saved to:")
print(f"  - Categorized image: {category_viz_path}")
print(f"  - Cluster visualization: {cluster_viz_path}")
print(f"  - Organized icons: {base_icons_directory}")

# ====== IMPROVED LINE DETECTION SECTION WITH NO OVERLAP ======
# Create a dedicated directory for line detection results
lines_directory = os.path.join(output_directory, "line_detection")
if not os.path.exists(lines_directory):
    os.makedirs(lines_directory)
    print(f"Created directory: {lines_directory}")

# Create a new directory specifically for line snapshots
line_snapshots_directory = os.path.join(lines_directory, "line_snapshots")
if not os.path.exists(line_snapshots_directory):
    os.makedirs(line_snapshots_directory)
    print(f"Created directory: {line_snapshots_directory}")

# Further refined tolerance factors for more accurate line detection
horizontal_tolerance_factor = 0.01  # Very strict horizontal grouping
vertical_tolerance_factor = 0.008  # Even stricter vertical grouping

# Detect initial horizontal and vertical lines
initial_horizontal_lines = group_horizontal_lines(icons_info, img_height,
                                                  y_tolerance_factor=horizontal_tolerance_factor)
initial_vertical_lines = detect_vertical_lines(icons_info, img_width, x_tolerance_factor=vertical_tolerance_factor)


# Function to check if two rectangles overlap
def rectangles_overlap(rect1, rect2):
    # rect format: (min_x, min_y, max_x, max_y)
    if rect1[2] <= rect2[0] or rect1[0] >= rect2[2]:  # One rectangle is to the left of the other
        return False
    if rect1[3] <= rect2[1] or rect1[1] >= rect2[3]:  # One rectangle is above the other
        return False
    return True  # Rectangles overlap


# Function to create non-overlapping line bounding boxes
def create_non_overlapping_lines(horizontal_lines, vertical_lines, img_width, img_height):
    h_rectangles = []  # (min_x, min_y, max_x, max_y, line_index)
    v_rectangles = []  # (min_x, min_y, max_x, max_y, line_index)

    # Create initial rectangle definitions with padding
    h_padding = 5
    v_padding = 3

    # Process horizontal lines first (they get priority)
    for i, line in enumerate(horizontal_lines):
        if len(line) > 0:
            min_x = max(0, min(icon['x'] for icon in line) - h_padding)
            max_x = min(img_width, max(icon['x'] + icon['w'] for icon in line) + h_padding)
            min_y = max(0, min(icon['y'] for icon in line) - h_padding)
            max_y = min(img_height, max(icon['y'] + icon['h'] for icon in line) + h_padding)

            h_rectangles.append((min_x, min_y, max_x, max_y, i))

    # Process vertical lines, adjusting to avoid horizontal lines
    for i, line in enumerate(vertical_lines):
        if len(line) > 0:
            # Initial vertical rectangle
            min_x = max(0, min(icon['x'] for icon in line) - v_padding)
            max_x = min(img_width, max(icon['x'] + icon['w'] for icon in line) + v_padding)
            min_y = max(0, min(icon['y'] for icon in line) - v_padding)
            max_y = min(img_height, max(icon['y'] + icon['h'] for icon in line) + v_padding)

            # Check for overlaps with horizontal rectangles and adjust
            original_rect = (min_x, min_y, max_x, max_y)
            adjusted_rect = original_rect

            for h_rect in h_rectangles:
                if rectangles_overlap(adjusted_rect, h_rect[:4]):
                    # Found overlap - we need to split the vertical rectangle
                    h_min_y, h_max_y = h_rect[1], h_rect[3]

                    # Create segments above and below the horizontal line
                    upper_segment = None
                    lower_segment = None

                    if min_y < h_min_y:
                        # There's space above the horizontal line
                        upper_segment = (min_x, min_y, max_x, h_min_y)

                    if max_y > h_max_y:
                        # There's space below the horizontal line
                        lower_segment = (min_x, h_max_y, max_x, max_y)

                    # If we have valid segments, add them as separate vertical rectangles
                    if upper_segment:
                        v_rectangles.append(upper_segment + (i,))

                    if lower_segment:
                        v_rectangles.append(lower_segment + (i,))

                    # Mark this vertical rectangle as processed
                    adjusted_rect = None
                    break

            # If the rectangle wasn't split, add it as is
            if adjusted_rect:
                v_rectangles.append(adjusted_rect + (i,))

    return h_rectangles, v_rectangles


# Add new function to save line snapshots
def save_line_snapshots(h_rects, v_rects, original_image, snapshots_directory):
    """
    Extract and save snapshots of each line (horizontal and vertical) from the original image

    Parameters:
    h_rects -- List of horizontal line rectangles (min_x, min_y, max_x, max_y, line_index)
    v_rects -- List of vertical line rectangles (min_x, min_y, max_x, max_y, line_index)
    original_image -- The original image to extract snapshots from
    snapshots_directory -- Directory to save the snapshots
    """
    # Create subdirectories for horizontal and vertical lines
    h_dir = os.path.join(snapshots_directory, "horizontal")
    v_dir = os.path.join(snapshots_directory, "vertical")

    # Create directories if they don't exist
    for directory in [h_dir, v_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Save horizontal line snapshots
    for i, rect in enumerate(h_rects):
        min_x, min_y, max_x, max_y, line_index = rect

        # Add a small padding for better visibility
        padding = 5
        min_x_pad = max(0, min_x - padding)
        min_y_pad = max(0, min_y - padding)
        max_x_pad = min(original_image.shape[1], max_x + padding)
        max_y_pad = min(original_image.shape[0], max_y + padding)

        # Extract the line snapshot
        line_snapshot = original_image[min_y_pad:max_y_pad, min_x_pad:max_x_pad]

        # Create a path for the snapshot
        snapshot_path = os.path.join(h_dir, f"h_line_{line_index + 1}.png")

        # Save the snapshot
        cv2.imwrite(snapshot_path, line_snapshot)

        print(f"Saved horizontal line {line_index + 1} snapshot to {snapshot_path}")

    # Save vertical line snapshots with segment tracking
    v_segments = {}

    for rect in v_rects:
        min_x, min_y, max_x, max_y, line_index = rect

        # Track segments for each vertical line
        if line_index not in v_segments:
            v_segments[line_index] = 0
        v_segments[line_index] += 1
        segment_num = v_segments[line_index]

        # Add a small padding for better visibility
        padding = 5
        min_x_pad = max(0, min_x - padding)
        min_y_pad = max(0, min_y - padding)
        max_x_pad = min(original_image.shape[1], max_x + padding)
        max_y_pad = min(original_image.shape[0], max_y + padding)

        # Extract the line snapshot
        line_snapshot = original_image[min_y_pad:max_y_pad, min_x_pad:max_x_pad]

        # Create a path for the snapshot - include segment number if needed
        if v_segments[line_index] > 1:
            snapshot_path = os.path.join(v_dir, f"v_line_{line_index + 1}_segment_{segment_num}.png")
        else:
            snapshot_path = os.path.join(v_dir, f"v_line_{line_index + 1}.png")

        # Save the snapshot
        cv2.imwrite(snapshot_path, line_snapshot)

        print(f"Saved vertical line {line_index + 1} segment {segment_num} snapshot to {snapshot_path}")

    # Create a composite visualization (optional)
    h_count = len(set([rect[4] for rect in h_rects]))
    v_count = len(v_segments)

    print(f"Saved {h_count} horizontal line snapshots and {v_count} vertical line snapshots")


# Generate distinct colors for lines
def generate_line_colors(num_lines, saturation=0.8, value=0.9):
    import colorsys
    colors = []
    for i in range(num_lines):
        # Use HSV color space to generate evenly distributed colors
        h = i / num_lines
        r, g, b = colorsys.hsv_to_rgb(h, saturation, value)
        # Convert to OpenCV BGR format (0-255)
        color = (int(b * 255), int(g * 255), int(r * 255))
        colors.append(color)
    return colors


# Get non-overlapping rectangle definitions
h_rects, v_rects = create_non_overlapping_lines(
    initial_horizontal_lines,
    initial_vertical_lines,
    img_width,
    img_height
)

# Generate colors
h_colors = generate_line_colors(len(initial_horizontal_lines))
v_colors = generate_line_colors(len(initial_vertical_lines))

# Create separate visualizations
combined_visualization = original.copy()
horizontal_visualization = original.copy()
vertical_visualization = original.copy()

# Draw horizontal lines
for rect in h_rects:
    min_x, min_y, max_x, max_y, i = rect
    color = h_colors[i % len(h_colors)]

    # Draw on both visualizations
    cv2.rectangle(combined_visualization, (min_x, min_y), (max_x, max_y), color, 2)
    cv2.rectangle(horizontal_visualization, (min_x, min_y), (max_x, max_y), color, 2)

    # Add labels - position inside the rectangle to avoid potential overlap
    text_offset = 15
    cv2.putText(combined_visualization, f"H{i + 1}", (min_x + text_offset, min_y + text_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.putText(horizontal_visualization, f"H{i + 1}", (min_x + text_offset, min_y + text_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Create a mapping from original vertical line index to segment count
v_segment_counts = {}
for _, _, _, _, i in v_rects:
    v_segment_counts[i] = v_segment_counts.get(i, 0) + 1

# Draw vertical lines (potentially split into segments)
for rect in v_rects:
    min_x, min_y, max_x, max_y, i = rect
    color = v_colors[i % len(v_colors)]

    # Draw on both visualizations
    cv2.rectangle(combined_visualization, (min_x, min_y), (max_x, max_y), color, 1)
    cv2.rectangle(vertical_visualization, (min_x, min_y), (max_x, max_y), color, 2)

    # For vertical lines, place label at the bottom if it's tall enough
    if max_y - min_y > 30:  # Only add label if segment is tall enough
        segment_id = v_segment_counts.get(i, 1)
        if segment_id > 1:
            label = f"V{i + 1}.{segment_id}"  # Add segment number if line was split
        else:
            label = f"V{i + 1}"

        cv2.putText(combined_visualization, label, (min_x + 2, max_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.putText(vertical_visualization, label, (min_x + 2, max_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Save the visualizations
horizontal_lines_path = os.path.join(lines_directory, "horizontal_lines_no_overlap.png")
vertical_lines_path = os.path.join(lines_directory, "vertical_lines_no_overlap.png")
combined_lines_path = os.path.join(lines_directory, "combined_lines_no_overlap.png")

cv2.imwrite(horizontal_lines_path, horizontal_visualization)
cv2.imwrite(vertical_lines_path, vertical_visualization)
cv2.imwrite(combined_lines_path, combined_visualization)

# Call the new function to save line snapshots
save_line_snapshots(h_rects, v_rects, original, line_snapshots_directory)

# Display line detection results summary
print("\nImproved Line detection results:")
print(f"  - Number of horizontal lines: {len(initial_horizontal_lines)}")
print(f"  - Number of vertical lines: {len(initial_vertical_lines)}")
print(f"  - Number of vertical line segments (after splitting): {len(v_rects)}")
print(f"  - Results saved to: {lines_directory}")
print(f"  - Line snapshots saved to: {line_snapshots_directory}")

# Optional: Display the visualizations
cv2.namedWindow("Non-Overlapping Line Detection", cv2.WINDOW_NORMAL)
cv2.imshow("Horizontal Lines", horizontal_visualization)
cv2.imshow("Vertical Lines", vertical_visualization)
cv2.imshow("Combined Lines (No Overlap)", combined_visualization)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Add this code at the end of your main script, after the existing line detection section
# ====== DISPLAY AND SAVE INDIVIDUAL LINES ======
print("\nPreparing to display and save individual lines...")

# Import the new function if you're adding it to a separate file
# from line_detection import display_and_save_lines

# Call the function