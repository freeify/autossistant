import cv2
import numpy as np
import os
import shutil
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from collections import defaultdict
# Define the input image path
image_path = r"C:\Users\EphraimMataranyika\Desktop\Capture_20250509_1135\window_113524.png"

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

# Assign any remaining uncategorized icons
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

# Display results
cv2.namedWindow("UI Icons - Boxes Only", cv2.WINDOW_NORMAL)
cv2.imshow("Original", original)
cv2.imshow("UI Icons - Boxes Only", visualization)
cv2.waitKey(0)
cv2.destroyAllWindows()