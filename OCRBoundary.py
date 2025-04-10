import cv2
import numpy as np
import os


def detect_lines_and_boundaries(image_path, output_path, debug_path=None):
    """
    Detects horizontal and vertical lines in an image with improved accuracy.

    Parameters:
    - image_path: Path to the input image
    - output_path: Path to save the output image with detected lines
    - debug_path: Optional path to save intermediate processing images
    """
    # Read the image from file
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # Get image dimensions for parameter scaling
    height, width = img.shape[:2]

    # Create debug folder if debug_path is specified
    if debug_path:
        os.makedirs(debug_path, exist_ok=True)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Save grayscale image if debugging
    if debug_path:
        cv2.imwrite(os.path.join(debug_path, "1_gray.png"), gray)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Save blurred image if debugging
    if debug_path:
        cv2.imwrite(os.path.join(debug_path, "2_blurred.png"), blurred)

    # Adaptive thresholding to better handle lighting variations
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # Save binary image if debugging
    if debug_path:
        cv2.imwrite(os.path.join(debug_path, "3_binary.png"), binary)

    # Morphological operations to enhance line structures
    kernel_h = np.ones((1, 15), np.uint8)  # Horizontal kernel
    kernel_v = np.ones((15, 1), np.uint8)  # Vertical kernel

    # Detect horizontal lines
    morph_h = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_h)
    if debug_path:
        cv2.imwrite(os.path.join(debug_path, "4_morph_horizontal.png"), morph_h)

    # Detect vertical lines
    morph_v = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_v)
    if debug_path:
        cv2.imwrite(os.path.join(debug_path, "5_morph_vertical.png"), morph_v)

    # Combine horizontal and vertical line detections
    morph_combined = cv2.bitwise_or(morph_h, morph_v)
    if debug_path:
        cv2.imwrite(os.path.join(debug_path, "6_morph_combined.png"), morph_combined)

    # Use Canny edge detection with parameters scaled to image size
    low_threshold = int(50 * (width / 1000))  # Scale threshold based on image width
    high_threshold = int(150 * (width / 1000))
    edges = cv2.Canny(morph_combined, low_threshold, high_threshold, apertureSize=3)

    if debug_path:
        cv2.imwrite(os.path.join(debug_path, "7_edges.png"), edges)

    # Scale line detection parameters based on image dimensions
    min_line_length = max(50, int(width * 0.05))  # At least 5% of image width
    max_line_gap = max(10, int(width * 0.01))  # At least 1% of image width
    threshold = max(100, int(width * 0.1))  # At least 10% of image width

    # Use HoughLinesP with adaptive parameters
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )

    vertical_lines = []
    horizontal_lines = []

    # Create a line visualization image
    line_vis = np.zeros_like(img)

    # Classify lines based on their angle with improved classification
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calculate angle and length
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # More accurate classification with stricter angle thresholds
            if abs(angle) < 5 or abs(angle - 180) < 5 or abs(angle + 180) < 5:
                horizontal_lines.append((x1, y1, x2, y2, length))
            elif abs(angle - 90) < 5 or abs(angle + 90) < 5:
                vertical_lines.append((x1, y1, x2, y2, length))

    # Sort lines by length (longest first) to prioritize major boundaries
    horizontal_lines.sort(key=lambda x: x[4], reverse=True)
    vertical_lines.sort(key=lambda x: x[4], reverse=True)

    # Create a copy of the original image for output
    output_img = img.copy()

    # Draw horizontal lines in green
    for (x1, y1, x2, y2, _) in horizontal_lines:
        cv2.line(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.line(line_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw vertical lines in blue
    for (x1, y1, x2, y2, _) in vertical_lines:
        cv2.line(output_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.line(line_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Save line visualization if debugging
    if debug_path:
        cv2.imwrite(os.path.join(debug_path, "8_lines_only.png"), line_vis)

    # Save the processed image
    cv2.imwrite(output_path, output_img)
    print(f"Processed image saved to {output_path}")

    # Count detected lines
    print(f"Detected {len(horizontal_lines)} horizontal lines and {len(vertical_lines)} vertical lines")

    # Display results if running in interactive mode
    cv2.imshow("Detected Lines", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return horizontal_lines, vertical_lines


def main():
    image_path = r"C:\Users\EphraimMataranyika\Pictures\Screenshots\Omni Parser\Screenshot 2025-04-10 080625.png"
    output_path = r"C:\Users\EphraimMataranyika\Pictures\Screenshots\Omni Parser\organized_icons\detected_lines.png"
    debug_path = r"C:\Users\EphraimMataranyika\Pictures\Screenshots\Omni Parser\organized_icons\debug"

    detect_lines_and_boundaries(image_path, output_path, debug_path)


if __name__ == "__main__":
    main()