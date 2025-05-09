import platform
import os
import sys
from datetime import datetime
import numpy as np
import cv2
import pyautogui
from collections import defaultdict
from sklearn.cluster import DBSCAN


class MouseElementDetector:
    def __init__(self):
        # Basic setup
        self.system = platform.system()
        print(f"Detected operating system: {self.system}")

        # Colors for visualization
        self.colors = {
            'element_border': (255, 0, 0),  # Red
            'mouse_box': (0, 255, 0),  # Green
            'text': (255, 255, 255),  # White
            'text_background': (0, 0, 0, 128)  # Semi-transparent black
        }

        # Mouse box size (pixels around cursor)
        self.mouse_box_size = 30

        # Minimum and maximum sizes for UI elements (to filter out too small or too large elements)
        self.min_element_area = 25  # Min area in pixels
        self.max_element_area_ratio = 0.5  # Max percentage of screen area

        # Directory to save output
        self.output_directory = self._get_output_directory()
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

    def _get_output_directory(self):
        """Get an appropriate directory to save results based on the platform."""
        # Try to use the desktop or pictures directory
        try:
            if self.system == 'Windows':
                desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
                return os.path.join(desktop, "MouseElementDetector")
            elif self.system == 'Darwin':  # macOS
                pictures = os.path.join(os.path.expanduser('~'), 'Pictures')
                return os.path.join(pictures, "MouseElementDetector")
            elif self.system == 'Linux':
                pictures = os.path.join(os.path.expanduser('~'), 'Pictures')
                if not os.path.exists(pictures):
                    pictures = os.path.join(os.path.expanduser('~'), 'Desktop')
                return os.path.join(pictures, "MouseElementDetector")
        except Exception as e:
            print(f"Could not determine standard save path: {e}")

        # Fall back to current directory
        return os.path.join(os.getcwd(), "MouseElementDetector")

    def _get_save_path(self, filename):
        """Get full path for a file to be saved."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return os.path.join(self.output_directory, f"{filename}_{timestamp}.png")

    def capture_screenshot(self):
        """Take a screenshot and return it as a numpy array for OpenCV."""
        # Take screenshot using pyautogui
        screenshot = pyautogui.screenshot()

        # Convert PIL Image to numpy array (for OpenCV)
        screenshot_np = np.array(screenshot)

        # Convert RGB to BGR (OpenCV uses BGR)
        screenshot_bgr = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)

        return screenshot_bgr

    def get_mouse_position(self):
        """Get current mouse position."""
        return pyautogui.position()

    def detect_elements(self, image):
        """Detect UI elements in the image and return their bounding boxes."""
        # Create a copy of the image
        original = image.copy()
        img_height, img_width = original.shape[:2]
        total_image_area = img_height * img_width
        max_element_area = total_image_area * self.max_element_area_ratio

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply multiple detection methods for better results

        # Method 1: Adaptive thresholding for text and UI elements
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        adaptive_thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Method 2: Edge detection for boundaries
        edges = cv2.Canny(blurred, 30, 150)
        kernel = np.ones((3, 3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)

        # Method 3: Color-based segmentation for UI elements with distinct colors
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Extract saturation channel (helps identify colored UI elements)
        saturation = hsv[:, :, 1]
        _, sat_threshold = cv2.threshold(saturation, 30, 255, cv2.THRESH_BINARY)

        # Find contours using all methods
        contours_thresh, _ = cv2.findContours(
            adaptive_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        contours_edges, _ = cv2.findContours(
            dilated_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        contours_color, _ = cv2.findContours(
            sat_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        # Combine all methods
        all_contours = contours_thresh + contours_edges + contours_color

        # Extract bounding rectangle information for all contours
        elements_info = []
        for i, cnt in enumerate(all_contours):
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h

            # Filter by size - reject too small or too large elements
            if area < self.min_element_area or area > max_element_area:
                continue

            # Add to elements list
            elements_info.append({
                'id': i,
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'center_x': x + w / 2,
                'center_y': y + h / 2,
                'area': area,
                'aspect_ratio': w / h if h > 0 else 0
            })

        # Remove duplicates based on overlap
        non_duplicate_elements = self.remove_duplicates(elements_info)

        # Also detect nested elements by finding contours with hierarchy
        contours_nested, hierarchy = cv2.findContours(
            adaptive_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        # Process hierarchy to find parent-child relationships
        if hierarchy is not None and len(hierarchy) > 0:
            hierarchy = hierarchy[0]  # Get the first hierarchy level
            for i, (cnt, h) in enumerate(zip(contours_nested, hierarchy)):
                # h[3] > -1 means this contour has a parent (it's nested)
                if h[3] > -1:
                    x, y, w, h = cv2.boundingRect(cnt)
                    area = w * h

                    # Filter by size
                    if area < self.min_element_area or area > max_element_area:
                        continue

                    # Add to elements list
                    elements_info.append({
                        'id': len(elements_info) + i,
                        'x': x,
                        'y': y,
                        'w': w,
                        'h': h,
                        'center_x': x + w / 2,
                        'center_y': y + h / 2,
                        'area': area,
                        'aspect_ratio': w / h if h > 0 else 0,
                        'nested': True
                    })

        # Re-run duplicate removal with the added nested elements
        non_duplicate_elements = self.remove_duplicates(elements_info)

        return non_duplicate_elements, original

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union of two bounding boxes."""
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

    def remove_duplicates(self, elements_info):
        """Remove duplicate elements based on high IOU."""
        non_duplicate_elements = []
        iou_threshold = 0.7

        for element in elements_info:
            is_duplicate = False
            for unique_element in non_duplicate_elements:
                if self.calculate_iou(element, unique_element) > iou_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                non_duplicate_elements.append(element)

        return non_duplicate_elements

    def find_element_at_mouse(self, elements, mouse_x, mouse_y):
        """Find the smallest UI element that contains the mouse cursor."""
        # First, filter all elements that contain the mouse cursor
        containing_elements = []
        for element in elements:
            x, y, w, h = element['x'], element['y'], element['w'], element['h']
            if (x <= mouse_x <= x + w) and (y <= mouse_y <= y + h):
                containing_elements.append(element)

        if not containing_elements:
            return None

        # Sort by area (ascending) to find the smallest element containing the mouse
        # This helps to find the most specific UI element (like a button)
        # rather than a larger container
        containing_elements.sort(key=lambda e: e['area'])

        # Return the smallest element that contains the mouse cursor
        # If we want to prioritize nested elements, we could also prioritize elements
        # that have the 'nested' flag
        for element in containing_elements:
            if element.get('nested', False):
                return element

        # Otherwise return the smallest element
        return containing_elements[0]

    def draw_mouse_and_element(self, image, mouse_x, mouse_y, element=None):
        """Draw mouse position and highlighting element if found."""
        # Make a copy of the image for drawing
        result_image = image.copy()

        # Draw a crosshair at mouse position
        cv2.drawMarker(
            result_image,
            (mouse_x, mouse_y),
            self.colors['mouse_box'],
            markerType=cv2.MARKER_CROSS,
            markerSize=20,
            thickness=2
        )

        # If element was found, highlight it
        if element:
            x, y, w, h = element['x'], element['y'], element['w'], element['h']
            cv2.rectangle(
                result_image,
                (x, y),
                (x + w, y + h),
                self.colors['element_border'],
                2
            )

            # Add element info on top of the box
            text = f"Element ID: {element['id']} (Area: {element['area']})"
            cv2.putText(
                result_image,
                text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.colors['element_border'],
                1
            )

        return result_image

    def extract_element_image(self, image, element):
        """Extract the image of just the element."""
        if not element:
            return None

        x, y, w, h = element['x'], element['y'], element['w'], element['h']
        return image[y:y + h, x:x + w]

    def visualize_all_elements(self, image, elements, mouse_x, mouse_y, highlighted_element=None):
        """Create a debug visualization showing all detected elements."""
        debug_image = image.copy()

        # Draw all elements with different colors based on size
        for element in elements:
            x, y, w, h = element['x'], element['y'], element['w'], element['h']

            # Choose color based on element size (green for small, blue for medium, red for large)
            color = (0, 255, 0)  # Default small (green)
            if element['area'] > 10000:
                color = (255, 0, 0)  # Large (red)
            elif element['area'] > 1000:
                color = (255, 165, 0)  # Medium (orange)

            # Highlight nested elements with a different color
            if element.get('nested', False):
                color = (255, 0, 255)  # Magenta for nested elements

            # Draw rectangle for this element
            cv2.rectangle(debug_image, (x, y), (x + w, y + h), color, 1)

            # Add element ID
            cv2.putText(
                debug_image,
                str(element['id']),
                (x + 5, y + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1
            )

        # Draw cross at mouse position
        cv2.drawMarker(
            debug_image,
            (mouse_x, mouse_y),
            (0, 0, 255),  # Red
            markerType=cv2.MARKER_CROSS,
            markerSize=20,
            thickness=2
        )

        # Highlight the selected element with a thicker border
        if highlighted_element:
            x = highlighted_element['x']
            y = highlighted_element['y']
            w = highlighted_element['w']
            h = highlighted_element['h']
            cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 255), 3)  # Yellow, thicker

        return debug_image

    def run(self):
        """Main function to run the detector."""
        try:
            print("Capturing screenshot...")
            screenshot = self.capture_screenshot()

            print("Getting mouse position...")
            mouse_x, mouse_y = self.get_mouse_position()

            print("Detecting UI elements...")
            elements, original = self.detect_elements(screenshot)
            print(f"Detected {len(elements)} UI elements")

            print("Finding smallest element at mouse position...")
            element_at_mouse = self.find_element_at_mouse(elements, mouse_x, mouse_y)

            # Create visualization with just the detected element
            result_image = self.draw_mouse_and_element(original, mouse_x, mouse_y, element_at_mouse)

            # Also create a debug visualization showing all detected elements
            debug_image = self.visualize_all_elements(original, elements, mouse_x, mouse_y, element_at_mouse)

            # Save both visualizations
            full_path = self._get_save_path("annotated_screenshot")
            cv2.imwrite(full_path, result_image)

            debug_path = self._get_save_path("debug_all_elements")
            cv2.imwrite(debug_path, debug_image)

            print(f"Annotated screenshot saved to: {full_path}")
            print(f"Debug visualization saved to: {debug_path}")

            # If an element was found, extract and save it
            if element_at_mouse:
                element_image = self.extract_element_image(original, element_at_mouse)
                element_path = self._get_save_path("mouse_element")
                cv2.imwrite(element_path, element_image)
                print(f"Element containing mouse saved to: {element_path}")
                print(f"Element details: ID={element_at_mouse['id']}, "
                      f"Position=({element_at_mouse['x']}, {element_at_mouse['y']}), "
                      f"Size={element_at_mouse['w']}x{element_at_mouse['h']}")
                if element_at_mouse.get('nested', False):
                    print("This is a nested UI element (likely a control within a container)")

                # Return only the specific element at mouse position
                return element_at_mouse
            else:
                print("No UI element found at mouse position")
                return None

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


def main():
    print("Initializing Mouse Element Detector...")

    # Check for required dependencies
    try:
        import cv2
        import numpy
        import pyautogui
        from sklearn.cluster import DBSCAN
    except ImportError as e:
        print(f"Error: Missing dependency: {e}")
        print("Please install required packages with: pip install opencv-python numpy pyautogui scikit-learn")
        sys.exit(1)

    detector = MouseElementDetector()
    print("Running detector...")
    element = detector.run()

    if element:
        print("Success! Found element at mouse position.")
    else:
        print("No element found at mouse position.")

    print("Analysis complete!")


if __name__ == "__main__":
    main()