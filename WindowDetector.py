import platform
import os
import sys
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import pyautogui

# Conditional imports based on platform
if platform.system() == 'Windows':
    import pygetwindow as gw
elif platform.system() == 'Darwin':  # macOS
    try:
        from AppKit import NSWorkspace, NSScreen
        import Quartz
    except ImportError:
        print("Error: macOS requires PyObjC to be installed.")
        print("Please install it with: pip install pyobjc")
        sys.exit(1)
elif platform.system() == 'Linux':
    try:
        import Xlib.display
        from ewmh import EWMH
    except ImportError:
        print("Error: Linux requires python-xlib and ewmh packages.")
        print("Please install them with: pip install python-xlib ewmh")
        sys.exit(1)
else:
    print(f"Unsupported platform: {platform.system()}")
    sys.exit(1)


class WindowDetector:
    def __init__(self):
        self.system = platform.system()
        print(f"Detected operating system: {self.system}")

        # Initialize font with platform-specific fallbacks
        self.font = self._initialize_font()

        self.colors = {
            'window_border': (255, 0, 0),  # Red
            'text': (255, 255, 255),  # White
            'text_background': (0, 0, 0, 128),  # Semi-transparent black
            'mouse_box': (255, 0, 0)  # Red for mouse bounding box
        }

        # Mouse box size (pixels around cursor)
        self.mouse_box_size = 30

        # Initialize platform-specific window managers
        if self.system == 'Linux':
            self.ewmh = EWMH()
            self.display = Xlib.display.Display()

    def _initialize_font(self):
        """Initialize font with platform-specific fallbacks"""
        font_options = []

        if self.system == 'Windows':
            font_options = ['arial.ttf', 'segoeui.ttf', 'calibri.ttf']
        elif self.system == 'Darwin':  # macOS
            font_options = [
                '/System/Library/Fonts/Helvetica.ttc',
                '/System/Library/Fonts/SFNSText.ttf',
                '/System/Library/Fonts/Geneva.ttf'
            ]
        elif self.system == 'Linux':
            font_options = [
                '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
                '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
                '/usr/share/fonts/TTF/Ubuntu-R.ttf'
            ]

        # Try each font option
        for font_path in font_options:
            try:
                return ImageFont.truetype(font_path, 14)
            except (IOError, OSError):
                continue

        # Fall back to default font
        print("Warning: Could not load system fonts, using default font")
        return ImageFont.load_default()

    def draw_label(self, draw, text, x, y, fill_color):
        """Draw text with a semi-transparent background."""
        # Get text size
        text_bbox = draw.textbbox((x, y), text, font=self.font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Draw semi-transparent background
        draw.rectangle(
            [(x, y), (x + text_width, y + text_height)],
            fill=self.colors['text_background']
        )

        # Draw text
        draw.text((x, y), text, fill=fill_color, font=self.font)

        return text_height

    def draw_mouse_box(self, draw):
        """Draw a bounding box around the current mouse position."""
        # Get current mouse position
        mouse_x, mouse_y = pyautogui.position()

        # Calculate box coordinates
        box_left = mouse_x - self.mouse_box_size
        box_top = mouse_y - self.mouse_box_size
        box_right = mouse_x + self.mouse_box_size
        box_bottom = mouse_y + self.mouse_box_size

        # Draw the box
        draw.rectangle(
            [(box_left, box_top), (box_right, box_bottom)],
            outline=self.colors['mouse_box'],
            width=2
        )

        # Add mouse position label
        self.draw_label(
            draw,
            f"Mouse: ({mouse_x}, {mouse_y})",
            box_left,
            box_top - 20,  # Position label above the box
            self.colors['mouse_box']
        )

        return mouse_x, mouse_y

    def is_point_in_window(self, x, y, window):
        """Check if a point is inside a window."""
        if self.system == 'Windows':
            return (window.left <= x <= window.left + window.width and
                    window.top <= y <= window.top + window.height)
        elif self.system == 'Darwin':  # macOS
            return (window['kCGWindowBounds']['X'] <= x <= window['kCGWindowBounds']['X'] + window['kCGWindowBounds'][
                'Width'] and
                    window['kCGWindowBounds']['Y'] <= y <= window['kCGWindowBounds']['Y'] + window['kCGWindowBounds'][
                        'Height'])
        elif self.system == 'Linux':
            return (window.get_geometry().x <= x <= window.get_geometry().x + window.get_geometry().width and
                    window.get_geometry().y <= y <= window.get_geometry().y + window.get_geometry().height)
        return False

    def get_windows(self):
        """Get all visible windows in a platform-independent way."""
        if self.system == 'Windows':
            return [w for w in gw.getAllWindows() if w.visible]
        elif self.system == 'Darwin':  # macOS
            return Quartz.CGWindowListCopyWindowInfo(
                Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
                Quartz.kCGNullWindowID
            )
        elif self.system == 'Linux':
            self.ewmh.display.flush()
            return self.ewmh.getClientList()
        return []

    def get_window_geometry(self, window):
        """Get window geometry in a platform-independent way."""
        if self.system == 'Windows':
            return {
                'left': window.left,
                'top': window.top,
                'width': window.width,
                'height': window.height,
                'title': window.title
            }
        elif self.system == 'Darwin':  # macOS
            bounds = window['kCGWindowBounds']
            return {
                'left': bounds['X'],
                'top': bounds['Y'],
                'width': bounds['Width'],
                'height': bounds['Height'],
                'title': window.get('kCGWindowName', 'Unknown')
            }
        elif self.system == 'Linux':
            geo = window.get_geometry()
            try:
                title = self.ewmh.getWmName(window).decode('utf-8', errors='replace')
            except:
                title = "Unknown"

            return {
                'left': geo.x,
                'top': geo.y,
                'width': geo.width,
                'height': geo.height,
                'title': title
            }
        return None

    def capture_and_analyze_windows(self):
        try:
            # Take a screenshot
            screenshot = pyautogui.screenshot()

            # Create a drawing object
            draw = ImageDraw.Draw(screenshot, 'RGBA')

            # Draw mouse box and get mouse position
            mouse_x, mouse_y = self.draw_mouse_box(draw)

            # Get all windows
            windows = self.get_windows()

            # Find and highlight the window containing the mouse cursor
            found_window = False
            for window in windows:
                if self.is_point_in_window(mouse_x, mouse_y, window):
                    found_window = True
                    geo = self.get_window_geometry(window)

                    # Draw window border
                    draw.rectangle(
                        [(geo['left'], geo['top']),
                         (geo['left'] + geo['width'], geo['top'] + geo['height'])],
                        outline=self.colors['window_border'],
                        width=2
                    )

                    # Draw window label
                    self.draw_label(
                        draw,
                        f"Window: {geo['title']}",
                        geo['left'] + 5,
                        geo['top'] + 5,
                        self.colors['window_border']
                    )
                    break

            if not found_window:
                print("No window found at mouse position.")

            # Save the annotated screenshot to an appropriate location
            save_path = self._get_save_path()
            screenshot.save(save_path)
            print(f"Annotated screenshot saved to: {save_path}")

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            import traceback
            traceback.print_exc()

    def _get_save_path(self):
        """Get an appropriate save path based on the platform."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'annotated_screenshot_{timestamp}.png'

        # Try to use the desktop
        try:
            if self.system == 'Windows':
                desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
            elif self.system == 'Darwin':  # macOS
                desktop = os.path.join(os.path.expanduser('~'), 'Desktop')
            elif self.system == 'Linux':
                desktop = os.path.join(os.path.expanduser('~'), 'Desktop')
                if not os.path.exists(desktop):
                    # Try XDG standard
                    desktop = os.path.join(os.path.expanduser('~'), 'Pictures')

            # Check if desktop exists and is writable
            if os.path.exists(desktop) and os.access(desktop, os.W_OK):
                return os.path.join(desktop, filename)
        except Exception as e:
            print(f"Could not determine desktop path: {e}")

        # Fall back to current directory if desktop is not available
        return os.path.join(os.getcwd(), filename)


def main():
    print("Initializing cross-platform window detector...")

    # Check for required dependencies
    try:
        import PIL
        import pyautogui
    except ImportError as e:
        print(f"Error: Missing dependency: {e}")
        print("Please install required packages with: pip install pillow pyautogui")
        if platform.system() == 'Windows':
            print("And: pip install pygetwindow")
        elif platform.system() == 'Darwin':  # macOS
            print("And: pip install pyobjc")
        elif platform.system() == 'Linux':
            print("And: pip install python-xlib ewmh")
        sys.exit(1)

    detector = WindowDetector()
    print("Capturing and analyzing windows... Please wait.")
    detector.capture_and_analyze_windows()
    print("Analysis complete!")


if __name__ == "__main__":
    main()