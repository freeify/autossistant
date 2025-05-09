#!/usr/bin/env python3
"""
Precise Window Capture Tool
---------------------------
Captures a precise screenshot of the window under the mouse cursor
with advanced edge detection and high-quality output.
"""

import os
import sys
import time
import platform
from datetime import datetime
import tempfile
import subprocess
from pathlib import Path
import traceback

# Try to import required libraries
try:
    from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageFont
    import pyautogui
except ImportError as e:
    print(f"Error: Required dependencies not found: {e}")
    print("Please install required packages with: pip install pillow pyautogui")
    sys.exit(1)

# Platform-specific imports
if platform.system() == 'Windows':
    try:
        import pygetwindow as gw
        import ctypes
        from ctypes import wintypes
    except ImportError:
        print("Error: Windows requires pygetwindow.")
        print("Please install with: pip install pygetwindow")
        sys.exit(1)
elif platform.system() == 'Darwin':  # macOS
    try:
        import Quartz
    except ImportError:
        print("Error: macOS requires PyObjC.")
        print("Please install with: pip install pyobjc")
        sys.exit(1)
elif platform.system() == 'Linux':
    try:
        import Xlib.display
        from ewmh import EWMH
    except ImportError:
        print("Error: Linux requires python-xlib and ewmh.")
        print("Please install with: pip install python-xlib ewmh")
        sys.exit(1)


class PreciseWindowCapture:
    """Captures precise window screenshots across multiple platforms"""

    def __init__(self):
        """Initialize the window capture tool"""
        self.system = platform.system()
        self.output_dir = self._create_output_directory()

        # Configure capture settings
        self.settings = {
            'dpi': (600, 600),  # High DPI for quality
            'quality': 100,  # Maximum quality
            'format': 'PNG',  # Lossless format
            'mouse_box_size': 30  # Size of mouse highlight box
        }

        # Configure visual elements
        self.colors = {
            'border': (255, 0, 0),  # Red for borders
            'text': (255, 255, 255),  # White for text
            'text_bg': (0, 0, 0, 180),  # Semi-transparent black for text background
            'mouse': (255, 0, 0)  # Red for mouse indicator
        }

        # Initialize platform-specific components
        self._init_platform_components()

        # Initialize font
        self.font = self._init_font()

    def _init_platform_components(self):
        """Initialize platform-specific components"""
        if self.system == 'Linux':
            self.ewmh = EWMH()
            self.display = Xlib.display.Display()

    def _init_font(self):
        """Initialize an appropriate font for the current platform"""
        # Define platform-specific font paths
        font_paths = {
            'Windows': ['arial.ttf', 'segoeui.ttf', 'calibri.ttf'],
            'Darwin': [
                '/System/Library/Fonts/Helvetica.ttc',
                '/System/Library/Fonts/SFNSText.ttf',
                '/System/Library/Fonts/Geneva.ttf'
            ],
            'Linux': [
                '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
                '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
                '/usr/share/fonts/TTF/Ubuntu-R.ttf'
            ]
        }

        # Try each font for the current platform
        for font_path in font_paths.get(self.system, []):
            try:
                return ImageFont.truetype(font_path, 14)
            except (IOError, OSError):
                continue

        # Fall back to default font
        print("Warning: Using default font (system fonts not found)")
        return ImageFont.load_default()

    def _create_output_directory(self):
        """Create directory for output files"""
        try:
            # Determine desktop path based on platform
            if self.system == 'Windows':
                desktop = os.path.join(os.environ['USERPROFILE'], 'Desktop')
            elif self.system == 'Darwin':  # macOS
                desktop = os.path.join(os.path.expanduser('~'), 'Desktop')
            elif self.system == 'Linux':
                desktop = os.path.join(os.path.expanduser('~'), 'Desktop')
                if not os.path.exists(desktop):
                    desktop = os.path.join(os.path.expanduser('~'), 'Pictures')
            else:
                desktop = os.getcwd()

            # Create directory with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            output_dir = os.path.join(desktop, f'Capture_{timestamp}')
            os.makedirs(output_dir, exist_ok=True)

            print(f"Created output directory: {output_dir}")
            return output_dir

        except Exception as e:
            print(f"Failed to create output directory: {e}")
            # Fallback to current directory
            fallback = os.path.join(os.getcwd(), f'Capture_{datetime.now().strftime("%Y%m%d_%H%M")}')
            os.makedirs(fallback, exist_ok=True)
            return fallback

    def take_screenshot(self):
        """Capture high-quality screenshot using the best available method"""
        # First try platform-specific methods for better quality
        if self.system == 'Windows':
            try:
                # Use PowerShell for highest quality on Windows
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp:
                    temp_path = temp.name

                ps_command = f"""
                Add-Type -AssemblyName System.Windows.Forms
                Add-Type -AssemblyName System.Drawing
                $screen = [System.Windows.Forms.Screen]::PrimaryScreen
                $bounds = $screen.Bounds
                $bitmap = New-Object System.Drawing.Bitmap $bounds.Width, $bounds.Height, [System.Drawing.Imaging.PixelFormat]::Format32bppArgb
                $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
                $graphics.CompositingQuality = [System.Drawing.Drawing2D.CompositingQuality]::HighQuality
                $graphics.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
                $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::HighQuality
                $graphics.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality
                $graphics.CopyFromScreen($bounds.X, $bounds.Y, 0, 0, $bounds.Size)
                $bitmap.Save('{temp_path}', [System.Drawing.Imaging.ImageFormat]::Png)
                $graphics.Dispose()
                $bitmap.Dispose()
                """

                subprocess.run(['powershell', '-Command', ps_command],
                               check=True, capture_output=True)
                screenshot = Image.open(temp_path)
                os.unlink(temp_path)
                return screenshot
            except Exception as e:
                print(f"Windows screenshot method failed: {e}, falling back to PyAutoGUI")

        elif self.system == 'Darwin':  # macOS
            try:
                # Use native screencapture for macOS
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp:
                    temp_path = temp.name

                # Get the best quality with native macOS tool
                subprocess.run(['screencapture', '-x', temp_path], check=True)
                screenshot = Image.open(temp_path)
                os.unlink(temp_path)
                return screenshot
            except Exception as e:
                print(f"macOS screenshot method failed: {e}, falling back to PyAutoGUI")

        # Fall back to PyAutoGUI for any failures or other platforms
        screenshot = pyautogui.screenshot()
        screenshot.info['dpi'] = self.settings['dpi']
        return screenshot

    def get_windows(self):
        """Get all visible windows for the current platform"""
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

    def is_point_in_window(self, x, y, window):
        """Check if a point (x,y) is inside a window"""
        if self.system == 'Windows':
            return (window.left <= x <= window.left + window.width and
                    window.top <= y <= window.top + window.height)
        elif self.system == 'Darwin':  # macOS
            bounds = window['kCGWindowBounds']
            return (bounds['X'] <= x <= bounds['X'] + bounds['Width'] and
                    bounds['Y'] <= y <= bounds['Y'] + bounds['Height'])
        elif self.system == 'Linux':
            geo = window.get_geometry()
            return (geo.x <= x <= geo.x + geo.width and
                    geo.y <= y <= geo.y + geo.height)
        return False

    def get_basic_window_geometry(self, window):
        """Get basic window geometry information"""
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

    def get_precise_window_bounds(self, window, basic_geo):
        """Get precise window boundaries excluding window decorations"""
        # Start with the basic geometry
        geo = basic_geo.copy()

        # Try platform-specific precise methods
        if self.system == 'Windows':
            try:
                # Define required Win32 API structures
                class RECT(ctypes.Structure):
                    _fields_ = [("left", ctypes.c_long),
                                ("top", ctypes.c_long),
                                ("right", ctypes.c_long),
                                ("bottom", ctypes.c_long)]

                # Get window handle
                hwnd = window._hWnd

                # Try to get client area (content without borders)
                rect = RECT()

                # First try DwmGetWindowAttribute for accurate bounds with shadows
                try:
                    DWMWA_EXTENDED_FRAME_BOUNDS = 9
                    ctypes.windll.dwmapi.DwmGetWindowAttribute(
                        hwnd,
                        DWMWA_EXTENDED_FRAME_BOUNDS,
                        ctypes.byref(rect),
                        ctypes.sizeof(rect)
                    )

                    return {
                        'left': rect.left,
                        'top': rect.top,
                        'width': rect.right - rect.left,
                        'height': rect.bottom - rect.top,
                        'title': geo['title']
                    }
                except:
                    # If DWM fails, try GetClientRect for client area
                    ctypes.windll.user32.GetClientRect(hwnd, ctypes.byref(rect))
                    # Convert client coordinates to screen coordinates
                    point = wintypes.POINT(rect.left, rect.top)
                    ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(point))
                    left, top = point.x, point.y
                    point = wintypes.POINT(rect.right, rect.bottom)
                    ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(point))
                    right, bottom = point.x, point.y

                    return {
                        'left': left,
                        'top': top,
                        'width': right - left,
                        'height': bottom - top,
                        'title': geo['title']
                    }
            except Exception as e:
                print(f"Precise Windows bounds detection failed: {e}")

        # For macOS, we could implement similar precise boundary detection
        # For now, return the basic geometry
        return geo

    def detect_edges(self, image, geo):
        """Use image processing to find precise window edges"""
        try:
            # Add a small margin to detect edges just outside the window
            margin = 15
            left = max(0, geo['left'] - margin)
            top = max(0, geo['top'] - margin)
            right = min(image.width, geo['left'] + geo['width'] + margin)
            bottom = min(image.height, geo['top'] + geo['height'] + margin)

            # Extract the region with margin
            region = image.crop((left, top, right, bottom))

            # Convert to grayscale for edge detection
            gray = region.convert('L')

            # Apply edge detection and threshold
            edges = gray.filter(ImageFilter.FIND_EDGES)
            edges = edges.point(lambda p: p > 128 and 255)

            # Scan from four sides to find edges
            width, height = edges.size
            edge_data = list(edges.getdata())

            # Initialize with conservative values
            edge_left = margin
            edge_top = margin
            edge_right = width - margin
            edge_bottom = height - margin

            # Scan from left
            for x in range(margin, width // 3):
                for y in range(margin, height - margin):
                    if edge_data[y * width + x] > 200:
                        edge_left = x
                        break
                if edge_left != margin:
                    break

            # Scan from top
            for y in range(margin, height // 3):
                for x in range(margin, width - margin):
                    if edge_data[y * width + x] > 200:
                        edge_top = y
                        break
                if edge_top != margin:
                    break

            # Scan from right
            for x in range(width - margin - 1, 2 * width // 3, -1):
                for y in range(margin, height - margin):
                    if edge_data[y * width + x] > 200:
                        edge_right = x
                        break
                if edge_right != width - margin:
                    break

            # Scan from bottom
            for y in range(height - margin - 1, 2 * height // 3, -1):
                for x in range(margin, width - margin):
                    if edge_data[y * width + x] > 200:
                        edge_bottom = y
                        break
                if edge_bottom != height - margin:
                    break

            # Convert back to screen coordinates
            precise_geo = {
                'left': left + edge_left,
                'top': top + edge_top,
                'width': edge_right - edge_left,
                'height': edge_bottom - edge_top,
                'title': geo['title']
            }

            return precise_geo
        except Exception as e:
            print(f"Edge detection error: {e}")
            return geo

    def draw_label(self, draw, text, x, y, color):
        """Draw text with a semi-transparent background"""
        # Measure text size
        text_bbox = draw.textbbox((x, y), text, font=self.font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Draw background
        draw.rectangle(
            [(x, y), (x + text_width, y + text_height)],
            fill=self.colors['text_bg']
        )

        # Draw text
        draw.text((x, y), text, fill=color, font=self.font)

        return text_height

    def draw_mouse_indicator(self, draw):
        """Draw a box around the current mouse position"""
        # Get mouse position
        mouse_x, mouse_y = pyautogui.position()

        # Calculate box
        box_size = self.settings['mouse_box_size']
        box_left = mouse_x - box_size
        box_top = mouse_y - box_size
        box_right = mouse_x + box_size
        box_bottom = mouse_y + box_size

        # Draw the box
        draw.rectangle(
            [(box_left, box_top), (box_right, box_bottom)],
            outline=self.colors['mouse'],
            width=2
        )

        # Add label
        self.draw_label(
            draw,
            f"Mouse: ({mouse_x}, {mouse_y})",
            box_left,
            box_top - 20,
            self.colors['mouse']
        )

        return mouse_x, mouse_y

    def enhance_image(self, img):
        """Apply subtle enhancements to the image for better quality"""
        try:
            # Sharpen slightly
            img = ImageEnhance.Sharpness(img).enhance(1.2)

            # Enhance contrast slightly
            img = ImageEnhance.Contrast(img).enhance(1.1)

            return img
        except Exception as e:
            print(f"Image enhancement failed: {e}")
            return img

    def save_image(self, img, filename):
        """Save image with optimal quality settings"""
        path = os.path.join(self.output_dir, filename)
        img.save(
            path,
            format=self.settings['format'],
            dpi=self.settings['dpi'],
            quality=self.settings['quality'],
            optimize=True
        )
        return path

    def capture(self):
        """Capture the window under the mouse cursor with high precision"""
        try:
            # Take high-quality screenshot
            print("Taking screenshot...")
            screenshot = self.take_screenshot()

            # Create a copy for annotations
            annotated = screenshot.copy()
            annotated_draw = ImageDraw.Draw(annotated, 'RGBA')

            # Get current timestamp for filenames
            timestamp = datetime.now().strftime('%H%M%S')

            # Draw mouse indicator and get position
            mouse_x, mouse_y = self.draw_mouse_indicator(annotated_draw)

            # Get all windows
            windows = self.get_windows()

            # Find window under cursor
            for window in windows:
                if self.is_point_in_window(mouse_x, mouse_y, window):
                    # Get geometry at three levels of precision
                    basic_geo = self.get_basic_window_geometry(window)
                    precise_geo = self.get_precise_window_bounds(window, basic_geo)
                    edge_geo = self.detect_edges(screenshot, precise_geo)

                    # Log the detected boundaries
                    print("\nPrecise window detection results:")
                    print(f"  Basic bounds:   L={basic_geo['left']}, T={basic_geo['top']}, "
                          f"W={basic_geo['width']}, H={basic_geo['height']}")
                    print(f"  Precise bounds: L={precise_geo['left']}, T={precise_geo['top']}, "
                          f"W={precise_geo['width']}, H={precise_geo['height']}")
                    print(f"  Edge detection: L={edge_geo['left']}, T={edge_geo['top']}, "
                          f"W={edge_geo['width']}, H={edge_geo['height']}")
                    print(f"  Window title: {edge_geo['title']}")

                    # Draw window border on annotated screenshot
                    annotated_draw.rectangle(
                        [(edge_geo['left'], edge_geo['top']),
                         (edge_geo['left'] + edge_geo['width'], edge_geo['top'] + edge_geo['height'])],
                        outline=self.colors['border'],
                        width=2
                    )

                    # Add window label
                    self.draw_label(
                        annotated_draw,
                        f"Window: {edge_geo['title']}",
                        edge_geo['left'] + 5,
                        edge_geo['top'] + 5,
                        self.colors['border']
                    )

                    # Extract the window with precise bounds
                    window_img = screenshot.crop((
                        edge_geo['left'],
                        edge_geo['top'],
                        edge_geo['left'] + edge_geo['width'],
                        edge_geo['top'] + edge_geo['height']
                    ))

                    # Apply subtle enhancements
                    window_img = self.enhance_image(window_img)

                    # Create bordered version
                    window_bordered = window_img.copy()
                    bordered_draw = ImageDraw.Draw(window_bordered)
                    bordered_draw.rectangle(
                        [(0, 0), (window_img.width - 1, window_img.height - 1)],
                        outline=self.colors['border'],
                        width=2
                    )

                    # Save all images with simplified naming
                    full_path = self.save_image(screenshot, f'full_{timestamp}.png')
                    window_path = self.save_image(window_img, f'window_{timestamp}.png')
                    bordered_path = self.save_image(window_bordered, f'window_border_{timestamp}.png')
                    annotated_path = self.save_image(annotated, f'annotated_{timestamp}.png')

                    print("\nSaved images:")
                    print(f"  Full screenshot:   {full_path}")
                    print(f"  Window capture:    {window_path}")
                    print(f"  Bordered window:   {bordered_path}")
                    print(f"  Annotated reference: {annotated_path}")

                    return True

            # If no window found
            print("No window found under mouse cursor!")
            # Still save the full and annotated screenshots
            full_path = self.save_image(screenshot, f'full_{timestamp}.png')
            annotated_path = self.save_image(annotated, f'annotated_{timestamp}.png')

            print("\nSaved images:")
            print(f"  Full screenshot:     {full_path}")
            print(f"  Annotated reference: {annotated_path}")

            return False

        except Exception as e:
            print(f"Capture failed: {e}")
            traceback.print_exc()
            return False


def main():
    """Main function to run the window capture tool"""
    print("=" * 60)
    print(f"Precise Window Capture Tool - {platform.system()}")
    print("=" * 60)

    try:
        capture_tool = PreciseWindowCapture()

        # Countdown before capture
        print("\nPosition your mouse over the window to capture...")
        for i in range(5, 0, -1):
            print(f"Capturing in {i} seconds...")
            time.sleep(1)

        print("\nCapturing window with precise edge detection...")
        result = capture_tool.capture()

        if result:
            print("\nCapture completed successfully!")
        else:
            print("\nCapture completed, but no window was detected under the cursor.")

        print("\nDone.")

    except KeyboardInterrupt:
        print("\nCapture cancelled by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()