import socket
import win32gui
import win32api
import win32con
import win32ui
import time
import threading
import pyperclip
from pynput import mouse, keyboard


class InputMonitor:
    def __init__(self):
        self.last_icon_shown = 0
        self.icon_cooldown = 0.2  # Reduced cooldown for more responsive UI
        self.mouse_listener = None
        self.keyboard_listener = None
        self.last_click_pos = None
        self.is_editing = False
        self.socket = None
        self.cursor_tracker = None
        self.cursor_track_active = False
        self.last_cursor_update = 0
        self.cursor_update_interval = 0.05  # 50ms updates for smooth tracking
        self.textbox_visible = False
        self.textbox_hwnd = None
        self.textbox_bounds = None

        # Add I-beam cursor detection
        self.last_cursor_type = None
        self.is_ibeam_cursor = False

        # Try to establish a persistent connection
        self.connect_socket()

        # Start cursor position tracking thread
        self.start_cursor_tracker()

    def connect_socket(self):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect(('localhost', 65432))
            print("Connected to Electron app")
        except Exception as e:
            print(f"Error connecting to Electron app: {e}")
            self.socket = None

    def send_message(self, message):
        try:
            if self.socket is None:
                self.connect_socket()

            if self.socket:
                self.socket.sendall(message.encode())
            else:
                print("Socket not available - creating new connection")
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.connect(('localhost', 65432))
                    s.sendall(message.encode())
        except Exception as e:
            print(f"Error sending message: {e}")
            # Socket might have been closed, try reconnecting next time
            self.socket = None

    def check_cursor_type(self):
        """Check if the current cursor is an I-beam cursor"""
        try:
            # Get current cursor handle
            cursor_info = win32gui.GetCursorInfo()
            cursor_handle = cursor_info[1]  # The handle is the second item in the tuple

            # I-beam cursor has predefined ID in Windows
            is_ibeam = cursor_handle == win32con.IDC_IBEAM

            # Common I-beam cursor handles (some applications use custom cursors)
            # Checking against common cursor resources
            common_ibeam_handles = [
                win32con.IDC_IBEAM,  # Standard I-beam
                65541,  # Common custom I-beam value
                65545,  # Another common I-beam variant
            ]

            # Check for predefined I-beam or common custom I-beams
            is_ibeam = cursor_handle in common_ibeam_handles

            # Update state if cursor type changed
            if is_ibeam != self.is_ibeam_cursor:
                self.is_ibeam_cursor = is_ibeam
                print(f"Cursor is {'I-beam' if is_ibeam else 'not I-beam'}, handle: {cursor_handle}")

                # Send cursor type information to the app
                self.send_message(f"cursor_type|{'ibeam' if is_ibeam else 'default'}\n")

            return is_ibeam
        except Exception as e:
            print(f"Error checking cursor type: {e}")
            return False

    def start_cursor_tracker(self):
        """Start a thread to continuously track cursor position and type"""

        def track_cursor():
            while True:
                if self.cursor_track_active:
                    current_time = time.time()
                    if current_time - self.last_cursor_update >= self.cursor_update_interval:
                        try:
                            cursor_x, cursor_y = win32gui.GetCursorPos()

                            # Check cursor type (I-beam or not)
                            self.check_cursor_type()

                            # Send cursor position to app
                            self.send_message(f"cursor_position|{cursor_x},{cursor_y}\n")
                            self.last_cursor_update = current_time

                            # If textbox is visible, check if cursor is inside the textbox bounds
                            if self.textbox_hwnd and self.textbox_bounds:
                                if self.is_point_in_rect(cursor_x, cursor_y, self.textbox_bounds):
                                    # Cursor is inside textbox, don't close it
                                    self.cursor_track_active = True
                                    self.textbox_visible = True
                        except Exception as e:
                            print(f"Error tracking cursor: {e}")
                time.sleep(0.01)  # Short sleep to prevent CPU overuse

        self.cursor_tracker = threading.Thread(target=track_cursor, daemon=True)
        self.cursor_tracker.start()
        print("Cursor tracker started")

    def is_point_in_rect(self, x, y, rect):
        """Check if a point is inside a rectangle (left, top, right, bottom)"""
        left, top, right, bottom = rect
        return left <= x <= right and top <= y <= bottom

    def find_electron_textbox_window(self):
        """Find and store the Electron textbox window handle and bounds"""

        def enum_windows_callback(hwnd, extra):
            if win32gui.IsWindowVisible(hwnd):
                window_text = win32gui.GetWindowText(hwnd)
                class_name = win32gui.GetClassName(hwnd)

                # Look for Electron window with textbox
                if "Electron" in class_name or "Chrome_WidgetWin" in class_name:
                    try:
                        rect = win32gui.GetWindowRect(hwnd)
                        # Store potential textbox window
                        self.textbox_hwnd = hwnd
                        self.textbox_bounds = rect
                        return False  # Stop enumeration once found
                    except Exception as e:
                        print(f"Error getting window rect: {e}")
            return True  # Continue enumeration

        try:
            win32gui.EnumWindows(enum_windows_callback, None)
        except Exception as e:
            print(f"Error enumerating windows: {e}")

    def is_electron_textbox(self, hwnd):
        """Check if a window handle belongs to our Electron textbox"""
        try:
            # Additional checks specific to our application
            class_name = win32gui.GetClassName(hwnd).lower()
            title = win32gui.GetWindowText(hwnd)

            # Try to match our Electron window
            return ("electron" in class_name.lower() or
                    "chrome_widgetwin" in class_name.lower()) and not title
        except Exception:
            return False

    # Add these improvements to your InputMonitor class

    def update_textbox_state(self, visible):
        """Update the textbox visibility state and find its bounds if visible"""
        self.textbox_visible = visible
        if visible:
            # Try to find the textbox window to get its bounds
            self.find_electron_textbox_window()
            # Deactivate plus icon when textbox is visible
            self.send_message(f"hide_plus_icon\n")
        else:
            # Reset textbox information when hidden
            self.textbox_hwnd = None
            self.textbox_bounds = None
            # Always show plus icon when textbox is hidden
            self.send_message(f"show_plus_icon\n")

    def on_mouse_move(self, x, y):
        """Track mouse movement to detect hover state for textbox"""
        try:
            # Get the actual screen coordinates
            cursor_x, cursor_y = win32gui.GetCursorPos()

            # Only send position updates when the textbox is not visible
            if not self.textbox_visible:
                self.send_message(f"cursor_position|{cursor_x},{cursor_y}\n")

                # If we have a plus icon identified, check if mouse is over it
                if self.plus_icon_bounds:
                    if self.is_point_in_rect(cursor_x, cursor_y, self.plus_icon_bounds):
                        # Mouse is over plus icon, show textbox
                        self.send_message(f"show_textbox|{cursor_x},{cursor_y}\n")
                        self.textbox_visible = True
        except Exception as e:
            print(f"Error tracking mouse movement: {e}")

    def find_electron_textbox_window(self):
        """Find and store the Electron textbox window handle and bounds"""

        def enum_windows_callback(hwnd, extra):
            if win32gui.IsWindowVisible(hwnd):
                window_text = win32gui.GetWindowText(hwnd)
                class_name = win32gui.GetClassName(hwnd)

                # Look for Electron window with textbox
                if "Electron" in class_name or "Chrome_WidgetWin" in class_name:
                    try:
                        rect = win32gui.GetWindowRect(hwnd)
                        # Check if window size matches our expected textbox
                        width = rect[2] - rect[0]
                        height = rect[3] - rect[1]

                        # Store potential textbox window if size is reasonable
                        if 300 <= width <= 350 and 70 <= height <= 100:
                            self.textbox_hwnd = hwnd
                            self.textbox_bounds = rect
                            print(f"Found textbox window: {hwnd}, bounds: {rect}")
                            return False  # Stop enumeration once found
                    except Exception as e:
                        print(f"Error getting window rect: {e}")
            return True  # Continue enumeration

        try:
            win32gui.EnumWindows(enum_windows_callback, None)
        except Exception as e:
            print(f"Error enumerating windows: {e}")

    def find_plus_icon_window(self):
        """Find and store the Electron plus icon window handle and bounds"""

        def enum_windows_callback(hwnd, extra):
            if win32gui.IsWindowVisible(hwnd):
                class_name = win32gui.GetClassName(hwnd)

                # Look for Electron window with plus icon (very small window)
                if "Electron" in class_name or "Chrome_WidgetWin" in class_name:
                    try:
                        rect = win32gui.GetWindowRect(hwnd)
                        # Check if window size matches our expected plus icon
                        width = rect[2] - rect[0]
                        height = rect[3] - rect[1]

                        # Plus icon should be very small
                        if 20 <= width <= 40 and 20 <= height <= 40:
                            self.plus_icon_hwnd = hwnd
                            self.plus_icon_bounds = rect
                            print(f"Found plus icon window: {hwnd}, bounds: {rect}")
                            return False  # Stop enumeration once found
                    except Exception as e:
                        print(f"Error getting window rect: {e}")
            return True  # Continue enumeration

        try:
            win32gui.EnumWindows(enum_windows_callback, None)
        except Exception as e:
            print(f"Error enumerating windows: {e}")

    # Add a mouse move listener to your start method
    def start(self):
        # Start mouse listener for clicks
        self.mouse_listener = mouse.Listener(on_click=self.on_click)
        self.mouse_listener.start()
        print("Mouse listener started")

        # Add mouse move listener
        self.mouse_move_listener = mouse.Listener(on_move=self.on_mouse_move)
        self.mouse_move_listener.start()
        print("Mouse move listener started")

        # Start keyboard listener
        self.keyboard_listener = keyboard.Listener(on_press=self.on_key_press)
        self.keyboard_listener.start()
        print("Keyboard listener started")

        # Start clipboard monitor
        clipboard_thread = threading.Thread(target=self.check_clipboard, daemon=True)
        clipboard_thread.start()
        print("Clipboard monitor started")

        # Start listening for app messages
        self.listen_for_app_messages()

        # Activate cursor tracking
        self.cursor_track_active = True

        # Find the plus icon window
        self.find_plus_icon_window()

        # Keep the script running
        try:
            print("Input monitor running. Press Ctrl+C to stop.")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping monitors...")
            self.cursor_track_active = False
            self.mouse_listener.stop()
            self.mouse_move_listener.stop()
            self.keyboard_listener.stop()
            if self.socket:
                self.socket.close()

    def on_click(self, x, y, button, pressed):
        if pressed:
            # Get the actual screen coordinates
            cursor_x, cursor_y = win32gui.GetCursorPos()
            print(f"Mouse click detected at screen coordinates: {cursor_x}, {cursor_y}")

            # Check if click is inside textbox - if so, don't trigger textbox appearance
            if self.textbox_visible and self.textbox_bounds:
                if self.is_point_in_rect(cursor_x, cursor_y, self.textbox_bounds):
                    print("Click inside textbox - ignoring")
                    return

            # Check if this is an editable field
            hwnd = win32gui.WindowFromPoint((cursor_x, cursor_y))

            # Ignore clicks in our textbox window
            if self.is_electron_textbox(hwnd):
                print("Click in our textbox - ignoring")
                return

            # Check if we have an I-beam cursor at click time
            is_ibeam = self.check_cursor_type()
            print(f"Is I-beam cursor: {is_ibeam}")

            # First check for text field in current window
            is_text_field = self.can_receive_text_input(hwnd)

            # If not found in current window, check parent (for compound controls)
            if not is_text_field:
                parent_hwnd = win32gui.GetParent(hwnd)
                if parent_hwnd:
                    is_text_field = self.can_receive_text_input(parent_hwnd)

            # Then check grand-parent (for deeply nested controls)
            if not is_text_field:
                grandparent_hwnd = win32gui.GetParent(win32gui.GetParent(hwnd)) if win32gui.GetParent(hwnd) else None
                if grandparent_hwnd:
                    is_text_field = self.can_receive_text_input(grandparent_hwnd)

            print(f"Is editable text field: {is_text_field}")

            # Send the click position to the app
            self.send_message(f"mouse_click|{cursor_x},{cursor_y}\n")

            # Show the popup for editable fields or when I-beam cursor is active
            if is_text_field or is_ibeam:
                self.is_editing = True
                self.cursor_track_active = True  # Activate continuous cursor tracking
                self.send_message(f"show_popup|{cursor_x},{cursor_y}\n")
                print("Detected editable text field, showing popup")
            else:
                self.send_message(f"hide_popup\n")
                self.is_editing = False
                print("Not an editable field, hiding popup")

            # Store for potential text editing context
            self.last_click_pos = (cursor_x, cursor_y)
            self.last_icon_shown = time.time()

            # STRICT CHECK: Only show popup for confirmed editable text fields
            # Ignore the general text content check that was causing false positives
            if is_text_field:
                self.is_editing = True
                self.cursor_track_active = True  # Activate continuous cursor tracking
                # Send explicit command to show the popup
                self.send_message(f"show_popup|{cursor_x},{cursor_y}\n")
                print("Detected click on EDITABLE text field, showing popup")
            else:
                # Send explicit message NOT to show popup
                self.send_message(f"hide_popup\n")
                self.is_editing = False
                print("Not clicking on editable text field - no popup")

            # Store for potential text editing context
            self.last_click_pos = (cursor_x, cursor_y)
            self.last_icon_shown = time.time()

    def on_key_press(self, key):
        try:
            if self.is_editing:
                # When typing in an editing context, make sure cursor tracking is active
                self.cursor_track_active = True

                # Also send current cursor position
                cursor_x, cursor_y = win32gui.GetCursorPos()
                self.send_message(f"cursor_position|{cursor_x},{cursor_y}\n")
        except Exception as e:
            print(f"Error in key press handler: {e}")

    def can_receive_text_input(self, hwnd):
        """Check if a window can receive text input (is an editable field)"""
        try:
            # Skip our own textbox
            if self.is_electron_textbox(hwnd):
                return False

            # Get various window properties
            style = win32gui.GetWindowLong(hwnd, win32con.GWL_STYLE)

            # List of common text input class identifiers
            text_classes = [
                'edit',
                'text',
                'richedit',
                'textbox',
                'textarea',
                'input'
            ]

            # Get window class name
            class_name = win32gui.GetClassName(hwnd).lower()

            # Check if it's a known text input class
            is_text_class = any(text_class in class_name.lower() for text_class in text_classes)

            # Get editable styles
            edit_styles = [
                win32con.ES_MULTILINE,
                win32con.ES_AUTOVSCROLL,
                win32con.ES_AUTOHSCROLL,
                win32con.WS_TABSTOP  # Many editable fields are tabbable
            ]

            # Check if it has editable styles
            has_edit_style = any(style & edit_style for edit_style in edit_styles)

            # Handle special cases for web browsers and other complex UIs
            is_browser = False
            browser_classes = ["chrome", "mozilla", "firefox", "msedge", "opera"]

            if any(browser in class_name.lower() for browser in browser_classes):
                is_browser = True
                # For browsers, use I-beam cursor as a strong signal
                if self.is_ibeam_cursor:
                    return True

            # For HTML elements in browsers
            html_classes = ["internetexplorer_server", "chrome_widgetwin", "chromium"]
            is_html = any(html_class in class_name.lower() for html_class in html_classes)

            # The most reliable indicator for text fields - I-beam cursor when clicked
            if self.is_ibeam_cursor:
                return True

            # Return true if any of our checks succeed
            return is_text_class or has_edit_style or is_html

        except Exception as e:
            print(f"Error checking text input capability: {e}")
            return False

        except Exception as e:
            print(f"Error checking text input capability: {e}")
            return False

    def is_clicking_on_text(self, hwnd, x, y):
        """Determine if the user is clicking on text content (not necessarily editable)"""
        try:
            # First check if the window or any of its parents has text content
            current_hwnd = hwnd
            window_text = ""

            # Check if this window or any parent has text content
            while current_hwnd and not window_text:
                window_text = win32gui.GetWindowText(current_hwnd)
                if not window_text:
                    current_hwnd = win32gui.GetParent(current_hwnd)

            if window_text:
                # We found some text, now check more specifically if we're clicking on a text area

                # 1. Check if cursor is I-beam (strong indicator we're over text)
                if self.is_ibeam_cursor:
                    return True

                # 2. Check if it's a common browser or document viewer application
                class_name = win32gui.GetClassName(hwnd).lower()
                parent = win32gui.GetParent(hwnd)
                parent_class = win32gui.GetClassName(parent).lower() if parent else ""

                text_container_classes = [
                    'chrome',
                    'firefox',
                    'edge',
                    'safari',
                    'opera',
                    'mozilla',
                    'document',
                    'reader',
                    'pdf',
                    'word',
                    'office'
                ]

                if any(tc in class_name.lower() or (parent_class and tc in parent_class.lower())
                       for tc in text_container_classes):
                    # In a browser or document viewer, look for text selection capability
                    if self.is_ibeam_cursor:
                        return True

            # For browsers and other complex UIs, try to detect based on cursor shape
            # If the cursor is an I-beam when clicking, it's likely text content
            return self.is_ibeam_cursor

        except Exception as e:
            print(f"Error checking if clicking on text: {e}")
            return False

    def check_clipboard(self):
        last_text = pyperclip.paste()
        while True:
            try:
                current_text = pyperclip.paste()
                if current_text != last_text and self.is_editing:
                    print(f"Clipboard content changed, sending selection")
                    self.send_message(f"selection_made|{current_text}\n")
                    last_text = current_text
            except Exception as e:
                print(f"Error checking clipboard: {e}")
            time.sleep(0.1)

    # Listen for messages from Electron app
    def listen_for_app_messages(self):
        def listen_thread():
            while True:
                try:
                    if self.socket:
                        data = self.socket.recv(1024)
                        if data:
                            message = data.decode().strip()
                            if message == "textbox-shown":
                                print("Textbox is now visible")
                                self.update_textbox_state(True)
                            elif message == "textbox-hidden":  # Fixed missing handler
                                print("Textbox is now hidden")
                                self.update_textbox_state(False)
                except Exception as e:
                    print(f"Error listening for messages: {e}")
                    # Reconnect if connection lost
                    self.socket = None
                    self.connect_socket()
                time.sleep(0.1)

        thread = threading.Thread(target=listen_thread, daemon=True)
        thread.start()
        print("Message listener started")



if __name__ == "__main__":
    print("Starting Input Monitor...")
    monitor = InputMonitor()
    monitor.start()# Start mouse