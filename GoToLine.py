import tkinter as tk
import sys
import os
import time
import threading
import traceback

# Define needed dependencies
DEPENDENCIES = [
    "keyboard",
    "pynput",
    "pygetwindow",
    "pyautogui",
    "pyperclip"
]


# Check and install missing dependencies
def check_dependencies():
    missing = []
    for dep in DEPENDENCIES:
        try:
            __import__(dep)
        except ImportError:
            missing.append(dep)

    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print("Installing missing dependencies...")
        try:
            import pip
            for dep in missing:
                pip.main(['install', dep])
            print("Dependencies installed successfully!")
            # Force a small delay to allow imports to become available
            time.sleep(2)
        except Exception as e:
            print(f"Failed to install dependencies: {e}")
            print("Please run: pip install " + " ".join(missing))
            sys.exit(1)


# Import dependencies after checking
check_dependencies()

import keyboard
import pyperclip
from pynput import mouse, keyboard as kb
import pyautogui

# Import platform-specific modules with error handling
try:
    if sys.platform == 'win32':
        import pygetwindow as gw
        import win32gui
        import win32con

        PLATFORM = "windows"
    elif sys.platform == 'darwin':
        import pygetwindow as gw
        import AppKit

        PLATFORM = "mac"
    else:  # Linux and others
        import Xlib.display

        PLATFORM = "linux"
except ImportError as e:
    print(f"Warning: Some platform-specific modules couldn't be imported: {e}")
    print("Falling back to limited functionality mode")
    PLATFORM = "limited"


class OverlayTextApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Text Overlay")
        self.setup_ui()

        # Initialize state variables
        self.last_active_window = None
        self.last_active_window_title = None
        self.is_visible = False
        self.debug_mode = True  # Set to True to see debug information

        # Start monitors and set up hooks
        self.setup_keyboard_hook()
        self.start_monitoring_thread()

        # Log startup
        self.log("Application started")
        self.log(f"Running on: {PLATFORM} platform")

    def setup_ui(self):
        """Set up the user interface"""
        # Configure the main window
        self.root.attributes('-topmost', True)
        self.root.attributes('-alpha', 0.9)

        # On Windows, we can make a tool window
        if sys.platform == 'win32':
            self.root.attributes("-toolwindow", True)

        # Size and position
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = 400
        window_height = 150
        position_x = (screen_width - window_width) // 2
        position_y = (screen_height - window_height) // 3
        self.root.geometry(f"{window_width}x{window_height}+{position_x}+{position_y}")

        # Frame for content
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Instructions label
        instructions = tk.Label(main_frame, text="Type text to insert at last cursor position:")
        instructions.pack(pady=(0, 5))

        # Text entry
        self.text_entry = tk.Entry(main_frame, width=40, font=("Arial", 12))
        self.text_entry.pack(fill=tk.X, pady=5)
        self.text_entry.bind("<Return>", self.insert_text_and_hide)
        self.text_entry.bind("<Escape>", lambda e: self.hide_overlay())

        # Buttons frame
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=5)

        # Insert button
        self.insert_button = tk.Button(button_frame, text="Insert Text",
                                       command=self.insert_text_and_hide)
        self.insert_button.pack(side=tk.LEFT, padx=5)

        # Cancel button
        self.cancel_button = tk.Button(button_frame, text="Cancel",
                                       command=self.hide_overlay)
        self.cancel_button.pack(side=tk.LEFT, padx=5)

        # Status frame
        status_frame = tk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=5)

        # Status label
        self.status_label = tk.Label(status_frame, text="Ready", fg="gray",
                                     anchor="w", font=("Arial", 9))
        self.status_label.pack(fill=tk.X)

        # Initially hide the window
        self.root.withdraw()

    def setup_keyboard_hook(self):
        """Set up keyboard hooks for activation"""
        try:
            # For global hotkey activation
            keyboard.add_hotkey('ctrl+shift+t', self.show_overlay)

            # Start keyboard listener to track typing activity
            self.kb_listener = kb.Listener(on_press=self.on_key_press)
            self.kb_listener.daemon = True  # Allow the program to exit
            self.kb_listener.start()

            self.log("Keyboard hooks set up successfully")
        except Exception as e:
            self.log(f"Failed to set up keyboard hooks: {e}", error=True)

    def start_monitoring_thread(self):
        """Start thread to monitor active window"""
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self.monitor_active_window, daemon=True)
        self.monitor_thread.start()
        self.log("Window monitoring thread started")

    def monitor_active_window(self):
        """Monitor and record the active window information"""
        while self.monitoring_active:
            try:
                # Platform-specific active window detection
                if PLATFORM == "windows":
                    hwnd = win32gui.GetForegroundWindow()
                    if hwnd:
                        window_title = win32gui.GetWindowText(hwnd)
                        if window_title and window_title != "Text Overlay":
                            self.last_active_window = hwnd
                            self.last_active_window_title = window_title
                            self.update_status(f"Last window: {window_title[:30]}...")

                elif PLATFORM == "mac":
                    # macOS approach - if pygetwindow is working
                    try:
                        active_window = gw.getActiveWindow()
                        if active_window and active_window.title != "Text Overlay":
                            self.last_active_window = active_window
                            self.last_active_window_title = active_window.title
                            self.update_status(f"Last window: {active_window.title[:30]}...")
                    except:
                        # Fallback using AppKit
                        app = AppKit.NSWorkspace.sharedWorkspace().activeApplication()
                        if app:
                            app_name = app['NSApplicationName']
                            self.last_active_window_title = app_name
                            self.update_status(f"Last window: {app_name}")

                elif PLATFORM == "linux":
                    # Basic X11 approach
                    display = Xlib.display.Display()
                    window = display.get_input_focus().focus
                    wmname = window.get_wm_name()
                    if wmname and "Text Overlay" not in wmname:
                        self.last_active_window = window
                        self.last_active_window_title = wmname
                        self.update_status(f"Last window: {wmname[:30]}...")

                elif PLATFORM == "limited":
                    # Limited mode - just track if our window is active or not
                    if not self.is_visible:
                        self.last_active_window_title = "Last application"

            except Exception as e:
                self.log(f"Error monitoring window: {e}", error=True)

            time.sleep(0.5)  # Check every half second

    def on_key_press(self, key):
        """Track key presses to infer cursor position"""
        # Only track when our window is not visible and in a tracked window
        if not self.is_visible and self.last_active_window_title:
            # We don't need to do anything special here, just knowing user was typing
            # in the last active window is enough for our simplified approach
            pass

    def show_overlay(self):
        """Show the overlay window"""
        if not self.is_visible:
            try:
                # Save the current active window before we take focus
                self.update_status("Ready to insert text")

                # Show the window
                self.root.deiconify()
                self.is_visible = True

                # Reset and focus text entry
                self.text_entry.delete(0, tk.END)
                self.text_entry.focus_set()

                self.log("Overlay shown")
            except Exception as e:
                self.log(f"Error showing overlay: {e}", error=True)

    def hide_overlay(self):
        """Hide the overlay window"""
        if self.is_visible:
            self.root.withdraw()
            self.is_visible = False
            self.log("Overlay hidden")

    def insert_text_and_hide(self, event=None):
        """Insert text into the last active application"""
        text_to_insert = self.text_entry.get().strip()
        if not text_to_insert:
            self.hide_overlay()
            return

        if not self.last_active_window_title:
            self.update_status("No previous window detected", error=True)
            return

        self.log(f"Attempting to insert text into {self.last_active_window_title}")

        # Save original clipboard content
        original_clipboard = pyperclip.paste()

        try:
            # Hide our window first
            self.hide_overlay()
            time.sleep(0.3)  # Wait briefly

            # Copy text to clipboard
            pyperclip.copy(text_to_insert)

            # Platform specific window activation
            if PLATFORM == "windows":
                try:
                    # Try to activate the window
                    win32gui.SetForegroundWindow(self.last_active_window)
                    time.sleep(0.3)  # Allow time for focus
                except Exception as e:
                    self.log(f"Failed to focus window: {e}", error=True)

            elif PLATFORM == "mac":
                try:
                    # Try pygetwindow approach first
                    if hasattr(self.last_active_window, 'activate'):
                        self.last_active_window.activate()
                    else:
                        # Try AppleScript as fallback
                        os.system(f"osascript -e 'tell application \"{self.last_active_window_title}\" to activate'")
                    time.sleep(0.3)
                except Exception as e:
                    self.log(f"Failed to focus window: {e}", error=True)

            # Wait a moment for window to activate
            time.sleep(0.3)

            # Try paste (ctrl+v or command+v)
            if sys.platform == 'darwin':
                pyautogui.hotkey('command', 'v')
            else:
                pyautogui.hotkey('ctrl', 'v')

            self.log("Text inserted successfully")

            # Restore original clipboard after a brief pause
            time.sleep(0.3)
            pyperclip.copy(original_clipboard)

        except Exception as e:
            self.log(f"Error during text insertion: {e}", error=True)
            self.show_error(f"Failed to insert text: {str(e)}")

    def show_error(self, message):
        """Show error message in overlay"""
        self.root.deiconify()
        self.is_visible = True
        self.update_status(message, error=True)

    def update_status(self, message, error=False):
        """Update status label with message"""
        if not hasattr(self, 'status_label'):
            return

        color = "red" if error else "gray"
        try:
            self.status_label.config(text=message, fg=color)
            # Force UI update
            self.root.update_idletasks()
        except:
            pass  # Ignore if window is being destroyed

    def log(self, message, error=False):
        """Log message for debugging"""
        if self.debug_mode:
            prefix = "ERROR: " if error else "INFO: "
            print(f"{prefix}{message}")
            if error:
                traceback.print_exc()


def main():
    """Main entry point"""
    print("Starting Text Overlay Application...")
    print(f"Platform detected: {sys.platform}")

    try:
        # Create and run app
        root = tk.Tk()
        app = OverlayTextApp(root)

        # Show instructions
        print("\nInstructions:")
        print("1. Press Ctrl+Shift+T to show the overlay")
        print("2. Type the text you want to insert")
        print("3. Press Enter or click 'Insert Text'")
        print("4. Press Escape or click 'Cancel' to hide the overlay")
        print("\nDebug information will be shown in this console.")
        print("Application running... (Press Ctrl+C in this console to exit)")

        # Start the Tkinter event loop
        root.mainloop()
    except KeyboardInterrupt:
        print("\nApplication terminated by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
        input("Press Enter to exit...")


if __name__ == "__main__":
    main()