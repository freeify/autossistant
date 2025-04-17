import sys
import os
import time
import threading
import traceback
import socket
import json

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
from pynput.mouse import Button, Controller as MouseController
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


class TextboxIntegration:
    def __init__(self):
        # Initialize state variables
        self.last_active_window = None
        self.last_active_window_title = None
        self.debug_mode = True  # Set to True to see debug information
        self.mouse = MouseController()
        self.stop_event = threading.Event()

        # TCP socket for communication with Electron app
        self.sock = None
        self.server_address = ('localhost', 65433)  # Match the port in the Electron app

        # Set up connection
        self.setup_socket()

        # Start monitors and set up hooks
        self.setup_keyboard_hook()
        self.start_monitoring_thread()
        self.start_mouse_monitor()

        # Log startup
        self.log("Application started")
        self.log(f"Running on: {PLATFORM} platform")

    def setup_socket(self):
        """Set up socket communication with Electron app"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect(self.server_address)
            self.log("Socket connection established with Electron app")
        except Exception as e:
            self.log(f"Failed to connect to Electron app: {e}", error=True)
            print("Make sure the Electron app is running before starting this script")
            sys.exit(1)

    def send_command(self, command, text=""):
        """Send command to Electron app"""
        try:
            message = f"{command}|{text}\n"
            self.sock.sendall(message.encode())
            self.log(f"Sent command: {command}")
        except Exception as e:
            self.log(f"Failed to send command: {e}", error=True)

    def setup_keyboard_hook(self):
        """Set up keyboard hooks for activation"""
        try:
            # For global hotkey activation
            keyboard.add_hotkey('ctrl+shift+t', self.show_textbox)

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

    def start_mouse_monitor(self):
        """Start thread to monitor mouse selections"""
        self.mouse_monitor = threading.Thread(target=self.monitor_mouse_selections, daemon=True)
        self.mouse_monitor.start()
        self.log("Mouse selection monitoring started")

    def monitor_mouse_selections(self):
        """Monitor mouse selections"""
        with mouse.Listener(on_click=self.on_click) as listener:
            self.stop_event.wait()
            listener.stop()

    def on_click(self, x, y, button, pressed):
        """Track mouse clicks"""
        if pressed and button == Button.left:
            # Send click position to Electron
            self.send_command("click_detected", f"{x},{y}")

            # Check if text is selected after click
            time.sleep(0.2)  # Brief delay to allow selection to complete
            try:
                # Try to get selected text via clipboard
                original_clipboard = pyperclip.paste()

                # Attempt copy operation
                pyautogui.hotkey('ctrl', 'c') if sys.platform != 'darwin' else pyautogui.hotkey('command', 'c')
                time.sleep(0.1)

                selected_text = pyperclip.paste()

                # If clipboard changed, we have a selection
                if selected_text != original_clipboard and selected_text.strip():
                    self.send_command("selection_made", selected_text)

                # Restore original clipboard
                pyperclip.copy(original_clipboard)

            except Exception as e:
                self.log(f"Error checking selection: {e}", error=True)

    def monitor_active_window(self):
        """Monitor and record the active window information"""
        while self.monitoring_active and not self.stop_event.is_set():
            try:
                # Platform-specific active window detection
                if PLATFORM == "windows":
                    hwnd = win32gui.GetForegroundWindow()
                    if hwnd:
                        window_title = win32gui.GetWindowText(hwnd)
                        if window_title and "Electron" not in window_title:
                            self.last_active_window = hwnd
                            self.last_active_window_title = window_title

                elif PLATFORM == "mac":
                    # macOS approach - if pygetwindow is working
                    try:
                        active_window = gw.getActiveWindow()
                        if active_window and "Electron" not in active_window.title:
                            self.last_active_window = active_window
                            self.last_active_window_title = active_window.title
                    except:
                        # Fallback using AppKit
                        app = AppKit.NSWorkspace.sharedWorkspace().activeApplication()
                        if app:
                            app_name = app['NSApplicationName']
                            self.last_active_window_title = app_name

                elif PLATFORM == "linux":
                    # Basic X11 approach
                    display = Xlib.display.Display()
                    window = display.get_input_focus().focus
                    wmname = window.get_wm_name()
                    if wmname and "Electron" not in wmname:
                        self.last_active_window = window
                        self.last_active_window_title = wmname

            except Exception as e:
                self.log(f"Error monitoring window: {e}", error=True)

            time.sleep(0.5)  # Check every half second

    def on_key_press(self, key):
        """Track key presses to infer cursor position"""
        # We're now just using this to detect keyboard activity
        # Actual text handling is done through the Electron app
        pass

    def show_textbox(self):
        """Show the textbox by sending a command to Electron"""
        try:
            self.send_command("toggle_third_window")
            self.log("Textbox display command sent")
        except Exception as e:
            self.log(f"Error showing textbox: {e}", error=True)

    def log(self, message, error=False):
        """Log message for debugging"""
        if self.debug_mode:
            prefix = "ERROR: " if error else "INFO: "
            print(f"{prefix}{message}")
            if error:
                traceback.print_exc()

    def shutdown(self):
        """Clean shutdown"""
        self.log("Shutting down...")
        self.monitoring_active = False
        self.stop_event.set()
        if self.sock:
            self.sock.close()


def main():
    """Main entry point"""
    print("Starting Textbox Integration...")
    print(f"Platform detected: {sys.platform}")

    try:
        app = TextboxIntegration()

        # Show instructions
        print("\nInstructions:")
        print("1. Press Ctrl+Shift+T to show the textbox overlay")
        print("2. Select text with your mouse to automatically show the textbox")
        print("3. Type the text you want to heinsert in the textbox and press Enter")
        print("\nDebug information will be shown in this console.")
        print("Application running... (Press Ctrl+C in this console to exit)")

        # Keep the main thread alive
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nApplication terminated by user")
        if 'app' in locals():
            app.shutdown()
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
        if 'app' in locals():
            app.shutdown()
        input("Press Enter to exit...")


if __name__ == "__main__":
    main()