import pyperclip
import time
import threading
import pyautogui
from pynput import mouse, keyboard


class ClipboardMonitor:
    def __init__(self):
        self.last_clipboard = ""
        self.is_dragging = False
        self.kb_controller = keyboard.Controller()
        self.stop_event = threading.Event()

    def on_mouse_click(self, x, y, button, pressed):
        # Track when the mouse is being dragged (left button held down)
        if button == mouse.Button.left:
            self.is_dragging = pressed

            # When mouse is released after dragging, simulate Ctrl+C
            if not pressed and self.is_dragging:
                time.sleep(0.1)  # Small delay to ensure text is selected
                self.kb_controller.press(keyboard.Key.ctrl)
                self.kb_controller.press('c')
                self.kb_controller.release('c')
                self.kb_controller.release(keyboard.Key.ctrl)

    def monitor_clipboard(self):
        print("Clipboard monitor started. Press Ctrl+Q to exit.")
        print("Drag to select text anywhere, and it will automatically be copied and shown here.\n")

        while not self.stop_event.is_set():
            # Get current clipboard content
            current_clipboard = pyperclip.paste()

            # If clipboard content changed, display it
            if current_clipboard != self.last_clipboard and current_clipboard.strip():
                print(f"Copied at {time.strftime('%H:%M:%S')}:")
                print(f"\"{current_clipboard}\"")
                print("-" * 50)

                # Update last seen content
                self.last_clipboard = current_clipboard

            # Small delay to reduce CPU usage
            time.sleep(0.1)

    def on_key_press(self, key):
        try:
            # Exit on Ctrl+Q
            if key == keyboard.Key.ctrl_l and keyboard.KeyCode.from_char('q'):
                self.stop_event.set()
                return False
        except AttributeError:
            pass

    def start(self):
        # Start clipboard monitoring thread
        clipboard_thread = threading.Thread(target=self.monitor_clipboard)
        clipboard_thread.daemon = True
        clipboard_thread.start()

        # Start mouse listener
        mouse_listener = mouse.Listener(on_click=self.on_mouse_click)
        mouse_listener.start()

        # Start keyboard listener for exit command
        keyboard_listener = keyboard.Listener(on_press=self.on_key_press)
        keyboard_listener.start()

        try:
            # Keep main thread alive
            while not self.stop_event.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            # Cleanup
            self.stop_event.set()
            mouse_listener.stop()
            keyboard_listener.stop()


if __name__ == "__main__":
    # Check if required modules are installed
    try:
        import pyperclip
        import pyautogui
        from pynput import mouse, keyboard
    except ImportError:
        print("Required modules missing. Install them with:")
        print("pip install pyperclip pyautogui pynput")
        exit(1)

    monitor = ClipboardMonitor()
    monitor.start()