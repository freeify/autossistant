import platform
import time


def get_window_icon(wait_seconds=10):
    """
    Get the icon of the currently active window based on the operating system.
    Waits for the specified number of seconds before capturing the icon.

    Args:
        wait_seconds (int): Number of seconds to wait before capturing. Defaults to 10.

    Returns:
        The icon as an image object or file path depending on the platform.
    """
    print(f"Waiting {wait_seconds} seconds before capturing window icon...")
    print("Switch to your desired window now.")
    time.sleep(wait_seconds)
    print("Capturing window icon now!")
    system = platform.system()

    if system == "Windows":
        # Windows implementation
        import win32gui
        import win32ui
        import win32con
        from PIL import Image
        import io

        # Get handle of the foreground window
        hwnd = win32gui.GetForegroundWindow()

        # Get the icon handle
        icon_handle = win32gui.SendMessage(hwnd, win32con.WM_GETICON, win32con.ICON_BIG, 0)

        # If the above fails, try getting the small icon
        if icon_handle == 0:
            icon_handle = win32gui.SendMessage(hwnd, win32con.WM_GETICON, win32con.ICON_SMALL, 0)

        # If that also fails, try getting the icon from the window class
        if icon_handle == 0:
            icon_handle = win32gui.GetClassLong(hwnd, win32con.GCL_HICON)

        # If we got an icon handle, convert it to a PIL Image
        if icon_handle != 0:
            # Create a device context
            hdc = win32ui.CreateDCFromHandle(win32gui.GetDC(hwnd))
            dc = hdc.CreateCompatibleDC()

            # Create a bitmap
            bitmap = win32ui.CreateBitmap()
            bitmap.CreateCompatibleBitmap(hdc, 32, 32)  # Icon size

            # Select the bitmap into the DC
            old_bmp = dc.SelectObject(bitmap)

            # Draw the icon onto the bitmap
            win32gui.DrawIconEx(dc.GetHandleOutput(), 0, 0, icon_handle, 32, 32, 0, None, win32con.DI_NORMAL)

            # Convert the bitmap to a PIL Image
            bmpinfo = bitmap.GetInfo()
            bmpstr = bitmap.GetBitmapBits(True)
            icon_image = Image.frombuffer(
                'RGBA',
                (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
                bmpstr, 'raw', 'BGRA', 0, 1
            )

            # Clean up
            dc.SelectObject(old_bmp)
            dc.DeleteDC()
            hdc.DeleteDC()

            window_title = win32gui.GetWindowText(hwnd)
            return {
                "icon": icon_image,
                "window_title": window_title
            }
        else:
            return {"error": "No icon found for the current window"}

    elif system == "Darwin":  # macOS
        from AppKit import NSWorkspace, NSImage
        import tempfile
        import os

        # Get the active app
        active_app = NSWorkspace.sharedWorkspace().activeApplication()

        if active_app:
            # Get the app bundle path
            bundle_path = active_app['NSApplicationPath']

            # Get the app icon
            app_icon = NSWorkspace.sharedWorkspace().iconForFile_(bundle_path)

            # Save the icon to a temporary file
            if app_icon:
                temp_dir = tempfile.gettempdir()
                icon_path = os.path.join(temp_dir, "current_app_icon.png")

                # Convert NSImage to PNG and save
                app_icon.TIFFRepresentation().representationUsingType_properties_(
                    NSPNGFileType, None
                ).writeToFile_atomically_(icon_path, True)

                return {
                    "icon_path": icon_path,
                    "app_name": active_app['NSApplicationName']
                }

        return {"error": "Could not retrieve icon for current application"}

    elif system == "Linux":
        # Linux implementation using Xlib and GTK
        try:
            import gi
            gi.require_version('Gtk', '3.0')
            from gi.repository import Gtk, Gio
            import Xlib
            from Xlib import display
            import tempfile
            import os

            # Get the X display
            d = display.Display()
            root = d.screen().root

            # Get the active window
            active_window = root.get_full_property(
                d.intern_atom('_NET_ACTIVE_WINDOW'), Xlib.X.AnyPropertyType
            ).value[0]

            # Get the window class (WM_CLASS)
            wmclass = d.intern_atom('WM_CLASS')
            window_class = d.create_resource_object('window', active_window).get_full_property(wmclass, 0)

            if window_class:
                # The class name is the second string in WM_CLASS
                app_name = window_class.value.split(b'\0')[0].decode('utf-8')

                # Try to find the icon using GTK icon theme
                icon_theme = Gtk.IconTheme.get_default()
                icon = icon_theme.lookup_icon(app_name.lower(), 48, 0)

                if icon:
                    icon_path = icon.get_filename()
                    return {
                        "icon_path": icon_path,
                        "app_name": app_name
                    }
                else:
                    # Try to find using desktop files
                    for app in Gio.AppInfo.get_all():
                        if app.get_name().lower() == app_name.lower():
                            icon = app.get_icon()
                            if icon:
                                # Save icon to temporary file
                                temp_dir = tempfile.gettempdir()
                                icon_path = os.path.join(temp_dir, f"{app_name}_icon.png")

                                # Use GTK to save the icon
                                icon_theme = Gtk.IconTheme.get_default()
                                icon_info = icon_theme.lookup_by_gicon(icon, 48, 0)
                                if icon_info:
                                    pixbuf = icon_info.load_icon()
                                    pixbuf.savev(icon_path, "png", [], [])

                                    return {
                                        "icon_path": icon_path,
                                        "app_name": app_name
                                    }

            return {"error": "Could not retrieve icon for current window"}

        except ImportError:
            return {"error": "Required libraries not installed. Please install python-xlib, PyGObject, and GTK."}

    else:
        return {"error": f"Unsupported operating system: {system}"}


if __name__ == "__main__":
    # Default wait time is 10 seconds, but you can change it if needed
    result = get_window_icon(wait_seconds=10)
    print(result)

    # If we got an icon as a PIL Image, show it
    if "icon" in result:
        result["icon"].show()
    elif "icon_path" in result:
        print(f"Icon saved at: {result['icon_path']}")

        # Optional: open the image file with the default viewer
        import subprocess
        import os

        if platform.system() == "Windows":
            os.startfile(result["icon_path"])
        elif platform.system() == "Darwin":  # macOS
            subprocess.call(['open', result["icon_path"]])
        else:  # Linux
            subprocess.call(['xdg-open', result["icon_path"]])