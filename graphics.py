import matplotlib
import matplotlib.pyplot as plt



def set_figure_position(fig, screen=0, position="full"):
    """
    Place a matplotlib figure on a specific screen and position.

    Args:
        fig : matplotlib.figure.Figure
            The figure to move.
        screen : int
            Screen index (0 = primary, 1 = second monitor, etc.).
        position : str or list
            - "full" → fullscreen on the target screen
            - [x, y, w, h] → custom geometry in pixels, relative to that screen
    """
    import matplotlib
    import matplotlib.pyplot as plt
    backend = matplotlib.get_backend().lower()
    manager = plt._pylab_helpers.Gcf.get_fig_manager(fig.number)

    # --- Qt-based backends ---
    if "qt" in backend:
        try:
            from PyQt5 import QtWidgets  # or PySide6/PySide2 if installed
            app = QtWidgets.QApplication.instance()
            if app is None:
                app = QtWidgets.QApplication([])
            screens = app.screens()
            if screen >= len(screens):
                raise ValueError(f"Requested screen {screen}, but only {len(screens)} available.")

            geometry = screens[screen].geometry()
            window = manager.window
            if position == "full":
                window.setGeometry(geometry)
                window.showFullScreen()
            else:
                x, y, w, h = position
                window.setGeometry(geometry.x() + x, geometry.y() + y, w, h)
                window.showNormal()
            return
        except ImportError:
            print("Qt backend active but no Qt bindings installed.")

    # --- TkAgg backend ---
    if "tkagg" in backend:
        window = manager.window  # Tkinter.Tk
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()

        x_offset = screen * screen_width
        y_offset = 0

        if position == "full":
            window.geometry(f"{screen_width}x{screen_height}+{x_offset}+{y_offset}")
            window.attributes('-fullscreen', True)
        else:
            x, y, w, h = position
            window.geometry(f"{w}x{h}+{x_offset + x}+{y_offset + y}")
        return

    # --- WXAgg backend ---
    if "wxagg" in backend:
        import wx
        app = wx.App(False)
        displays = [wx.Display(i) for i in range(wx.Display.GetCount())]
        if screen >= len(displays):
            raise ValueError(f"Requested screen {screen}, but only {len(displays)} available.")
        geometry = displays[screen].GetGeometry()
        frame = manager.frame
        if position == "full":
            frame.SetPosition((geometry.x, geometry.y))
            frame.SetSize((geometry.width, geometry.height))
        else:
            x, y, w, h = position
            frame.SetPosition((geometry.x + x, geometry.y + y))
            frame.SetSize((w, h))
        return

    print(f"Fullscreen/positioning not implemented for backend: {backend}")




def set_figure_as_full_screen(fig, screen_number=0):
    """
    Open a matplotlib figure fullscreen on a specific monitor.

    Args:
        fig: matplotlib Figure object
        screen_number (int): index of the screen (0 = primary, 1 = second monitor, etc.)
    """
    backend = matplotlib.get_backend().lower()
    manager = plt._pylab_helpers.Gcf.get_fig_manager(fig.number)

    # --- Qt backends (Qt5Agg / QtAgg / Qt6Agg) ---
    if "qt" in backend:
        try:
            from PyQt5 import QtWidgets  # or PySide2/PySide6 if installed
            app = QtWidgets.QApplication.instance()
            if app is None:
                app = QtWidgets.QApplication([])

            screens = app.screens()
            if screen_number >= len(screens):
                raise ValueError(f"Requested screen {screen_number}, but only {len(screens)} available.")

            geometry = screens[screen_number].geometry()
            window = manager.window
            window.setGeometry(geometry)
            window.showFullScreen()
            return
        except ImportError:
            print("Qt backend active but no PyQt/PySide installed.")

    # --- Tk backend (TkAgg) ---
    if "tkagg" in backend:
        window = manager.window  # Tkinter.Tk object
        # Tkinter only knows about the "virtual screen" → you must know offsets
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()

        # naive guess: monitors are aligned horizontally
        x_offset = screen_number * screen_width
        y_offset = 0
        window.geometry(f"{screen_width}x{screen_height}+{x_offset}+{y_offset}")
        window.attributes('-fullscreen', True)
        return

    # --- Wx backend (WXAgg) ---
    if "wxagg" in backend:
        import wx
        app = wx.App(False)
        displays = [wx.Display(i) for i in range(wx.Display.GetCount())]
        if screen_number >= len(displays):
            raise ValueError(f"Requested screen {screen_number}, but only {len(displays)} available.")
        geometry = displays[screen_number].GetGeometry()
        frame = manager.frame
        frame.SetPosition((geometry.x, geometry.y))
        frame.SetSize((geometry.width, geometry.height))
        return

    # --- Fallback ---
    print(f"Fullscreen not implemented for backend: {backend}")


# ------------------ Example usage ------------------
if __name__ == "__main__":
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    set_figure_as_full_screen(fig, screen_number=1)  # try second screen
    plt.show()
