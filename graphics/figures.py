import matplotlib.pyplot as plt

from env import SCREENS_TO_COORDS
from graphics.helpers import wait_for_keypress


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

    is_full = isinstance(position, str) and position == "full"

    # --- Qt-based backends ---
    if "qt" in backend:
        try:
            from PyQt5 import QtWidgets  # or PySide6/PySide2 if installed
        except ImportError:
            raise ImportError("PyQt5 (or PySide6/PySide2) is required for setting figure position with Qt backend.")
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])
        screens = app.screens()
        if screen >= len(screens):
            raise ValueError(f"Requested screen {screen}, but only {len(screens)} available.")

        geometry = screens[screen].geometry()
        window = manager.window
        if is_full:
            window.setGeometry(geometry)
            window.showFullScreen()
        elif position is None:
            window.show()
        else:
            x, y, w, h = position
            window.setGeometry(geometry.x() + x, geometry.y() + y, w, h)
            window.showNormal()
            fig.set_size_inches(w / fig.dpi, h / fig.dpi)  # to get the internal canvas size correct
        return

    # --- TkAgg backend ---
    if "tkagg" in backend:
        screen_width, screen_height, x_offset, y_offset = SCREENS_TO_COORDS[screen]
        window = manager.window  # Tkinter.Tk

        if is_full:
            new_geom = f"{300}x{300}+{x_offset}+{y_offset}"
            window.geometry(new_geom)
            # refresh the graphics:
            window.update_idletasks()
            window.attributes('-fullscreen', True)
            window.update_idletasks()
            plt.pause(0.1)  # allow time for the window to update
        else:
            x, y, w, h = position
            new_geom = f"{w}x{h}+{x_offset + x}+{y_offset + y}"
            plt.pause(0.1)  # allow time for the window to update
            window.geometry(new_geom)
            window.update_idletasks()
            plt.pause(0.1)  # allow time for the window to update
        return

    if "macosx" in backend:
        # macOS backend uses native Cocoa windows, which are not easily manipulated from Python.
        # As a workaround, we will ask the use to place the window manually.
        plt.pause(0.1)  # allow time for the window to appear
        print("Warning: Setting figure position is not supported on macOS with the default backend. "
              "Please move the window manually.")
        wait_for_keypress(fig, options=('enter',))
        return

    # --- WXAgg backend ---
    if "wxagg" in backend:
        try:
            import wx
        except ImportError:
            raise ImportError("wxPython is required for setting figure position with WX backend.")
        app = wx.App(False)
        displays = [wx.Display(i) for i in range(wx.Display.GetCount())]
        if screen >= len(displays):
            raise ValueError(f"Requested screen {screen}, but only {len(displays)} available.")
        geometry = displays[screen].GetGeometry()
        frame = manager.frame
        if is_full:
            frame.SetPosition((geometry.x, geometry.y))
            frame.SetSize((geometry.width, geometry.height))
        else:
            x, y, w, h = position
            frame.SetPosition((geometry.x + x, geometry.y + y))
            frame.SetSize((w, h))
        return

    raise NotImplementedError(f"Setting figure position not implemented for backend '{backend}'.")


def create_positioned_figure_and_axes(screen=0, figure_position="full", axes_position="full", is_image=False,
                                      remove_toolbar: bool = False):
    fig = plt.figure()
    set_figure_position(fig, screen=screen, position=figure_position)

    if remove_toolbar:
        fig.canvas.toolbar_visible = False
        fig.canvas.header_visible = False
        fig.canvas.footer_visible = False
        fig.canvas.resizable = False

    is_full = isinstance(axes_position, str) and axes_position == "full"
    ax = fig.add_axes((0, 0, 1, 1) if is_full else axes_position)
    if is_image:
        ax.set_xticks([])
        ax.set_yticks([])
    return fig, ax


if __name__ == "__main__":
    fig, ax = create_positioned_figure_and_axes(figure_position="full", axes_position="full", is_image=True)
    plt.show()
