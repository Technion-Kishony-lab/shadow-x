import matplotlib.pyplot as plt

from env import SCREENS_TO_COORDS

NAMES_TO_FIGUES_AND_AXES = {}


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
            return
        except ImportError:
            print("Qt backend active but no Qt bindings installed.")

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

        print(f"Set window geometry to: {new_geom}, fullscreen: {is_full}")
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
        if is_full:
            frame.SetPosition((geometry.x, geometry.y))
            frame.SetSize((geometry.width, geometry.height))
        else:
            x, y, w, h = position
            frame.SetPosition((geometry.x + x, geometry.y + y))
            frame.SetSize((w, h))
        return

    print(f"Fullscreen/positioning not implemented for backend: {backend}")


def get_or_create_named_figure(name, screen=0, figure_position="full", axes_position="full", is_image=False,
                               remove_toolbar: bool = False):
    if name in NAMES_TO_FIGUES_AND_AXES:
        return NAMES_TO_FIGUES_AND_AXES[name]
    fig = plt.figure()
    set_figure_position(fig, screen=screen, position=figure_position)
    is_full = isinstance(axes_position, str) and axes_position == "full"
    if remove_toolbar:
        fig.canvas.toolbar_visible = False
        fig.canvas.header_visible = False
        fig.canvas.footer_visible = False
        fig.canvas.resizable = False

    ax = fig.add_axes([0, 0, 1, 1] if is_full else axes_position)
    if is_image:
        ax.set_xticks([])
        ax.set_yticks([])
    NAMES_TO_FIGUES_AND_AXES[name] = (fig, ax)
    return fig, ax


def get_figure_if_existing(name):
    return NAMES_TO_FIGUES_AND_AXES.get(name, (None, None))
