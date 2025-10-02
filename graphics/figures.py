import matplotlib.pyplot as plt

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


def get_or_create_named_figure(name, screen=0, figure_position="full", axes_position="full", is_image=False):
    if name in NAMES_TO_FIGUES_AND_AXES:
        return NAMES_TO_FIGUES_AND_AXES[screen]
    fig = plt.figure()
    set_figure_position(fig, screen=screen, position=figure_position)
    ax = fig.add_axes([0, 0, 1, 1] if axes_position == "full" else axes_position)
    if is_image:
        ax.set_xticks([])
        ax.set_yticks([])
    NAMES_TO_FIGUES_AND_AXES[screen] = (fig, ax)
    return fig, ax
