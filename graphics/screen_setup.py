from matplotlib import pyplot as plt

from graphics.helpers import wait_for_keypress
from resources import set_matplotlib_backend


def get_figure_geomotry(figure):
    return figure.canvas.manager.window.geometry()


def parse_geometry(geom: str):
    x, y = geom.split('+')[0].split('x')
    w, h = geom.split('+')[1:]
    return int(x), int(y), int(w), int(h)


def get_screen_geometry(screen=0):
    """
    Open a figure. Ask the user to place it on the desired screen in full screen mode, then press Enter.
    """
    set_matplotlib_backend()
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title("Place this figure on the desired screen in full screen mode, then press Enter.")
    plt.show(block=False)
    wait_for_keypress(fig, options=('enter', 'return'))
    geom = get_figure_geomotry(fig)
    plt.close(fig)
    print(f"Detected screen {screen} geometry: {geom}")
    print("Parsed geometry (w, h, x, y):", parse_geometry(geom))



if __name__ == "__main__":
    get_screen_geometry(screen=0)
