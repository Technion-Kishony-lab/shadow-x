import numpy as np
from matplotlib import pyplot as plt


def get_size_in_pixels(ax):
    bbox = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
    width_in_pixels = bbox.width * ax.figure.dpi
    height_in_pixels = bbox.height * ax.figure.dpi
    return np.array([height_in_pixels, width_in_pixels], dtype=int)


def create_chessboard_binary_pattern(n_squares, square_size=10):
    chessboard = np.zeros(n_squares * square_size, dtype=bool)
    i, j = np.indices(chessboard.shape)
    chessboard[(i // square_size + j // square_size) % 2 == 0] = True
    return chessboard


def get_chessboard_image(size, square_size=10, black=0, white=255, margin_color=0):
    n_squares = np.array(size) // square_size
    chessboard_binary = create_chessboard_binary_pattern(n_squares, square_size)
    chessboard = (chessboard_binary * (white - black) + black).astype(np.uint8)

    if margin_color is not None:
        chessboard_with_margin = np.full(size, dtype=np.uint8, fill_value=margin_color)
        chessboard_with_margin[:chessboard.shape[0], :chessboard.shape[1]] = chessboard
        chessboard = chessboard_with_margin

    return chessboard, n_squares

