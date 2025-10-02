import numpy as np


def show_chessboard_pattern(ax, square_size=10):
    """
    square_size is the size of each square in the chessboard pattern.
    in pixels.
    """
    # number of pixels in ax:
    ax_width = int(ax.bbox.width)
    ax_height = int(ax.bbox.height)
    # number of squares in x and y direction:
    n_squares_x = ax_width // square_size
    n_squares_y = ax_height // square_size
    # create chessboard pattern:
    chessboard = np.zeros((n_squares_y * square_size, n_squares_x * square_size), dtype=np.uint8)
    i, j = np.indices(chessboard.shape)
    chessboard[(i // square_size + j // square_size) % 2 == 0] = 255

    ax.imshow(chessboard, cmap='gray', vmin=0, vmax=255)
    return chessboard

