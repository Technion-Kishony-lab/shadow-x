import numpy as np


def create_chessboard_pattern(n_squares, square_size=10, black=0, white=255):
    chessboard = np.full(n_squares * square_size, dtype=np.uint8, fill_value=black)
    i, j = np.indices(chessboard.shape)
    chessboard[(i // square_size + j // square_size) % 2 == 0] = white
    return chessboard


def show_chessboard_pattern(ax, square_size=10, black=0, white=255):
    ax_size = np.array([ax.bbox.height, ax.bbox.width], dtype=int)
    n_squares = ax_size // square_size
    chessboard = create_chessboard_pattern(n_squares, square_size, black, white)

    ax.imshow(chessboard, cmap='gray', vmin=0, vmax=255)
    return chessboard, n_squares

