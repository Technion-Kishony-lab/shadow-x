import numpy as np


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


def get_chessboard_image(size, num_tile_rows=10, black=0, white=255, margin_color=0):
    square_size = size[0] // num_tile_rows
    n_squares = np.array(size) // square_size
    chessboard_binary = create_chessboard_binary_pattern(n_squares, square_size)
    chessboard = (chessboard_binary * (white - black) + black).astype(np.uint8)

    if margin_color is not None:
        chessboard_with_margin = np.full(size, dtype=np.uint8, fill_value=margin_color)
        chessboard_with_margin[:chessboard.shape[0], :chessboard.shape[1]] = chessboard
        chessboard = chessboard_with_margin

    return chessboard, n_squares, square_size


def get_array_of_circles_image(size, num_rows, radius_frac=0.25, margin_frac=0.25,
                               background_color=(0, 0, 0), circle_color=(255, 255, 255)):
    d = size[0] / (num_rows - 1 + (margin_frac + radius_frac) * 2)
    radius = int(radius_frac * d)
    margin = int((margin_frac + radius_frac) * d)
    ys = np.linspace(margin, size[0] - margin, num_rows, dtype=int)
    num_cols = num_rows * size[1] // size[0]
    xs = np.linspace(margin, size[1] - margin, num_cols, dtype=int)
    image = np.zeros([size[0], size[1], 3], dtype=np.uint8)
    image[:, :] = background_color
    yy, xx = np.indices(size)
    for y in ys:
        for x in xs:
            mask = (yy - y) ** 2 + (xx - x) ** 2 <= radius ** 2
            image[mask] = circle_color
    return image, xs, ys
