import numpy as np


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
