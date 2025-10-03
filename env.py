import cv2

BACKGROUND_SCREEN = 1
BACKGROUND_SCREEN_BIN_SIZE = 2  # binning factor for the background screen

DISPLAY_SCREEN = 2

OVERHEAD_CAMERA = {
    'camera_index': 0,
    'fps': 30,
    'buffer_size': 1,
    'rotate': cv2.ROTATE_180,  # None, or cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180
    'color_order': 'BGR',
}

BACKEND = 'tkagg'  # 'tkagg', 'wxagg', 'qt5agg', 'qt6agg'


# If using TK, use get_screen_geometrys() from graphics.helpers to get the screen geometries:
SCREENS_TO_COORDS = {
    # W, H, x0, y0
    0: (1512, 945, 0, 37),
    1: (1920, 1080, 1512, 0),
    2: (1512, 982, -566, -1440),
}
