import cv2

BACKGROUND_SCREEN = 1
BACKGROUND_SCREEN_BIN_SIZE = 1  # binning factor for the background screen

DISPLAY_SCREEN = 2

OVERHEAD_CAMERA = {
    'camera_index': 0,
    'fps': 30,
    'buffer_size': 1,
    'rotate': cv2.ROTATE_180  # None, or cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180
}

BACKEND = 'qtagg'  # 'tkagg', 'wxagg', 'qt5agg', 'qt6agg'
