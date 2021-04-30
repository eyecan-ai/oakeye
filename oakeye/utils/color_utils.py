import numpy as np
import cv2


class ColorUtils:
    @classmethod
    def to_gray(cls, img: np.ndarray, bgr: bool = False) -> np.ndarray:
        flag = cv2.COLOR_BGR2GRAY if bgr else cv2.COLOR_RGB2GRAY
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, flag)
        return img

    @classmethod
    def to_bgr(cls, img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 2:
            conv = cv2.COLOR_GRAY2BGR
        else:
            conv = cv2.COLOR_RGB2BGR
        return cv2.cvtColor(img, conv)

    @classmethod
    def to_rgb(cls, img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 2:
            conv = cv2.COLOR_GRAY2RGB
        else:
            conv = cv2.COLOR_BGR2RGB
        return cv2.cvtColor(img, conv)