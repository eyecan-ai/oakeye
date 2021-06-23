from typing import Union, Sequence, Tuple
import numpy as np
from abc import ABC, abstractmethod
import cv2
from schema import Schema
from choixe.spooks import Spook
from oakeye.utils.color_utils import ColorUtils


class Board(ABC, Spook):
    def __init__(self, pattern_size: Tuple[int, int], square_size: float) -> None:
        super().__init__()
        self._pattern_size = tuple(pattern_size)
        self._square_size = square_size

    def detect_corners(
        self, img: np.ndarray, ref_win_size: int = 11, max_iter: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        img = ColorUtils.to_gray(img)
        corners, ids = self._detect_corners(img)
        if corners is not None:
            corners = self._refine_prediction(
                img, corners, ref_win_size=ref_win_size, max_iter=max_iter
            )
        return corners, ids

    def draw_corners(self, img: np.ndarray, corners: np.ndarray) -> np.ndarray:
        img = ColorUtils.to_bgr(img)
        valid = corners is not None
        res = img
        if valid:
            res = self._draw_corners(img, corners)
        res = ColorUtils.to_rgb(img)
        return res

    def obj_points(self) -> np.ndarray:
        corner_w, corner_h = self._pattern_size
        square_size = self._square_size
        objp = np.zeros((corner_w * corner_h, 3), np.float32)
        objp[:, :2] = np.mgrid[0:corner_w, 0:corner_h].T.reshape(-1, 2) * square_size
        return objp

    def trinocular_calibration(
        self,
        left_imgs: Sequence[np.ndarray],
        center_imgs: Sequence[np.ndarray],
        right_imgs: Sequence[np.ndarray],
        ref_win_size: int = 11,
        max_iter: int = 30,
    ):
        imgs = [left_imgs, center_imgs, right_imgs]
        N = len(imgs[0])
        assert all(N == len(x) for x in imgs)

        # Detect corners
        img_pts = [[] for j in range(3)]
        corner_ids = [[] for j in range(3)]
        objpts = self.obj_points()
        for i in range(N):
            res = [
                self.detect_corners(
                    imgs[j][i], ref_win_size=ref_win_size, max_iter=max_iter
                )
                for j in range(3)
            ]
            if all([x[0] is not None and len(objpts) == len(x[0]) for x in res]):
                for j in range(3):
                    img_pts[j].append(res[j][0])
                    corner_ids[j].append(res[j][1])

        # Calibrate
        new_mtx = []
        dist = []
        for j in range(3):
            h, w = imgs[j][0].shape[:2]
            new_mtx_j, dist_j = self.calibrate((h, w), img_pts[j], corner_ids[j])
            new_mtx.append(new_mtx_j)
            dist.append(dist_j)

        # Calibrate Stereo Pairs
        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, 0.001)
        rot_pairs = []
        t_pairs = []
        for j in range(2):
            (
                _,
                new_mtx[0],
                dist[0],
                new_mtx[j + 1],
                dist[j + 1],
                rot_pair,
                t_pair,
                *_,
            ) = cv2.stereoCalibrate(
                [self.obj_points() for _ in img_pts[0]],
                img_pts[0],
                img_pts[j + 1],
                new_mtx[0],
                dist[0],
                new_mtx[j + 1],
                dist[j + 1],
                None,
                criteria=criteria,
                flags=flags,
            )
            rot_pairs.append(rot_pair)
            t_pairs.append(t_pair)

        key_remap = {
            0: "left",
            1: "center",
            2: "right",
        }
        baselines = [np.linalg.norm(x).item() for x in t_pairs]
        return {
            "camera_matrix": {key_remap[i]: x.tolist() for i, x in enumerate(new_mtx)},
            "dist_coeff": {key_remap[i]: x.tolist() for i, x in enumerate(dist)},
            "baseline": {
                "left_center": baselines[0],
                "center_right": baselines[1] - baselines[0],
                "left_right": baselines[1],
            },
            "rotation": {
                "left_center": rot_pairs[0],
                "left_right": rot_pairs[1],
            },
            "translation": {
                "left_center": t_pairs[0],
                "left_right": t_pairs[1],
            },
            "image_size": {
                "left": left_imgs[0].shape[:2],
                "center": center_imgs[0].shape[:2],
                "right": right_imgs[0].shape[:2],
            },
        }

    @abstractmethod
    def _detect_corners(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def _draw_corners(self, img: np.ndarray, corners: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def calibrate(
        self,
        size: Tuple[int, int],
        points: Sequence[np.ndarray],
        ids: Sequence[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def _refine_prediction(
        self, img, corners, ref_win_size: int = 11, max_iter: int = 1
    ) -> np.ndarray:
        K = ref_win_size
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, 0.001)
        return cv2.cornerSubPix(img, corners, (K, K), (-1, -1), criteria=criteria)

    def to_dict(self) -> dict:
        return {
            "square_size": self._square_size,
            "pattern_size": list(self._pattern_size),
        }


class Chessboard(Board):
    def __init__(
        self,
        pattern_size: Tuple[int, int],
        square_size: float,
        flags: Tuple[str] = None,
    ) -> None:
        super().__init__(pattern_size, square_size)
        if flags is None:
            flags = tuple()
        self._flags = tuple(flags)
        self._flag_value = sum([eval(f"cv2.{x}") for x in flags])

    def _detect_corners(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        _, corners = cv2.findChessboardCorners(
            img, self._pattern_size, None, flags=self._flag_value
        )
        return corners, None

    def _draw_corners(self, img: np.ndarray, corners: np.ndarray) -> np.ndarray:
        return cv2.drawChessboardCorners(img, self._pattern_size, corners, True)

    def calibrate(
        self,
        size: Tuple[int, int],
        points: Sequence[np.ndarray],
        ids: Sequence[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        obj_points = [self.obj_points() for _ in points]
        ret, matrix, dist, *_ = cv2.calibrateCamera(
            obj_points, points, tuple(reversed(size)), None, None
        )
        if not ret:
            matrix = None
            dist = None
        return matrix, dist

    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "flags": self._flags,
        }

    @classmethod
    def spook_schema(cls) -> Union[None, dict]:
        return Schema(
            {
                "pattern_size": [int, int],
                "square_size": float,
                "flags": lambda x: all([isinstance(y, str) for y in x]),
            }
        )


class Charuco(Board):
    def __init__(
        self,
        pattern_size: Tuple[int, int],
        square_size: float,
        marker_size: float,
        dictionary: str,
    ) -> None:
        super().__init__(pattern_size, square_size)
        self._dictionary = dictionary
        self._marker_size = marker_size
        d = eval(f"cv2.aruco.{dictionary}")
        self._dict_ptr = cv2.aruco.Dictionary_get(d)
        self._board = cv2.aruco.CharucoBoard_create(
            *pattern_size, square_size, marker_size, self._dict_ptr
        )
        self._total_corners = (self._pattern_size[0] - 1) * (self._pattern_size[1] - 1)

    def obj_points(self) -> np.ndarray:
        corner_w, corner_h = self._pattern_size
        corner_h -= 1
        corner_w -= 1
        square_size = self._square_size
        objp = np.zeros((corner_w * corner_h, 3), np.float32)
        objp[:, :2] = np.mgrid[0:corner_w, 0:corner_h].T.reshape(-1, 2) * square_size
        return objp

    def _detect_corners(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        corners, ids, rejected = cv2.aruco.detectMarkers(img, self._dict_ptr)
        ret = len(corners) != 0
        if ret:
            # corners, ids, rejected, _ = cv2.aruco.refineDetectedMarkers(
            #     img, self._board, corners, ids, rejected
            # )

            ret, corners, ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, img, self._board
            )

            if corners is not None and len(corners) != self._total_corners:
                corners = None
                ids = None
        else:
            corners = None
            ids = None
        return corners, ids

    def _draw_corners(self, img: np.ndarray, corners: np.ndarray) -> np.ndarray:
        return cv2.aruco.drawDetectedCornersCharuco(img, corners)

    def calibrate(
        self,
        size: Tuple[int, int],
        points: Sequence[np.ndarray],
        ids: Sequence[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        ret, matrix, dist, *_ = cv2.aruco.calibrateCameraCharuco(
            points, ids, self._board, tuple(reversed(size)), None, None
        )
        if not ret:
            matrix = None
            dist = None
        return matrix, dist

    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "marker_size": self._marker_size,
            "dictionary": self._dictionary,
        }

    @classmethod
    def spook_schema(cls) -> Union[None, dict]:
        return Schema(
            {
                "pattern_size": [int, int],
                "square_size": float,
                "marker_size": float,
                "dictionary": str,
            }
        )
