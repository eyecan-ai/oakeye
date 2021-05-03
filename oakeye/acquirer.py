from abc import ABC, abstractmethod
from sys import flags
from typing import Dict, Sequence, Tuple
from itertools import count
import time
import cv2
import numpy as np
from pipelime.sequences.samples import Sample
from oakeye.board import Board
from oakeye.device import OakDevice
from oakeye.utils.color_utils import ColorUtils


class Acquirer(ABC):
    def __init__(self) -> None:
        super().__init__()
        self._running = False

    @abstractmethod
    def acquire(self) -> Sample:
        pass

    def _stop(self) -> None:
        self._running = False

    def run(self, max_frames: int = -1, max_time: float = -1) -> Sequence[Sample]:
        def _run_condition():
            num_frames_ok = max_frames < 0 or len(samples) < max_frames
            time_ok = max_time < 0 or elapsed < max_time
            return self._running and num_frames_ok and time_ok

        self._running = True
        samples = []
        t_start = time.time()
        elapsed = 0
        while _run_condition():
            sample = self.acquire()
            if sample is not None:
                samples.append(sample)
            elapsed = time.time() - t_start
        return samples

    def __call__(self, *args, **kwargs) -> Sequence[Sample]:
        return self.run(*args, **kwargs)


class DeviceAcquirer(Acquirer):
    def __init__(self, device: OakDevice, warmup: int = 10) -> None:
        super().__init__()
        self._device = device

        # Manual focus has issues
        # See https://github.com/luxonis/depthai/issues/363
        # for _ in range(warmup):
        #     self._device.focus = self._device.focus
        #     self._device.grab()

    def acquire(self) -> Sample:
        return self._device.grab()


class GuiAcquirer(Acquirer):
    def __init__(
        self,
        acquirer: Acquirer,
        keys: Sequence[str],
        quit_key: str = "q",
        acquire_key: str = "s",
        scale_factor: int = 2,
        ranges: Dict[str, Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        if ranges is None:
            ranges = {}
        self._acquirer = acquirer
        self._quit_key = quit_key
        self._acquire_key = acquire_key
        self._keys = keys
        self._scale_factor = scale_factor
        self._counter = count()
        self._ranges = ranges

    def _show_sample(self, sample: Sample) -> None:
        h, w = sample[self._keys[0]].shape[:2]
        h //= self._scale_factor
        w //= self._scale_factor
        imgs = []
        for k in self._keys:
            if k not in sample:
                continue
            img = sample[k]
            if k in self._ranges:
                m, M = self._ranges[k]
                img = np.clip(img, m, M)
                img = ((img - m) / (M - m) * 255).astype(np.uint8)
                img = cv2.applyColorMap(img, cv2.COLORMAP_INFERNO)

            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            img = cv2.resize(img, (w, h))
            imgs.append(img)

        img = np.concatenate(imgs, 1)
        cv2.imshow("sample", img)

    def _parse_input(self, sample: Sample) -> Sample:
        res = None
        c = cv2.waitKey(1)
        if c == ord(self._quit_key):
            self._stop()
            cv2.destroyAllWindows()
        elif c == ord(self._acquire_key) and sample is not None:
            sample.id = next(self._counter)
            print("Saving sample #%d" % sample.id)
            res = sample
        return res

    def acquire(self) -> Sample:
        sample = self._acquirer.acquire()
        self._show_sample(sample)
        return self._parse_input(sample)


class CornerAcquirer(GuiAcquirer):
    def __init__(
        self, acquirer: Acquirer, keys: Sequence[str], board: Board, **kwargs
    ) -> None:
        super().__init__(acquirer, keys, **kwargs)
        self._board = board

    def acquire(self) -> Sample:
        sample = self._acquirer.acquire()
        sample_with_corners = {}
        all_ret = True
        for k in self._keys:
            img = sample[k]
            s = self._scale_factor
            factor = (img.shape[1] // s, img.shape[0] // s)
            downsampled = cv2.resize(sample[k], factor)
            corners, _ = self._board.detect_corners(downsampled)
            ret = corners is not None
            img_with_corners = self._board.draw_corners(downsampled, corners)
            factor = (img.shape[1], img.shape[0])
            img_with_corners = cv2.resize(img_with_corners, factor)
            sample_with_corners[k] = img_with_corners
            all_ret = ret and all_ret

        self._show_sample(sample_with_corners)
        sample = sample if all_ret else None
        return self._parse_input(sample)


class RectifiedAcquirer(Acquirer):
    def __init__(
        self,
        acquirer: Acquirer,
        calibration: Dict,
    ) -> None:
        super().__init__()
        self._acquirer = acquirer
        self._calibration = calibration
        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC
        size = tuple(calibration["image_size"]["left"][::-1])
        camera_l = np.array(calibration["camera_matrix"]["left"])
        camera_c = np.array(calibration["camera_matrix"]["center"])
        camera_r = np.array(calibration["camera_matrix"]["right"])
        dist_l = np.array(calibration["dist_coeff"]["left"])
        dist_c = np.array(calibration["dist_coeff"]["center"])
        dist_r = np.array(calibration["dist_coeff"]["right"])
        _, rl, rc, rr, proj_r, proj_c, proj_l, *_ = cv2.rectify3Collinear(
            camera_l,
            dist_l,
            camera_c,
            dist_c,
            camera_r,
            dist_r,
            None,
            None,
            size,
            np.array(calibration["rotation"]["left_center"]),
            np.array(calibration["translation"]["left_center"]),
            np.array(calibration["rotation"]["left_right"]),
            np.array(calibration["translation"]["left_right"]),
            0,
            size,
            flags=flags,
        )
        self._map_l = cv2.initUndistortRectifyMap(
            camera_l, dist_l, rl, proj_l, size, cv2.CV_32F
        )
        self._map_c = cv2.initUndistortRectifyMap(
            camera_c, dist_c, rc, proj_c, size, cv2.CV_32F
        )
        self._map_r = cv2.initUndistortRectifyMap(
            camera_r, dist_r, rr, proj_r, size, cv2.CV_32F
        )

    def acquire(self) -> Sample:
        sample = self._acquirer.acquire()
        sample["left"] = cv2.remap(sample["left"], *self._map_l, cv2.INTER_LINEAR)
        sample["center"] = cv2.remap(sample["center"], *self._map_c, cv2.INTER_LINEAR)
        sample["right"] = cv2.remap(sample["right"], *self._map_r, cv2.INTER_LINEAR)

        return sample


class DisparityAcquirer(Acquirer):
    def __init__(
        self,
        acquirer: Acquirer,
        disp_skip: int = 1,
        disp_diff: int = 128,
        disp_block_size: int = 7,
        disp_mode: int = cv2.STEREO_SGBM_MODE_HH4,
        disp_smooth: Tuple[int, int] = (16, 32),
        disp12_max_diff: int = 1,
        uniqueness_ratio: int = 10,
    ) -> None:

        super().__init__()
        self._acquirer = acquirer
        self._disp_diff = disp_diff
        self._sgbm = cv2.StereoSGBM_create(
            1,
            self._disp_diff,
            disp_block_size,
            mode=disp_mode,
            P1=disp_smooth[0],
            P2=disp_smooth[1],
            disp12MaxDiff=disp12_max_diff,
            uniquenessRatio=uniqueness_ratio,
        )

        self._counter = count()
        self._disp_skip = disp_skip

        self.old_cl = None
        self.old_cr = None

    def _compute_disparity(self, sample):
        left = sample["left"]
        center = ColorUtils.to_gray(sample["center"])
        right = sample["right"]

        center_left = (
            np.fliplr(
                self._sgbm.compute(
                    np.pad(np.fliplr(center), ((0, 0), (self._disp_diff, 0))),
                    np.pad(np.fliplr(left), ((0, 0), (self._disp_diff, 0))),
                )
            )[:, self._disp_diff :]
            / 16.0
        )

        center_right = (
            self._sgbm.compute(
                np.pad(center, ((0, 0), (self._disp_diff, 0))),
                np.pad(right, ((0, 0), (self._disp_diff, 0))),
            )[:, self._disp_diff :]
            / 16.0
        )

        center_left = center_left.astype(np.uint16)
        center_right = center_right.astype(np.uint16)

        return center_left, center_right

    def acquire(self) -> Sample:
        sample = self._acquirer.acquire()
        cl, cr = self._compute_disparity(sample)

        if self.old_cl is None:
            self.old_cl = np.zeros_like(sample["left"])
            self.old_cr = np.zeros_like(sample["right"])

        if next(self._counter) % self._disp_skip == 0:
            cl, cr = self._compute_disparity(sample)
            self.old_cl = cl
            self.old_cr = cr
            sample["center_left"] = cl
            sample["center_right"] = cr

        else:
            sample["center_left"] = self.old_cl
            sample["center_right"] = self.old_cr

        return sample
