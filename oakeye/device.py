from itertools import count
from threading import Event, Thread
from queue import Queue
from typing import Dict, Sequence
from schema import Schema

import depthai as dai
from choixe.configurations import XConfig
from pipelime.sequences.samples import PlainSample, Sample
import numpy as np


class SyncSystem:
    def __init__(self, allowed_instances: Sequence[int]):
        self._allowed_instances = allowed_instances
        self.seq_packets = {}
        self.last_synced_seq = None

    def seq(self, packet):
        return packet.getSequenceNum()

    def has_keys(self, obj, keys):
        return all(stream in obj for stream in keys)

    def add_packet(self, packet):
        if packet is not None and packet.getInstanceNum() in self._allowed_instances:
            seq_key = self.seq(packet)
            self.seq_packets[seq_key] = {
                **self.seq_packets.get(seq_key, {}),
                packet.getInstanceNum(): packet,
            }

    def get_synced(self):
        results = []
        for key in sorted(list(self.seq_packets.keys())):
            if self.has_keys(self.seq_packets[key], self._allowed_instances):
                results.append(self.seq_packets[key])
                self.last_synced_seq = key
        if len(results) > 0:
            self.collect_garbage()
        return results

    def collect_garbage(self):
        for key in list(self.seq_packets.keys()):
            if key <= self.last_synced_seq:
                del self.seq_packets[key]


class TimeSyncSystem:
    def __init__(self, allowed_instances: Sequence[int]):
        self._allowed_instances = allowed_instances
        self._in_packets = {x: [] for x in self._allowed_instances}

    def has_keys(self, obj, keys):
        return all(stream in obj for stream in keys)

    def add_packet(self, packet):
        self._in_packets[packet.getInstanceNum()].append(packet)

    def get_synced(self):
        results = []
        f = {x: len(self._in_packets[x]) for x in self._in_packets}
        if not all([x > 0 for x in f.values()]):
            return results
        most_f = [k for k, v in f.items() if v == max(f.values())][0]
        tss = {
            x: np.array([p.getTimestamp() for p in self._in_packets[x]])
            for x in self._allowed_instances
        }
        for p in self._in_packets[most_f]:
            synced = {most_f: p}
            for x in self._allowed_instances:
                if x != most_f and f[x] > 0:
                    diffs = np.abs(tss[x] - p.getTimestamp())
                    nearest_idx = np.argmin(diffs)
                    nearest = self._in_packets[x][nearest_idx]
                    synced.update({x: nearest})
            if self.has_keys(synced, self._allowed_instances):
                results.append(synced)
        if len(results) > 0:
            self.collect_garbage(results)
        return results

    def collect_garbage(self, results):
        self._in_packets = {x: [results[-1][x]] for x in self._allowed_instances}


class OakDevice:
    QUEUE_LEFT = "left"
    QUEUE_RIGHT = "right"
    QUEUE_CENTER = "center"
    QUEUE_DEPTH = "depth"
    KEY_MAPPING = {0: "center", 1: "left", 2: "right", 3: "depth"}
    REV_KEY_MAPPING = {v: k for k, v in KEY_MAPPING.items()}
    SYNC_METHODS_MAP = {"id": SyncSystem, "time": TimeSyncSystem}

    def __init__(
        self, device: dai.Device, maxsize: int = 4, focus: int = 0, syncing_method="id"
    ) -> None:
        self._device = device
        names = self._device.getOutputQueueNames()
        # self._syncing = SyncSystem([self.REV_KEY_MAPPING[x] for x in names])
        # self._syncing = TimeSyncSystem([self.REV_KEY_MAPPING[x] for x in names])
        self._syncing = self.SYNC_METHODS_MAP[syncing_method](
            [self.REV_KEY_MAPPING[x] for x in names]
        )
        self._queues = [
            self._device.getOutputQueue(name=x, maxSize=maxsize, blocking=False)
            for x in names
        ]
        self._stop_event = Event()
        self._buffer = Queue(maxsize=1)
        self._counter = count()

        self._acquirer_thread = Thread(target=self._grab)
        self._acquirer_thread.start()

        self._focus = focus

    @property
    def focus(self) -> int:
        return self._focus

    @focus.setter
    def focus(self, value: int) -> None:
        self._focus = value
        q_control = self._device.getInputQueue(name="cam_control")
        cam_control = dai.CameraControl()
        cam_control.setManualFocus(value)
        q_control.send(cam_control)

    def _to_sample(self, synced_packet: Dict[int, dai.ImgFrame]) -> Sample:
        sample = {self.KEY_MAPPING[k]: v.getFrame() for k, v in synced_packet.items()}
        sample = PlainSample(sample, id=next(self._counter))
        return sample

    def _grab(self) -> None:
        def _add_packet() -> None:
            for q in self._queues:
                packet = q.tryGet()
                if packet is not None:
                    name = q.getName()
                    packet.setInstanceNum(self.REV_KEY_MAPPING[name])
                    self._syncing.add_packet(packet)

        while not self._stop_event.is_set():
            _add_packet()
            res = self._syncing.get_synced()
            if len(res) > 0:
                sample = self._to_sample(res[-1])
                self._buffer.put(sample)
                self._buffer.join()

    def grab(self, blocking: bool = True) -> Sample:
        if self._buffer.empty() and not blocking:
            return None
        res = self._buffer.get()
        self._buffer.task_done()
        return res

    def close(self) -> None:
        self._stop_event.set()
        self.grab(blocking=False)
        self._acquirer_thread.join()
        self._device.close()


class DeviceCfg(XConfig):
    def __init__(self, filename: str, **kwargs):
        super().__init__(filename=filename, **kwargs)
        schema = Schema(
            {
                "depth": bool,
                "focus": int,
                "autofocus": bool,
                "preview_size": [int, int],
                "resolutions": {
                    "left": lambda x: x in OakDeviceFactory.MONO_CAM_RES_MAP,
                    "center": lambda x: x in OakDeviceFactory.RGB_CAM_RES_MAP,
                    "right": lambda x: x in OakDeviceFactory.MONO_CAM_RES_MAP,
                },
            }
        )
        self.set_schema(schema)
        self.validate()


class OakDeviceFactory:

    RGB_CAM_RES_MAP = {
        "1080p": dai.ColorCameraProperties.SensorResolution.THE_1080_P,
        "4k": dai.ColorCameraProperties.SensorResolution.THE_4_K,
        "12mp": dai.ColorCameraProperties.SensorResolution.THE_12_MP,
        "13mp": dai.ColorCameraProperties.SensorResolution.THE_13_MP,
    }
    MONO_CAM_RES_MAP = {
        "400p": dai.MonoCameraProperties.SensorResolution.THE_400_P,
        "480p": dai.MonoCameraProperties.SensorResolution.THE_480_P,
        "800p": dai.MonoCameraProperties.SensorResolution.THE_800_P,
        "720p": dai.MonoCameraProperties.SensorResolution.THE_720_P,
    }

    def create(self, cfg: DeviceCfg) -> OakDevice:

        pipeline = dai.Pipeline()

        # RGB (center) Camera
        center_camera = pipeline.createColorCamera()
        center_camera.setPreviewSize(*cfg.preview_size)
        center_camera.setBoardSocket(dai.CameraBoardSocket.RGB)
        center_camera.setResolution(self.RGB_CAM_RES_MAP[cfg.resolutions.center])
        center_camera.setInterleaved(True)
        center_camera.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        center_camera.initialControl.setManualFocus(cfg.focus)

        # This may be redundant when setManualFocus is used
        if not cfg.autofocus:
            center_camera.initialControl.setAutoFocusMode(
                dai.RawCameraControl.AutoFocusMode.OFF
            )

        # left camera
        left_camera = pipeline.createMonoCamera()
        left_camera.setResolution(self.MONO_CAM_RES_MAP[cfg.resolutions.left])
        left_camera.setBoardSocket(dai.CameraBoardSocket.LEFT)

        # right camera
        right_camera = pipeline.createMonoCamera()
        right_camera.setResolution(self.MONO_CAM_RES_MAP[cfg.resolutions.right])
        right_camera.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        # center stream
        center_stream = pipeline.createXLinkOut()
        center_stream.setStreamName(OakDevice.QUEUE_CENTER)
        center_camera.preview.link(center_stream.input)

        # left stream
        left_stream = pipeline.createXLinkOut()
        left_stream.setStreamName(OakDevice.QUEUE_LEFT)
        left_camera.out.link(left_stream.input)

        # right stream
        right_stream = pipeline.createXLinkOut()
        right_stream.setStreamName(OakDevice.QUEUE_RIGHT)
        right_camera.out.link(right_stream.input)

        if cfg.depth:
            # Stereo camera
            stereo_camera = pipeline.createStereoDepth()
            stereo_camera.setConfidenceThreshold(255)
            stereo_camera.setMedianFilter(
                dai.StereoDepthProperties.MedianFilter.MEDIAN_OFF
            )
            stereo_camera.setLeftRightCheck(True)
            stereo_camera.setExtendedDisparity(False)
            stereo_camera.setSubpixel(False)

            # Link LEFT/RIGHT -> STEREO
            left_camera.out.link(stereo_camera.left)
            right_camera.out.link(stereo_camera.right)

            # Depth Stream
            depth_stream = pipeline.createXLinkOut()
            depth_stream.setStreamName(OakDevice.QUEUE_DEPTH)
            stereo_camera.depth.link(depth_stream.input)

        # camera control
        cam_control_in = pipeline.createXLinkIn()
        cam_control_in.setStreamName("cam_control")
        cam_control_in.out.link(center_camera.inputControl)

        # device
        device = dai.Device(pipeline, usb2Mode=False)
        device.startPipeline()
        device.setLogLevel(dai.LogLevel.DEBUG)

        return OakDevice(device, focus=cfg.focus, syncing_method=cfg.syncing_method)
