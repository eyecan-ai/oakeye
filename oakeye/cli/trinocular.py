from pathlib import Path
from choixe.configurations import XConfig
from choixe.spooks import Spook
import click
from click.exceptions import ClickException
from pipelime.sequences.writers.filesystem import UnderfolderWriter
from pipelime.sequences.readers.filesystem import UnderfolderReader
import oakeye
from oakeye.device import OakDeviceFactory
from oakeye.board import Board
from oakeye.acquirer import (
    CornerAcquirer,
    DeviceAcquirer,
    GuiAcquirer,
    RectifiedAcquirer,
    DisparityAcquirer,
)


@click.group("trinocular")
def trinocular():
    pass


@click.command("calibrate")
@click.option(
    "-i", "--input_folder", type=Path, default=None, help="Path to input folder"
)
@click.option(
    "-o", "--output_folder", type=Path, required=True, help="Path to output folder"
)
@click.option(
    "-b", "--board_cfg", type=Path, default=None, help="Path to board config file"
)
@click.option(
    "-d", "--device_cfg", type=Path, default=None, help="Path to device config file"
)
@click.option(
    "-s", "--scale_factor", type=int, default=2, help="Downsampling preview factor"
)
@click.option("-S", "--save", is_flag=True, help="Also save calibration dataset")
def calibrate(
    input_folder: Path,
    output_folder: Path,
    board_cfg: Path,
    device_cfg: Path,
    scale_factor: int,
    save: bool,
):
    acquire_corners = False
    if input_folder is None:
        acquire_corners = True
    if board_cfg is None:
        board_cfg = oakeye.data_folder / "board" / "chessboard.yml"
    if device_cfg is None:
        device_cfg = oakeye.data_folder / "device" / "device.yml"

    board: Board = Spook.create(XConfig(board_cfg))

    if acquire_corners:
        device_cfg = XConfig(device_cfg)
        device = OakDeviceFactory().create(device_cfg)
        keys = ["left", "center", "right"]
        acquirer = CornerAcquirer(
            DeviceAcquirer(device), keys, board, scale_factor=scale_factor
        )
        dataset = acquirer()
        device.close()
        if save:
            ext_map = {
                "left": "jpg",
                "center": "jpg",
                "right": "jpg",
                "depth": "png",
                "board": "yml",
                "device": "yml",
            }
            root_files = ["board", "device"]
            for d in dataset:
                d["board"] = board.serialize()
                d["device"] = device_cfg.to_dict()
            UnderfolderWriter(
                output_folder, root_files_keys=root_files, extensions_map=ext_map
            )(dataset)
    else:
        dataset = UnderfolderReader(input_folder)

    if len(dataset) < 1:
        raise ClickException(f"Input dataset must contain at least one sample")

    left = []
    center = []
    right = []
    for sample in dataset:
        left.append(sample["left"])
        center.append(sample["center"])
        right.append(sample["right"])
    calib = board.trinocular_calibration(left, center, right)

    output_folder.mkdir(parents=True, exist_ok=True)
    XConfig(plain_dict=calib).save_to(output_folder / "calibration.yml")


@click.command("acquire")
@click.option(
    "-o", "--output_folder", type=Path, default=None, help="Path to output folder"
)
@click.option(
    "-c", "--calibration", type=Path, default=None, help="Path to calibration file"
)
@click.option(
    "-d", "--device_cfg", type=Path, default=None, help="Path to device config file"
)
@click.option(
    "-s", "--scale_factor", type=int, default=2, help="Downsampling preview factor"
)
@click.option("--max_depth", type=int, default=1000, help="Max depth (mm)")
@click.option("--max_disparity", type=int, default=64, help="Max disparity")
def acquire(
    calibration: Path,
    device_cfg: Path,
    output_folder: Path,
    scale_factor: int,
    max_disparity: int,
    max_depth: int,
):
    if device_cfg is None:
        device_cfg = oakeye.data_folder / "device" / "device.yml"
    device_cfg = XConfig(device_cfg)
    device = OakDeviceFactory().create(device_cfg)
    keys = ["left", "center", "right", "depth", "center_left", "center_right"]
    acquirer = DeviceAcquirer(device)
    if calibration is not None:
        calib = XConfig(calibration)
        acquirer = RectifiedAcquirer(acquirer, calib)
        acquirer = DisparityAcquirer(acquirer, disp_diff=max_disparity)
    acquirer = GuiAcquirer(
        acquirer,
        keys,
        scale_factor=scale_factor,
        ranges={
            "center_left": [0, max_disparity],
            "center_right": [0, max_disparity],
            "depth": [0, max_depth],
        },
    )
    dataset = acquirer()
    device.close()
    if output_folder is not None:
        ext_map = {
            "left": "jpg",
            "^center$": "jpg",
            "right": "jpg",
            "depth": "png",
            "center_left": "png",
            "center_right": "png",
            "device": "yml",
        }
        root_files = ["device"]
        for d in dataset:
            d["device"] = device_cfg.to_dict()
        UnderfolderWriter(
            output_folder, root_files_keys=root_files, extensions_map=ext_map
        )(dataset)


trinocular.add_command(calibrate)
trinocular.add_command(acquire)

if __name__ == "__main__":
    trinocular()
