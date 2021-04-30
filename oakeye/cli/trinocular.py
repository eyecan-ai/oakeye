from pathlib import Path
from choixe.configurations import XConfig
from choixe.spooks import Spook
import click
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
)


@click.group("trinocular")
def trinocular():
    pass


@click.command("acquire")
@click.option(
    "-o", "--output_folder", type=Path, required=True, help="Path to output underfolder"
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
def acquire(
    output_folder: Path,
    board_cfg: Path,
    device_cfg: Path,
    scale_factor: float,
):
    if board_cfg is None:
        board_cfg = oakeye.data_folder / "board" / "chessboard.yml"
    if device_cfg is None:
        device_cfg = oakeye.data_folder / "device" / "device.yml"
    device_cfg = XConfig(device_cfg)
    device = OakDeviceFactory().create(device_cfg)
    board: Board = Spook.create(XConfig(board_cfg))
    keys = ["left", "center", "right"]
    acquirer = CornerAcquirer(
        DeviceAcquirer(device), keys, board, scale_factor=scale_factor
    )
    dataset = acquirer()
    device.close()
    extension = "jpg"
    ext_map = {
        "left": extension,
        "center": extension,
        "right": extension,
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


@click.command("calibrate")
@click.option(
    "-i", "--input_folder", type=Path, required=True, help="Path to input underfolder"
)
@click.option(
    "-o",
    "--output_file",
    type=Path,
    default=None,
    help="Path to output calibration file",
)
@click.option("-s", "--scale_factor", type=int, default=2, help="Downsampling factor")
def calibrate(input_folder: Path, output_file: Path, scale_factor: int):
    if output_file is None:
        output_file = input_folder / "calibration.yml"
    uf = UnderfolderReader(input_folder)
    board = Board.create(uf[0]["board"])
    left = []
    center = []
    right = []
    for sample in uf:
        left.append(sample["left"])
        center.append(sample["center"])
        right.append(sample["right"])
    calib = board.trinocular_calibration(left, center, right)
    XConfig(plain_dict=calib).save_to(output_file)


@click.command("rectify")
@click.option(
    "-c", "--calibration", type=Path, required=True, help="Path to calibration file"
)
@click.option(
    "-d", "--device_cfg", type=Path, default=None, help="Path to device config file"
)
@click.option(
    "-s", "--scale_factor", type=int, default=2, help="Downsampling preview factor"
)
def rectify(calibration: Path, device_cfg: Path, scale_factor: int):
    if device_cfg is None:
        device_cfg = oakeye.data_folder / "device" / "device.yml"
    device_cfg = XConfig(device_cfg)
    device = OakDeviceFactory().create(device_cfg)
    calib = XConfig(calibration)
    keys = ["left", "center", "right", "depth"]
    acquirer = GuiAcquirer(RectifiedAcquirer(DeviceAcquirer(device), calib), keys)
    dataset = acquirer()
    device.close()
    # extension = "jpg"
    # ext_map = {
    #     "left": extension,
    #     "center": extension,
    #     "right": extension,
    #     "board": "yml",
    #     "device": "yml",
    # }
    # root_files = ["board", "device"]
    # for d in dataset:
    #     d["board"] = board.serialize()
    #     d["device"] = device_cfg.to_dict()
    # UnderfolderWriter(
    #     output_folder, root_files_keys=root_files, extensions_map=ext_map
    # )(dataset)


trinocular.add_command(acquire)
trinocular.add_command(calibrate)
trinocular.add_command(rectify)

if __name__ == "__main__":
    trinocular()
