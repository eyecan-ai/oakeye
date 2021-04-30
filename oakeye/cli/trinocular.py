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
)


@click.group("trinocular")
def trinocular():
    pass


@click.command("calibrate")
@click.option(
    "-i", "--input_folder", type=Path, default=None, help="Path to input folder"
)
@click.option(
    "-o", "--output_folder", type=Path, default=None, help="Path to output folder"
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
        input_folder = Path("/", "tmp", "oakeye", "calibration")
        acquire_corners = True
    if output_folder is None:
        output_folder = input_folder
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
            extension = "jpg"
            ext_map = {
                "left": extension,
                "center": extension,
                "right": extension,
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


@click.command("rectify")
@click.option(
    "-o", "--output_folder", type=Path, required=True, help="Path to output folder"
)
@click.option(
    "-c", "--calibration", type=Path, default=None, help="Path to calibration file"
)
@click.option(
    "-d", "--device_cfg", type=Path, default=None, help="Path to device config file"
)
def rectify(calibration: Path, device_cfg: Path, output_folder: Path):
    if device_cfg is None:
        device_cfg = oakeye.data_folder / "device" / "device.yml"
    device_cfg = XConfig(device_cfg)
    device = OakDeviceFactory().create(device_cfg)
    keys = ["left", "center", "right", "depth"]
    acquirer = DeviceAcquirer(device)
    if calibration is not None:
        calib = XConfig(calibration)
        acquirer = RectifiedAcquirer(acquirer, calib)
    acquirer = GuiAcquirer(acquirer, keys)
    dataset = acquirer()
    device.close()
    extension = "jpg"
    ext_map = {
        "left": extension,
        "center": extension,
        "right": extension,
        "depth": "png",
        "device": "yml",
    }
    root_files = ["device"]
    for d in dataset:
        d["device"] = device_cfg.to_dict()
    UnderfolderWriter(
        output_folder, root_files_keys=root_files, extensions_map=ext_map
    )(dataset)


# trinocular.add_command(store_dataset)
trinocular.add_command(calibrate)
trinocular.add_command(rectify)

if __name__ == "__main__":
    trinocular()