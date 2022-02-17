from pathlib import Path
from choixe.configurations import XConfig
from choixe.spooks import Spook
import click
from click.exceptions import ClickException
from pipelime.sequences.writers.filesystem import UnderfolderWriter
from pipelime.sequences.readers.filesystem import UnderfolderReader
from pipelime.sequences.samples import SamplesSequence
import oakeye
from oakeye.device import OakDeviceFactory
from oakeye.board import Board
from oakeye.acquirer import (
    CornerAcquirer,
    DeviceAcquirer,
    GuiAcquirer,
    RectifiedAcquirer,
    DisparityAcquirer,
    UnderfolderAcquirer,
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
    "-r",
    "--rectification",
    type=Path,
    default=None,
    help="Path to calibration config file to calibrate with rectified images, works only with new acquired images",
)
@click.option(
    "-s", "--scale_factor", type=int, default=1, help="Downsampling preview factor"
)
@click.option("-S", "--save", is_flag=True, help="Also save calibration dataset")
def calibrate(
    input_folder: Path,
    output_folder: Path,
    board_cfg: Path,
    device_cfg: Path,
    rectification: Path,
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
        acquirer = DeviceAcquirer(device)
        if rectification is not None:
            calib = XConfig(filename=rectification)
            acquirer = RectifiedAcquirer(acquirer, calib)
        acquirer = CornerAcquirer(acquirer, keys, board, scale_factor=scale_factor)
        dataset = acquirer()
        device.close()
        if save:
            ext_map = {
                "left": "png",
                "center": "png",
                "right": "png",
                "depth": "png",
                "board": "yml",
                "device": "yml",
            }
            root_files = ["board", "device"]
            for d in dataset:
                d["board"] = board.serialize()
                d["device"] = device_cfg.to_dict()
            UnderfolderWriter(
                output_folder,
                root_files_keys=root_files,
                extensions_map=ext_map,
                num_workers=-1,
            )(SamplesSequence(dataset))
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
    "-s", "--scale_factor", type=int, default=1, help="Downsampling preview factor"
)
@click.option("--disparity", is_flag=True, help="Also acquire disparity")
@click.option("--max_depth", type=int, default=1000, help="Max depth (mm)")
@click.option("--max_disparity", type=int, default=64, help="Max disparity")
@click.option("--max_frames", type=int, default=-1, help="Max number of frames")
@click.option("--skip", type=int, default=1, help="Skip frames")
def acquire(
    calibration: Path,
    device_cfg: Path,
    output_folder: Path,
    scale_factor: int,
    disparity: bool,
    max_disparity: int,
    max_depth: int,
    max_frames: int,
    skip: int,
):
    if device_cfg is None:
        device_cfg = oakeye.data_folder / "device" / "device.yml"
    device_cfg = XConfig(device_cfg)
    ext_map = {
        "left": "png",
        "center": "png",
        "right": "png",
        "depth": "png",
        "disparityCL": "png",
        "disparityCR": "png",
        "device": "yml",
    }
    root_files = ["device"]
    writer = UnderfolderWriter(
        output_folder,
        root_files_keys=root_files,
        extensions_map=ext_map,
    )
    device_cfg.save_to(output_folder / "device.yml")
    device = OakDeviceFactory().create(device_cfg)
    keys = ["left", "center", "right", "depth", "disparityCL", "disparityCR"]
    acquirer = DeviceAcquirer(device)
    if calibration is not None:
        calib = XConfig(calibration)
        acquirer = RectifiedAcquirer(acquirer, calib)
    if disparity:
        acquirer = DisparityAcquirer(acquirer, disp_diff=max_disparity)
    acquirer = GuiAcquirer(
        acquirer,
        keys,
        scale_factor=scale_factor,
        ranges={
            "disparityCL": [0, max_disparity],
            "disparityCR": [0, max_disparity],
            "depth": [0, max_depth],
        },
        writer=writer,
    )
    dataset = acquirer(max_frames=max_frames, skip=skip)
    device.close()
    if dataset is not None:
        writer(dataset)


@click.command("rectify")
@click.option(
    "-i", "--input_folder", type=Path, required=True, help="Path to input folder"
)
@click.option(
    "-o", "--output_folder", type=Path, required=True, help="Path to output folder"
)
@click.option(
    "-c", "--calibration", type=Path, required=True, help="Path to calibration file"
)
def rectify(input_folder, output_folder, calibration):

    dataset = UnderfolderReader(input_folder)
    acquirer = UnderfolderAcquirer(dataset)
    calib = XConfig(filename=calibration)
    rect_keys = {
        "left": "leftrect",
        "center": "centerrect",
        "right": "rightrect",
    }
    acquirer = RectifiedAcquirer(
        acquirer,
        calib,
        rect_left_key=rect_keys["left"],
        rect_center_key=rect_keys["center"],
        rect_right_key=rect_keys["right"],
    )
    samples = acquirer()

    template = dataset.get_reader_template()
    ext_map = template.extensions_map
    ext_map.update(
        {
            rect_keys["left"]: "png",
            rect_keys["center"]: "png",
            rect_keys["right"]: "png",
        }
    )
    UnderfolderWriter(
        output_folder,
        root_files_keys=template.root_files_keys,
        extensions_map=ext_map,
        num_workers=-1,
    )(SamplesSequence(samples))


trinocular.add_command(calibrate)
trinocular.add_command(acquire)
trinocular.add_command(rectify)

if __name__ == "__main__":
    trinocular()
