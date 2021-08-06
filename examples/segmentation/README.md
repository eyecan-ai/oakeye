# Auto Labels Generation pipeline

Automatically segment objects lying on a plane in order to generate *BoundingBoxes* and *Masks* labels, given:
- An RGB image
- Disparity map
- Calibrated camera

The example script explores a target folder (`data_folder`) searching for:
- `image.png`
- `disparity.png`
- `calibration.yml`

The visual output will be:
- The reconstructed point cloud
- The main plane segmentation within the point cloud
- The instance segmentation masks and bounding boxes of extracted objects.

## Installation

To install the required packages, run the following commands. 

```bash
pip install -r requirements.txt
```

## Usage

To execute the pipeline on samples data, run one of the following command:

```bash
python generate_labels.py --data_folder sample_data/000

python generate_labels.py --data_folder sample_data/001

...
```

You can launch the pipeline on your custom data following the sample folders content structure.


You can see a list of optional parameters and a brief explanation with:

```bash
python segment.py --help
```