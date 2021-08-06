# Example segmentation script

Automatically segment object lying on a plane, given:
- An RGB image
- Disparity map
- Calibrated camera

The example script iterates on the inputs images and shows:
- The reconstructed point cloud
- The instance segmentation masks and their bounding box.

Press `Q` to close the point cloud visualization, press any key to close the segmentation masks visualization.

## Installation

To install the required packages, run the following commands. 

```bash
cd examples/segmentation
pip install -r requirements.txt
```

**Note**: make sure to use the `requirements.txt` file contained within the `examples/segmentation` directory, not the one located at oakeye root directory.

## Usage

To execute the example, run the following command:

```bash
python segment.py --images example_data/images \
                  --disparities example_data/disparities \ 
                  --calibration example_data/calibration.yml
```

You can see a list of optional parameters and a brief explanation with:

```bash
python segment.py --help
```