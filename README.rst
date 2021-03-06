======
Oakeye
======


.. .. image:: https://img.shields.io/pypi/v/oakeye.svg
..         :target: https://pypi.python.org/pypi/oakeye

.. .. image:: https://img.shields.io/travis/domef/oakeye.svg
..         :target: https://travis-ci.com/domef/oakeye

.. .. image:: https://readthedocs.org/projects/oakeye/badge/?version=latest
..         :target: https://oakeye.readthedocs.io/en/latest/?version=latest
..         :alt: Documentation Status




Oak Device calibration and acquisition tools. 
https://opencv.org/opencv-ai-competition-2021/


.. contents:: **What you can do?**


Trinocular Calibration
----------------------

Perform trinocular calibration with the following command:

.. code-block:: bash

        oakeye trinocular calibrate -o $OUTPUT_FOLDER [OPTIONS]


Where ``OPTIONS`` include:

- ``-i, --input_folder PATH`` - Calibrate using previously saved images. If not set
  images will be acquired live with an OpenCV GUI. Press 's' to acquire an image and 'q'
  to perform calibration and exit.
- ``-b, --board_cfg PATH`` - The board configuration file. If not set, a default configuration
  located in ``oakeye/data/board/chessboard.yml`` will be used.
- ``-d, --device_cfg PATH`` - The device configuration file. If not set, a default configuration
  located in ``oakeye/data/device/device.yml`` will be used.
- ``-r, --rectification PATH`` - A calibration file. If set, the calibration will be performed
  on rectified images. Works only with new acquired images (i.e. it doesn't work if ``-i`` is set).
- ``-s, --scale_factor INTEGER`` - Preview downscale factor, defaults to ``2``.
- ``-S, --save`` - Also save acquired images in $OUTPUT_FOLDER. If not set, only the calibration
  file will be saved.

Calibration results are stored in a file named ``calibration.yml`` inside ``$OUTPUT_FOLDER``.

All input and output datasets are stored using the **Underfolder** format.
See `Pipelime`_ for more info.

Time-synch Trinocular Live Acquisition
--------------------------------------

Acquire images from the sensor with the following command:

.. code-block:: bash

        oakeye trinocular acquire [OPTIONS]


Where ``OPTIONS`` include:

- ``--output_folder PATH`` - Save images in the specified folder. Press 's' to acquire an image and 'q'
  to save and exit.
- ``--calibration PATH`` - Calibration file computed with the ``calibration`` command. 
  If specified it will also rectify the images and compute the disparities.
- ``--device_cfg PATH`` - The device configuration file. 
- ``--scale_factor INTEGER`` - Preview downscale factor, defaults to ``2``.
- ``--max_depth INTEGER`` - Max value (mm) of the depth preview, defaults to ``1000``
- ``--max_disparity INTEGER`` - Max value of the disparities, defaults to ``64``

.. _`Pipelime`: https://github.com/eyecan-ai/pipelime

All input and output datasets are stored using the **Underfolder** format.
See `Pipelime`_ for more info.

Auto Labels Generation Pipeline
-------------------------------

`See the following example <https://github.com/eyecan-ai/oakeye/tree/2d9c82f24b816b85de68ad8208fb04408d98c9ee/examples/segmentation>`_

Issues
------

If you have problems (i.e. the application crash or the camera are not synchronized) install the branch develop of depthai (https://github.com/luxonis/depthai-python).
