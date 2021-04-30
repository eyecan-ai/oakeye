======
oakeye
======


.. image:: https://img.shields.io/pypi/v/oakeye.svg
        :target: https://pypi.python.org/pypi/oakeye

.. image:: https://img.shields.io/travis/domef/oakeye.svg
        :target: https://travis-ci.com/domef/oakeye

.. image:: https://readthedocs.org/projects/oakeye/badge/?version=latest
        :target: https://oakeye.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status




Oak Device calibration and acquisition tools. 
https://opencv.org/opencv-ai-competition-2021/


* Free software: GNU General Public License v3

--------
Features
--------

* Trinocular Camera Calibration
* Dataset Acquisition with rectified images, disparity and sensor depth.

-----
Usage
-----

Calibration
-----------

Perform trinocular calibration with the following command:

.. code-block:: bash

        oakeye trinocular calibrate -o $OUTPUT_FOLDER [OPTIONS]


Where ``OPTIONS`` include:

- ``--input_folder PATH`` - Calibrate using previously saved images.
- ``--board_cfg PATH`` - The board configuration file.
- ``--device_cfg PATH`` - The device configuration file.
- ``--scale_factor INTEGER`` - Resize the preview of the images, defaults to ``2``.
- ``--save`` - Save also acquired images in $OUTPUT_FOLDER, defaults to ``false``.

It will generate a file named ``calibration.yml`` inside ``$OUTPUT_FOLDER``.

Acquisition
-----------

Acquire images from the sensor with the following command:

.. code-block:: bash

        oakeye trinocular acquire [OPTIONS]


Where ``OPTIONS`` include:

- ``--output_folder PATH`` - Save images in the specified folder.
- ``--calibration PATH`` - Calibration file computed with the ``calibration`` command. If specified it will rectify the images and compute the disparities.
- ``--device_cfg PATH`` - The device configuration file.
- ``--scale_factor INTEGER`` - Resize the preview of the images, defaults to ``2``.
- ``--max_depth INTEGER`` - Max value (mm) of the depth preview, defaults to ``1000``
- ``--max_dispaity INTEGER`` - Max value of the disparities preview, defaults to ``64``


-------
Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
