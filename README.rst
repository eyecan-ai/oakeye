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
* Dataset Acquisition with rectified images, depth and disparity.

-----
Usage
-----

Calibration
-----------

Perform trinocular calibration with the following command:

.. code-block:: bash

        oakeye trinocular calibrate -o $OUTPUT_FOLDER

This will generate a file named ``calibration.yml`` inside the specified output folder

Acquisition
-----------

Acquire images from the sensor with the following command:

.. code-block:: bash

        oakeye trinocular acquire



-------
Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
