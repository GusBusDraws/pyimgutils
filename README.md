# **pyimgutils**
Package containing general image processing modules used across my projects.

#

Created: 2020-December-13

C. Gus Becker

chbecker@mines.edu

cgusbecker@gmail.com

720-363-3626

#

## **Install Instructions**
pyimgutils is not currently uploaded to PyPI.org, but can still be installed via pip using developer mode. The steps to install pyimgutils for use in a local project are as follows:

1. Download the pyimgutils directory and all of its contents to your local machine.
2. Load your Python 3 environment (note: pyimgutils was tested with the Anaconda distribution of Python 3, so Anaconda is recommended for using pyimgutils. It is also recommended to install pyimgutils into an Anaconda virtual environment.)
3. In the terminal window (or Anaconda Prompt if running Anaconda on Windows) with your desired environment activated, run the command `python -m pip install -e path\to\pyimgutils` where `path\to\pyimgutils` is the absolute path to the download location of the pyimgutils package directory.
4. Test your install by importing the package into a `.py` file with `import pyimgutils` and running the script with your Python environment into which you have just installed pyimgutils.

#

## **Modules**

### ***proc_funcs.py***
Module containing functions for performing image processing routines. These functions can be imported into a Jupyter Notebook and combined in a separate function that and passed to an image sequence function so that the whole routine can be applied to each image.

### ***seq_funcs.py***
Module containing functions for showing sequences of images from a directory of images. Functions have options for passing processing functions as well as a save path for an animated gif of the image sequence.

**Note:** The function `seq_animate()` has keyword argument `anim_type` that defaults to `'gif'`, but can be changed to `'mp4'` to save animations in a video format which can greatly reduce file size (over an order of magnitude in some cases) by allowing for copmression in the itme dimension. This requires the movie writer software FFMpeg, which is not included in the Anaconda distribution of Python by default but can be added by opening an Anaconda Prompt window on Windows 10 (terminal in Linux and MacOS) and running the following command:

`conda install -c conda-forge ffmpeg`