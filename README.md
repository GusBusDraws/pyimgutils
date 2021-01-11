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

### ***seq_funcs.py***
Module containing functions for showing sequences of images from a directory of images. Functions have options for passing processing functions as well as a save path for an animated gif of the image sequence.

**get_n_imgs()**

**get_img_dims()**

**animate()**

Function to animate images in a directory when the path to that directory is passed. Animations saved as GIFs, but can be saved as MP4s for lower file sizes if the correct movie writer software is installed (see below).

- img_dir_path (str): Path to directory containing images.

- img_range (2-tuple): Start and end for image sequence.

- img_step (int): Step size for moving between range defined by img_range.

- processing_func (function, optional): Optional processing function that will be applied to each image. Defaults to None.

- show_img_num (bool, optional): How to show image number. Defaults to False. 

- anim_type (str, optional): Pass 'gif' to save animation as 'gif', or 'mp4' to save animation as a video file. Defaults to 'gif'. Saving as video file requires can greatly reduce file size (over an order of magnitude in some cases) by allowing for compressing in the time dimension, but requires movie writer software FFMpeg. FFMpeg is not included in the Anaconda distribution of Python by default but can be added by opening an Anaconda Prompt window on Windows 10 (terminal in Linux and MacOS) and running the following command:
`conda install -c conda-forge ffmpeg`

- img_filetype (str, optional): The type of images in directory at img_dir_path. Defaults to '.tif'.

- jupyter (bool, optional): If using a Jupyter Notebook, jupyter=True to properly show the matplotlib plots. Defaults to True.

- figsize (2-tuple, optional): Figure size to be shown in inches. Defaults to (6, 6)

- colormap (str, optional): Colormap to apply to shown images. Does not save in animation. Defaults to 'viridis'.

- save_gif_path (str, optional): Path to save location of animated gif. Filename and .gif file extension must be included or ValueError raised. Overides save_dir_path. Defaults to None. Raises a ValueError when save_gif_path not passed with a .gif file extension.

- exp_name (str, optional): Required for auto-generating filenames with 'save_dir_path'. If a path to a directory to save an animation is given, exp_name must also be given or else a ValueError is raised. Defaults to None.

- save_dir_path (str, optional): Path to save directory of animated gif. If provided instead of save_gif_path, filename will be generated based on range and step. save_gif_path overides this. Defaults to None.

- anim_fps (int, optional): Framerate of saved animation in frames per second. Defaults to 10.

### ***proc_funcs.py***
Module containing functions for performing image processing routines. These functions can be imported into a Jupyter Notebook and combined in a separate function that and passed to an image sequence function so that the whole routine can be applied to each image.
