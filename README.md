# **pyimgutils**
Package containing general image processing modules used across my projects.

C. Gus Becker

chbecker@mines.edu

cgusbecker@gmail.com

720-363-3626

Created: 2020-December-13

#

# **Install Instructions**
pyimgutils is not currently uploaded to PyPI.org, but can still be installed via pip using developer mode. The steps to install pyimgutils for use in a local project are as follows:

1. Download the pyimgutils directory and all of its contents to your local machine.
2. Load your Python 3 environment (note: pyimgutils was tested with the Anaconda distribution of Python 3, so Anaconda is recommended for using pyimgutils. It is also recommended to install pyimgutils into an Anaconda virtual environment.)
3. In the terminal window (or Anaconda Prompt if running Anaconda on Windows) with your desired environment activated, run the command `python -m pip install -e path\to\pyimgutils` where `path\to\pyimgutils` is the absolute path to the download location of the pyimgutils package directory.
4. Test your install by importing the package into a `.py` file with the line `import pyimgutils` and running the script with your Python environment into which you have just installed pyimgutils.

# **Modules**

## ***seq_funcs.py***
#
Module containing functions for showing sequences of images from a directory of images. Functions have options for passing processing functions as well as a save path for an animated gif of the image sequence.

### `animate()`

Function to animate images in a directory when the path to that directory is passed. Animations saved as GIFs, but can be saved as MP4s for lower file sizes if the correct movie writer software is installed (see below).

*Arguments:*

- img_dir_path (str): Path to directory containing images.
- img_range (2-tuple): Start and end for image sequence.
- img_step (int): Step size for moving between range defined by img_range.
- processing_func (function, optional): Optional processing function that will be applied to each image. Defaults to None.
- show_img_num (bool, optional): How to show image number. Defaults to False. 
- anim_type (str, optional): Pass 'gif' to save animation as 'gif', or 'mp4' to save animation as a video file. Defaults to 'gif'. Saving as video file requires can greatly reduce file size (over an order of magnitude in some cases) by allowing for compressing in the time dimension, but requires movie writer software FFMpeg. FFMpeg is not included in the Anaconda distribution of Python by default but can be added by opening an Anaconda Prompt window on Windows 10 (terminal in Linux and MacOS) and running the following command:
`conda install -c conda-forge ffmpeg`
- img_filetype (str, optional): The type of images in directory at img_dir_path. Defaults to '.tif'.
- jupyter (bool, optional): If using a Jupyter Notebook, jupyter=True to properly show the matplotlib plots. Defaults to True.
- fig_h (float, optional): Figure height to be shown in inches. If None is passed, will be calculated from fig_w. Defaults to 6.
- fig_w (float, optional): Figure width to be shown in inches. If None is passed, will be calculated from fig_h. Defaults to None.
- colormap (str, optional): Colormap to apply to shown images. Does not save in animation. Defaults to 'viridis'.
- save_gif_path (str, optional): Path to save location of animated gif. Filename and .gif file extension must be included or ValueError raised. Overides save_dir_path. Defaults to None. Raises a ValueError when save_gif_path not passed with a .gif file extension.
- exp_name (str, optional): Required for auto-generating filenames with 'save_dir_path'. If a path to a directory to save an animation is given, exp_name must also be given or else a ValueError is raised. Defaults to None.
- save_dir_path (str, optional): Path to save directory of animated gif. If provided instead of save_gif_path, filename will be generated based on range and step. save_gif_path overides this. Defaults to None.
- anim_fps (int, optional): Framerate of saved animation in frames per second. Defaults to 10.

## ***proc_funcs.py***
#
Module containing functions for performing image processing routines. These functions can be imported into a Jupyter Notebook and combined in a separate function that and passed to an image sequence function so that the whole routine can be applied to each image.

### `subtract_img()`

Subtract img_to_subtract from img. This was developed to subtract an all liquid image from other images in a solidification dataset to increase contrast of solid regions.

*Arguments:*

- img (np.ndarray): Image that img_to_subtract will be subtracted from. 
- img_to_subtract (np.ndarray): Image that will be subtrated from img.

*Returns:*

- np.ndarray: Image that is result of subtraction.

### `crop_to_window()`

Function to crop image to a lighter region in the middle of the full image. Developed to be used in AET experiments featuring small region of interest in larger field-of-view due to the fact that APS experimental setup was used. 

*Arguments:*

- img (np.ndarray): Image to crop.

*Returns:* 

- np.ndarray: Cropped image that is result of cropping the full image to the window.

### `v_fft_filter(img)`

Function for performing a vertically oriented Fast Fourier Transform (FFT) filtering on an image by removing vertical frequencies in the frequency space image and inverse transforming the result back to the spatial domain. Developed to be used with AET experiments: artifacts from imaging consist of horizontal lines/bands that are repeated vertically throughout the image (once cropped and rotated). Since bands are repeating vertically, they show in the frequency domain as a vertical band that can be removed/filtered out before transforming back to the spatial domain.

*Arguments:*

- img (np.ndarray): Image that is to be transformed into the frequency domain and filtered.

*Returns:*

- np.ndarray: Image that is result of vertical FFT filtering.