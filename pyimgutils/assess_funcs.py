import os

import matplotlib.pyplot as plt
import numpy as np
from skimage import io, img_as_float


def match_ax_size(axes_array, row, col, row_to_match, col_to_match):
    asp = (
        np.diff(axes_array[row, col].get_xlim())[0] 
        / np.diff(axes_array[row, col].get_ylim())[0]
    )
    asp /= np.abs(
        np.diff(axes_array[row_to_match, col_to_match].get_xlim())[0] 
        / np.diff(axes_array[row_to_match, col_to_match].get_ylim())[0]
    )

    return asp

def compare(
    img_dir_path, 
    img_num,
    processing_funcs=None,
    processing_funcs_kwargs_list=None,
    img_titles=None,
    show_hist=False,
    line_x=None, 
    line_y=None, 
    img_filetype='.tif', 
    fig_h=6,
    fig_w=None,
    colormap='viridis',
    tight_layout=True
):
    """Function for comparing images with different processing functions applied. Can show histograms an line profiles

    Args:
        img_dir_path ([type]): [description]
        img_num ([type]): [description]
        processing_funcs ([type], optional): [description]. Defaults to None.
        processing_funcs_kwargs_list ([type], optional): [description]. Defaults to None.
        img_titles ([type], optional): [description]. Defaults to None.
        show_hist (bool, optional): [description]. Defaults to False.
        line_x ([type], optional): [description]. Defaults to None.
        line_y ([type], optional): [description]. Defaults to None.
        img_filetype (str, optional): [description]. Defaults to '.tif'.
        fig_h (int, optional): [description]. Defaults to 6.
        fig_w ([type], optional): [description]. Defaults to None.
        tight_layout (bool, optional): [description]. Defaults to True.

    Raises:
        ValueError: [description]
        ValueError: [description]
        ValueError: [description]
        ValueError: [description]
    """
    img_dir_fns = os.listdir(img_dir_path)
    img_dir_fns.sort()
    imgs_only_fns = [fn for fn in img_dir_fns if fn.endswith(img_filetype)]
    img_path = os.path.join(img_dir_path, imgs_only_fns[img_num])
    img = io.imread(img_path)
        
    # processing_funcs and processing_funcs_kwargs_list are prepared to accept
    # multiple functions in a list, so if no values are passed, a list
    # must be created with a None item
    if processing_funcs is None:
        processing_funcs = [None]
    # If a non-list object is passed, a list with only that object is created
    elif not isinstance(processing_funcs, list):
        processing_funcs = [processing_funcs]

    # If no kwargs are passed for the processing functions, fill a list with 
    # None objects for each processing function in processing_funcs
    if processing_funcs_kwargs_list is None:
        processing_funcs_kwargs_list = [None] * len(processing_funcs)
    elif not isinstance(processing_funcs_kwargs_list, list):
        processing_funcs_kwargs_list = [processing_funcs_kwargs_list]

    if isinstance(img_titles, list):
        if len(img_titles) != len(processing_funcs):
            raise ValueError(
                'Must pass an image title for each processing function.'
            )
    elif img_titles is True:
        # Make a list with each item matching the name of the function in 
        # processing_funcs
        img_titles = [func.__name__ for func in processing_funcs]
    elif img_titles is not None:
        raise ValueError(
            'If passed, img_titles must be True or a list of titles correspond to the items in processing_funcs'
        )
        
    n_rows = 1
    if line_x is not None and line_y is not None:
        raise ValueError(
            'A value cannot be passed for line_x and line_y at the same time.'
        )
    elif line_x is not None or line_y is not None:
        n_rows += 1

    if show_hist:
        n_rows += 1

    n_cols = len(processing_funcs)
        
    if fig_h is not None and fig_w is None:
        # Following equation algebraically rearranged to get fig_s_in:
        # (fig_w / n_cols) / (fig_h / n_rows) = img.shape[1] / img.shape[0]
        fig_w = img.shape[1] / img.shape[0] * (fig_h / n_rows) * n_cols
    elif fig_w is not None and fig_h is None:
        fig_h = img.shape[0] / img.shape[1] * (fig_w / n_cols) * n_rows
    elif fig_h is None and fig_w is None:
        raise ValueError('Either fig_h or fig_w must be non-None to '
                         'calculate the other value.')
    elif fig_h is not None and fig_w is not None:
        print('Warning: passing fig_h and fig_w may result in distorted aspect '
              'ratio of image.')

    #--------------#
    # Format Plots #
    #--------------#
    
    # When there is only a single row passed to plt.sublots(), a 1D array is 
    # returned, but we need a 2D array so that the same code can apply 
    # when there are multiple rows. We will create a 2D array with only one row 
    # and fill that first row (row = 0) with the 1D array returned by subplots()
    if n_rows == 1:
        # Create an empty array with the shape (n_rows, n_cols)
        axes = np.full((n_rows, n_cols), None)
        fig, axes[0] = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
    else:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
        
    # Zip processing_funcs and processing_func_kwargs together so they are returned in sync with each other
    for col, (processing_func, processing_func_kwargs) in enumerate(
        zip(processing_funcs, processing_funcs_kwargs_list)
    ):
        
        if processing_func is None:
            img_to_plot = img
        elif processing_func_kwargs is not None:
            img_to_plot = processing_func(img, **processing_func_kwargs)
        else:
            img_to_plot = processing_func(img)
            
        axes[0, col].imshow(img_to_plot, cmap=colormap)
        axes[0, col].set_axis_off()
        if img_titles is not None:
            axes[0, col].set_title(img_titles[col])

        if show_hist:
            axes[-1, col].hist(
                img_to_plot.ravel(), 
                bins=256, 
                histtype='step', 
                color='black'
            )
            axes[-1, col].set_xlim(0, 1)
            axes[-1, col].ticklabel_format(
                axis='y', 
                style='scientific', 
                scilimits=(0, 0)
            )

            asp = match_ax_size(axes, -1, col, 0, col)
            axes[-1, col].set_aspect(asp)
    
        if line_y is not None:
            img_row = img_to_plot[line_y, :]
            axes[0, col].plot(
                np.arange(img_row.shape[0]), 
                np.full_like(img_row, line_y), 
                color='red', 
                linewidth=1
            )
            axes[1, col].plot(
                np.arange(img_row.shape[0]), 
                img_row, 
                color='red', 
                linewidth=1
            )
            axes[1, col].set_xlim(0, img_row.shape[0])

            asp = match_ax_size(axes, 1, col, 0, col)
            axes[1, col].set_aspect(asp)
    
    if tight_layout:
        plt.tight_layout()
    plt.show()

