import os

import matplotlib.pyplot as plt
import numpy as np
from skimage import io, img_as_float


def compare_processing(img_dir_path, 
                       img_n,
                       preprocessing_func=None,
                       preprocessing_func_kwargs=None,
                       processing_funcs=None,
                       processing_funcs_kwargs_list=None,
                       line_x=None, 
                       line_y=None, 
                       img_type='.tif', 
                       fig_h=6,
                       fig_w=None,
                       tight_layout=True
):
    """Function that compares processing routines by showing processed images beside each other in a figure.

    Args:
        img_dir_path ([type]): [description]
        img_n ([type]): [description]
        preprocessing_func ([type], optional): [description]. Defaults to None.
        preprocessing_func_kwargs ([type], optional): [description]. Defaults to None.
        processing_funcs (list, optional): [description]. Defaults to [].
        processing_funcs_kwargs_list (list, optional): [description]. Defaults to [].
        line_x ([type], optional): [description]. Defaults to None.
        line_y ([type], optional): [description]. Defaults to None.
        img_type (str, optional): [description]. Defaults to '.tif'.
        fig_y_in (int, optional): [description]. Defaults to 6.

    Raises:
        ValueError: [description]
    """
    
    img_dir_fns = os.listdir(img_dir_path)
    imgs_only_fns = [fn for fn in img_dir_fns if fn.endswith(img_type)]
    img_path = os.path.join(img_dir_path, imgs_only_fns[img_n])
    img = io.imread(img_path)
    
    # Apply preprocessing to img so that figure size can be calculated
    if preprocessing_func is not None and preprocessing_func_kwargs is not None:
        img = preprocessing_func(img, **preprocessing_func_kwargs)
    elif preprocessing_func is not None:
        img = preprocessing_func(img)
        
    # processing_funcs and processing_funcs_kwargs_list are prepared to accept
    # multiple functions in a list, so if no values are passed, an empty list
    # must be created
    if processing_funcs is None:
        processing_funcs = []
    # If a non-list object is passed, a list with only that object is created
    elif not isinstance(processing_funcs, list):
        processing_funcs = [processing_funcs]

    if processing_funcs_kwargs_list is None:
        processing_funcs_kwargs_list = []
    elif not isinstance(processing_funcs_kwargs_list, list):
        processing_funcs_kwargs_list = [processing_funcs_kwargs_list]

    # We iterate through processing_funcs to plot the images in the figure, 
    # but processing_func is applied to the preprocessed image. To show the 
    # preprocessed image, no other processing is applied so we add None
    processing_funcs.insert(0, None)
    processing_funcs_kwargs_list.insert(0, None)

    # At this point, processing_funcs and processing_funcs_kwargs_list
    # both have None as the 0 entry in the list, but there may not be 
    # kwargs passed for each element in processing_funcs, meaning 
    # processing_funcs could be longer than processing_funcs_kwargs_list. 
    # We go through a 'while' loop to continue filling 
    # processing_funcs_kwargs_list with None until sizes match.
    processing_func_i = 0
    while len(processing_funcs) > len(processing_funcs_kwargs_list):
        processing_funcs_kwargs_list.insert(processing_func_i, None)
        processing_func_i += 1
    n_cols = len(processing_funcs)
        
    if line_x is not None and line_y is not None:
        raise ValueError('A value cannot be passed for line_x and line_y at the same time.')
    elif line_x is not None or line_y is not None:
        n_rows = 3
    else:
        n_rows = 2
        
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
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
        
    if n_cols > 1:
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
                
            axes[0, col].imshow(img_to_plot)
            axes[0, col].set_axis_off()
            axes[-1, col].hist(img_to_plot.ravel(), bins=256, histtype='step', color='black')
            axes[-1, col].set_xlim(0, 1)
            axes[-1, col].ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))

            asp_rats = np.zeros_like(axes)
            asp_rats[-1, col] = np.diff(axes[-1, col].get_xlim())[0] / np.diff(axes[-1, col].get_ylim())[0]
            asp_rats[-1, col] /= np.abs(np.diff(axes[0, col].get_xlim())[0] / np.diff(axes[0, col].get_ylim())[0])
            axes[-1, col].set_aspect(asp_rats[-1, col])
        
            if line_y is not None:
                img_row = img_to_plot[line_y, :]
                axes[0, col].plot(np.arange(img_row.shape[0]), np.full_like(img_row, line_y), color='red', linewidth=1)
                axes[1, col].plot(np.arange(img_row.shape[0]), img_row, color='red', linewidth=1)
                axes[1, col].set_xlim(0, img_row.shape[0])

                asp_rats[1, col] = np.diff(axes[1, col].get_xlim())[0] / np.diff(axes[1, col].get_ylim())[0]
                asp_rats[1, col] /= np.abs(np.diff(axes[0, col].get_xlim())[0] / np.diff(axes[0, col].get_ylim())[0])
                axes[1, col].set_aspect(asp_rats[1, col])
    else:
        processing_func = processing_funcs[0]
        if processing_func is None:
            img_to_plot = img
        elif processing_func_kwargs is not None:
            img_to_plot = processing_func(img, **processing_func_kwargs)
        else:
            img_to_plot = processing_func(img)
            
        axes[0].imshow(img_to_plot)
        axes[0].set_axis_off()
        axes[-1].hist(img_to_plot.ravel(), bins=256, histtype='step', color='black')
        axes[-1].set_xlim(0, 1)
        axes[-1].ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))

        asp_rats = np.zeros_like(axes)
        asp_rats[-1] = np.diff(axes[-1].get_xlim())[0] / np.diff(axes[-1].get_ylim())[0]
        asp_rats[-1] /= np.abs(np.diff(axes[0].get_xlim())[0] / np.diff(axes[0].get_ylim())[0])
        axes[-1].set_aspect(asp_rats[-1])
    
        if line_y is not None:
            img_row = img_to_plot[line_y, :]
            axes[0].plot(np.arange(img_row.shape[0]), np.full_like(img_row, line_y), color='red', linewidth=1)
            axes[1].plot(np.arange(img_row.shape[0]), img_row, color='red', linewidth=1)
            axes[1].set_xlim(0, img_row.shape[0])

            asp_rats[1] = np.diff(axes[1].get_xlim())[0] / np.diff(axes[1].get_ylim())[0]
            asp_rats[1] /= np.abs(np.diff(axes[0].get_xlim())[0] / np.diff(axes[0].get_ylim())[0])
            axes[1].set_aspect(asp_rats[1])
    
    if tight_layout:
        plt.tight_layout()
    plt.show()