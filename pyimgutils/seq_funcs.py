import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, exposure, img_as_float, img_as_ubyte


def is_loaded():
    return True

def animate(img_dir_path, 
            img_range,
            img_step=1,
            anim_fps=10,
            processing_func=None,
            processing_func_kwargs=None,
            exp_name=None,
            save_dir_path=None,
            anim_type='gif',
            fig_h=6,
            fig_w=None,
            img_asp_setting='auto',
            show_img_num=False,
            img_num_loc=(25, 50),
            colormap='viridis',
            img_filetype='.tif',
            jupyter=True,
            save_gif_path=None):

    """Show a sequence of images from a directory. Can be used to generate an animation from the images.

    Args:
        img_dir_path (str): Path to directory containing images.
        img_range (2-tuple): Start and end for image sequence.
        img_step (int, optional): Step size for moving between range defined by img_range. Defaults to 1.
        anim_fps (int, optional): Framerate of saved animation in frames per second. Defaults to 10.
        processing_func (function, optional): Optional processing function that will be applied to each image. Defaults to None.
        processing_func_kwargs (dict, optional): Optional keyword arguments that will be passed to the processing function (processing_func). Defaults to None.
        exp_name (str, optional): Required for auto-generating filenames with 'save_dir_path'. If a path to a directory to save an animation is given, exp_name must also be given or else a ValueError is raised. Defaults to None.
        save_dir_path (str, optional): Path to save directory of animated gif. If provided instead of save_gif_path, filename will be generated based on range and step. save_gif_path overides this. Defaults to None.
        anim_type (str, optional): Pass 'gif' to save animation as 'gif', or 'mp4' to save animation as a video file. Saving as video file requires can greatly reduce file size (over an order of magnitude in some cases) by allowing for copmressing in the time dimension, but requires movie writer software FFMpeg. See README.md for install details. Defaults to 'gif'.
        fig_h (float, optional): Figure height to be shown in inches. If None is passed, will be calculated from fig_w. Defaults to 6.
        fig_w (float, optional): Figure width to be shown in inches. If None is passed, will be calculated from fig_h. Defaults to None.
        img_aspect (str, optional): Determines how to handle the aspect ratio of the image. To match the figsize, pass 'auto'. For maintaining the actual aspect ratio, pass None. Defaults to 'auto'.
        show_img_num (bool, optional): How to show image number. Defaults to False.
        img_num_loc (2-tuple, optional): Pixel location for placing bottom left corner of img_num. Defaults to (25, 50).
        colormap (str, optional): Colormap to apply to shown images. Does not save in animation. Defaults to 'viridis'.
        img_filetype (str, optional): The type of images in directory at img_dir_path. Defaults to '.tif'.
        jupyter (bool, optional): If using a Jupyter Notebook, jupyter=True to properly show the matplotlib plots. Defaults to True.
        save_gif_path (str, optional): Path to save location of animated gif. Filename and .gif file extension must be included or ValueError raised. Overides save_dir_path. Defaults to None.

    Raises:
        ValueError: Raised when fig_h and fig_w are both None. (one should be calculated from the other.)
        ValueError: Raises error when save_gif_path not passed with a .gif file extension.
        ValueError: Raises error when exp_name not passed with save_dir_path.
    """

    if save_gif_path is not None and not save_gif_path.endswith('.gif'):
        raise ValueError('Save path must include filename with .gif extension.')

    img_dir_list = os.listdir(img_dir_path)
    img_fn_list = [fn for fn in img_dir_list if fn.endswith(img_filetype)]
    img_fn_list.sort()
    
    # Get first image and perform any processing to determine image aspect 
    # ratio which will determine figure size
    img_0_path = os.path.join(img_dir_path, img_fn_list[0])
    img_0 = io.imread(img_0_path)
    if processing_func is not None and processing_func_kwargs is not None:
        img_0 = processing_func(img_0, **processing_func_kwargs)
    elif processing_func is not None:
        img_0 = processing_func(img_0)
    img_asp = img_0.shape[0]/img_0.shape[1]
    
    if fig_h is not None and fig_w is None:
        fig_w = fig_h/img_asp
    elif fig_w is not None and fig_h is None:
        fig_h = fig_w*img_asp
    elif fig_h is None and fig_w is None:
        raise ValueError('Either fig_h or fig_w must be non-None to '
                         'calculate the other value.')
    elif fig_h is not None and fig_w is not None:
        print('Warning: passing fig_h and fig_w may result in distorted aspect '
              'ratio of image.')

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.subplots_adjust(
        left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

    img_axes = []
    # Set the filename prefixe to 'raw' to show that no additional processing 
    # functions were applied. If processing function is applied, fn_prefix will
    # be changed.
    fn_prefix = 'raw'

    for n in np.arange(img_range[0], img_range[1], img_step):

        img_n_path = os.path.join(img_dir_path, img_fn_list[n])
        img_n = io.imread(img_n_path)
        img_n = img_as_float(img_n)

        if processing_func is not None:
            if processing_func_kwargs is not None:
                img_n = processing_func(img_n, **processing_func_kwargs)
            else:
                img_n = processing_func(img_n)
            # Change filename prefix to match the name of the processing 
            # functions applied
            fn_prefix = processing_func.__name__

        img_n_ax = ax.imshow(
            img_n, aspect=img_asp_setting, cmap=colormap, animated=True)
        objs_to_anim = [img_n_ax]
        ax.set_axis_off()
        if show_img_num:
            img_num = ax.annotate(f'Image {n}/{len(img_dir_list)}', 
                                  img_num_loc,
                                  color='white',
                                  fontsize='medium')
            objs_to_anim.append(img_num)
        img_axes.append(objs_to_anim)
    
    img_anim = animation.ArtistAnimation(fig, img_axes, interval=1000/anim_fps, 
                                         blit=True)

    if anim_type == 'gif':
        writer = animation.PillowWriter(fps=anim_fps)
    elif anim_type == 'mp4':
        writer = animation.FFMpegWriter(fps=anim_fps)

    save_path = None
    if save_gif_path is not None:
        save_path = save_gif_path
    elif save_dir_path is not None:
        if exp_name is not None:
            fn = (f'{exp_name}_{fn_prefix}_{img_range[0]}-{img_range[1]}'
                  f'-{img_step}_{anim_fps}fps.{anim_type}')
            save_path = os.path.join(save_dir_path, fn)
        else:
            raise ValueError(
                'Value for exp_name must be passed with save_dir_path')
        
    if save_path is not None:
        img_anim.save(save_path, writer=writer)
        print(f'Animation saved: {save_path}')

    if jupyter:
        plt.close()
        return img_anim.to_jshtml()
    else:
        plt.show()


if __name__ == '__main__':
    test()

