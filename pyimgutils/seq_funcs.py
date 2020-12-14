import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, exposure, img_as_float, img_as_ubyte


def get_n_imgs(img_dir_path,
               img_filetype='.tif'):

    img_dir_list = os.listdir(img_dir_path)
    img_fn_list = [fn for fn in img_dir_list if fn.endswith(img_filetype)]
    img_fn_list.sort()

    return len(img_fn_list)

def get_img_dims(img_dir_path,
                 n,
                 img_filetype='.tif'):

    img_dir_list = os.listdir(img_dir_path)
    img_fn_list = [fn for fn in img_dir_list if fn.endswith(img_filetype)]
    img_fn_list.sort()

    img_n_path = os.path.join(img_dir_path, img_fn_list[n])
    img_n = io.imread(img_n_path)

    return img_n.shape[0], img_n.shape[1]

def seq_animate(img_dir_path, 
                img_range,
                img_step,
                processing_func=None,
                show_img_num=False,
                anim_type='gif',
                img_filetype='.tif',
                jupyter=True,
                figsize=(6, 6),
                colormap='viridis',
                save_gif_path=None, 
                save_dir_path=None,
                anim_fps=10):
    """Show a sequence of images from a directory. Can be used to generate an animation from the images.

    Args:
        img_dir_path (str): Path to directory containing images.
        img_range (2-tuple): Start and end for image sequence.
        img_step (int): Step size for moving between range defined by img_range.
        processing_func (function, optional): Optional processing function that will be applied to each image. Defaults to None.
        show_img_num (bool, optional): How to show image number. Defaults to False. 
        anim_type (str, optional): Pass 'gif' to save animation as 'gif', or 'mp4' to save animation as a video file. Saving as video file requires can greatly reduce file size (over an order of magnitude in some cases) by allowing for copmressing in the time dimension, but requires movie writer software FFMpeg. See README.md for install details. Defaults to 'gif'.
        img_filetype (str, optional): The type of images in directory at img_dir_path. Defaults to '.tif'.
        jupyter (bool, optional): If using a Jupyter Notebook, jupyter=True to properly show the matplotlib plots. Defaults to True.
        figsize (2-tuple, optional): Figure size to be shown in inches. Defaults to (6, 6)
        colormap (str, optional): Colormap to apply to shown images. Does not save in animation. Defaults to 'viridis'.
        save_gif_path (str, optional): Path to save location of animated gif. Filename and .gif file extension must be included or ValueError raised. Overides save_dir_path. Defaults to None.
        save_dir_path (str, optional): Path to save directory of animated gif. If provided instead of save_gif_path, filename will be generated based on range and step. save_gif_path overides this. Defaults to None.
        anim_fps (int, optional): Framerate of saved animation in frames per second. Defaults to 10.

    Raises:
        ValueError: Raises error when save_gif_path not passed with a .gif file extension.
    """
    if save_gif_path is not None and not save_gif_path.endswith('.gif'):
        raise ValueError('Save path must include filename with .gif extension.')

    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(
        left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

    img_dir_list = os.listdir(img_dir_path)
    img_fn_list = [fn for fn in img_dir_list if fn.endswith(img_filetype)]
    img_fn_list.sort()
        
    img_axes = []
    fn_prefix = 'raw'

    for n in np.arange(img_range[0], img_range[1], img_step):

        img_n_path = os.path.join(img_dir_path, img_fn_list[n])
        img_n = io.imread(img_n_path)
        img_n = img_as_float(img_n)

        if processing_func is not None:
            img_n = processing_func(img_n)
            fn_prefix = processing_func.__name__

        img_n_ax = ax.imshow(img_n, aspect='auto', animated=True)
        objs_to_anim = [img_n_ax]
        ax.set_axis_off()
        if show_img_num:
            img_num = ax.annotate(f'Image {n}/{len(img_dir_list)}', 
                                  (25, 50),
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
        fn = (f'{fn_prefix}_{img_range[0]}-{img_range[1]}-{img_step}'
              f'_{anim_fps}fps.{anim_type}')
        save_path = os.path.join(save_dir_path, fn)
        
    if save_path is not None:
        img_anim.save(save_path, writer=writer)
        print(f'Animation saved: {save_path}')

    if jupyter:
        plt.close()
        return img_anim.to_jshtml()
    else:
        plt.show()

def test():
    seq_animate(r'C:\Users\gusb\Research\APS_al-ag\Data\2014-08_APS\034_Al70Ag30_200',
                (2830, 2920),
                2,
                show_img_num=True,
                jupyter=False)


if __name__ == '__main__':
    test()

