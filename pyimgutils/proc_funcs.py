import os

import matplotlib.pyplot as plt
import numpy as np
from skimage import (
    filters, 
    img_as_float, 
    io,
    measure,
    morphology, 
    segmentation, 
    transform, 
    util
)


def avg_imgs(
    img_dir_path, 
    img_ns,
    return_float=True,
    preprocessing_func=None,
    img_filetype='.tif',
    **kwargs
):
    """Function for averaging multiple images together.

    Args:
        img_dir_path (str): Path to directory containing images.
        img_ns (iterable): An iterable (list, range, array, etc.) that represents the image numbers of the images to average together.
        return_float (bool, optional): When true, cinverts the images to floats before averaging. Defaults to True.
        preprocessing_func (function, optional): Preprocessing function to apply to images before averaging. Must take an image as the first argument and also return an image. Defaults to None.
        img_filetype (str, optional): Filetype of images located in img_dir_path. Defaults to '.tif'.
        **kwargs: Keyword arguments that are passed to preprocessing_func if provided.

    Returns:
        np.ndarray: Image where each pixel value has been averaged with other images in the same sequence.
    """
    # Build sorted list of images in img_dir_path
    img_dir_list = os.listdir(img_dir_path)
    img_fn_list = [fn for fn in img_dir_list if fn.endswith(img_filetype)]
    img_fn_list.sort()

    # Create empty list that will be filled with images to form 3D image 
    # (3D array consisting of stacked 2D images in the 0th axis)
    img_3d = []
    
    for img_n in img_ns:
        # Load each image
        img_fn = img_fn_list[img_n]
        img_path = os.path.join(img_dir_path, img_fn)
        img = io.imread(img_path)
        # Perform any preprocessing steps
        if preprocessing_func is not None:
            img = preprocessing_func(img, **kwargs)
        # Convert to float
        if return_float:
            img = img_as_float(img)
        # Append to list
        img_3d.append(img)
    
    # Convert list to 3D array
    img_3d = np.array(img_3d)
    # Average slices (averages each pixel value in the same position of all stacked images)
    img_mean = np.mean(img_3d, axis=0)
    
    return img_mean

def subtract_img(
    img, 
    img_to_sub=None, 
    img_dir_path=None, 
    img_n=None,
    preprocessing_func=None,
    preprocessing_func_kwargs=None,
    img_filetype='.tif'
):
    """Subtract img_to_sub from img.

    Args:
        img (np.ndarray): Image that img_to_sub will be subtracted from. 
        img_to_sub (np.ndarray): Image that will be subtrated from img.

    Returns:
        np.ndarray: Image that is result of subtraction.
    """

    img = img_as_float(img)

    if img_to_sub is not None:
        img_to_sub = img_as_float(img_to_sub)
    elif img_dir_path is not None:
        img_dir_list = os.listdir(img_dir_path)
        img_fn_list = [fn for fn in img_dir_list if fn.endswith(img_filetype)]
        img_fn_list.sort()
        img_to_sub_fn = img_fn_list[img_n]
        img_to_sub_path = os.path.join(img_dir_path, img_to_sub_fn)
        img_to_sub = io.imread(img_to_sub_path)
        img_to_sub = img_as_float(img_to_sub)
    else:
        raise ValueError('None passed to img_to_sub and img_dir_path.'
                         ' Non-None valuemust be passed to one or the other.')

    img_diff = (
        img[
            0 : min(img.shape[0], img_to_sub.shape[0]),
            0 : min(img.shape[1], img_to_sub.shape[1])
        ]
        - img_to_sub[
            0 : min(img.shape[0], img_to_sub.shape[0]),
            0 : min(img.shape[1], img_to_sub.shape[1])
        ]
    )

    return img_diff

def crop_to_window(img, rotate_deg=-90, region_area=100):
    """Funciton to crop image to a lighter region in the middle of the full image.

    Args:
        img (np.ndarray): Image to crop.
        rotate_deg (int, optional): Amount to rotate cropped region by. Defaults to -90.

    Returns:
        np.ndarray: Cropped image that is result of cropping the full image to the window.

    Raises:
        ValueError: Raised if there is not one large area
        ValueError: Raised if there is more than one large area
    """

    # Threshold image and select area not touching border
    thresh = filters.threshold_otsu(img)
    img_binary = morphology.closing(img > thresh, morphology.square(3))
    img_cleared = segmentation.clear_border(img_binary)

    # Label the remaining regions and select regions larger than 100 pixels 
    img_label = measure.label(img_cleared)
    regions = [region for region in measure.regionprops(img_label)
               if region.area > region_area]
    if len(regions) < 1:
        raise ValueError('No regions match assigned rules')
    elif len(regions) > 1:
        raise ValueError(f'One region expected, found {len(regions)} regions '
                         f'greater than {region_area}.')
    else:
        minr, minc, maxr, maxc = regions[0].bbox

        img_crop = img[minr:maxr, minc:maxc]

    # Rotate and trim remaining region
    img_rotate = transform.rotate(img_crop, rotate_deg, resize=True)
    img_trim = util.crop(img_rotate, 10)

    return img_trim

def v_fft_filter(
    img,
    show=False,
    colormap='viridis',
    **plt_kwargs
):
    """Function for performing a vertically oriented Fast Fourier Transform (FFT) filtering on an image by removing vertical frequencies in the frequency space image and inverse transforming the result back to the spatial domain.

    Args:
        img (np.ndarray): Image that is to be transformed into the frequency domain and filtered.

    Returns:
        np.ndarray: Image that is result of vertical FFT filtering.
    """
    # Use the two dimensional Fast Fourier Transform (FFT) 
    # to produce the Discrete Fourier Transform (DFT) of the image 
    f = np.fft.fft2(img)
    # Shift result so zero frequency component is at center
    f_shift = np.fft.fftshift(f)
    f_mag_spec = np.log(np.abs(f_shift))
    f_mag_spec_norm = ((f_mag_spec - f_mag_spec.min()) 
                       / (f_mag_spec.max() - f_mag_spec.min()))

    # Make a boolean array to mask the DFT
    f_mask = np.zeros_like(f_mag_spec, dtype='bool')

    n_rows = img.shape[0]
    v_mask = np.array([[0, n_rows//2],
                    [n_rows//2 + 1, n_rows]])
    # Go through each row of v_mask and apply the mask components
    # to the DFT mask (f_mask)
    for row in range(v_mask.shape[0]):
        f_mask[v_mask[row, 0]:v_mask[row, 1], f_mask.shape[1]//2] = 1

    # Set the values of the DFT that match the True values of the 
    # boolean mask to the mean value of the DFT
    f_mag_spec_masked = f_mag_spec_norm.copy()
    f_mag_spec_masked[f_mask] = np.mean(f_mag_spec_norm)

    # Changing the magnitude spectrum (f_mag_spec) of the DFT is only for 
    # visualization. In order to change the DFT itself, we must change the 
    # shifted frequency component array (f_shift). Since we took the abosulte 
    # value of this array for the visualization, we had to set the masked 
    # values to the mean, but these values are centered around zero, so that is 
    # where we will set the masked values.
    # print(np.mean(f_shift))
    f_shift[f_mask] = 0

    # Inverse shift the zero frequency components back to their natural position
    f_ishift = np.fft.ifftshift(f_shift)
    # Inverse FFT back to the spatial domain
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    if show:
        fig, axes = plt.subplots(1, 5, **plt_kwargs)
        ax = axes.ravel()
        ax[0].imshow(img, cmap=colormap)
        ax[0].set_title('Image In')
        ax[1].imshow(f_mag_spec_norm, cmap=colormap)
        ax[1].set_title('FFT of Image In')
        ax[2].imshow(f_mask, cmap=colormap, interpolation='nearest')
        ax[2].set_title('FFT Mask')
        ax[3].imshow(f_mag_spec_masked, cmap=colormap)
        ax[3].set_title('Masked FFT')
        ax[4].imshow(img_back, cmap=colormap)
        ax[4].set_title('Image from Inverse FFT')

        for a in ax:
            a.set_axis_off()

        plt.tight_layout()
        plt.show()
    else:
        return img_back

def h_fft_filter(img):
    """Function for performing a horizontally oriented Fast Fourier Transform (FFT) filtering on an image by removing vertical frequencies in the frequency space image and inverse transforming the result back to the spatial domain.

    Args:
        img (np.ndarray): Image that is to be transformed into the frequency domain and filtered.

    Returns:
        np.ndarray: Image that is result of vertical FFT filtering.
    """
    # Use the two dimensional Fast Fourier Transform (FFT) 
    # to produce the Discrete Fourier Transform (DFT) of the image 
    f = np.fft.fft2(img)
    # Shift result so zero frequency component is at center
    f_shift = np.fft.fftshift(f)
    f_mag_spec = np.log(np.abs(f_shift))
    f_mag_spec_norm = ((f_mag_spec - f_mag_spec.min()) 
                       / (f_mag_spec.max() - f_mag_spec.min()))

    # Make a boolean array to mask the DFT
    f_mask = np.zeros_like(f_mag_spec, dtype='bool')

    # Create an array that contains all the column indices except the center 
    # index that has other horizontal information stored that we don't want to 
    # remove
    n_cols = img.shape[1]
    h_mask = np.array([[0, n_cols//2],
                    [n_cols//2 + 1, n_cols]])
    # Go through each col of h_mask and apply the mask components
    # to the DFT mask (f_mask)
    for col in range(h_mask.shape[0]):
        f_mask[f_mask.shape[0]//2, h_mask[0, col]:h_mask[1, col]] = 1

    # Set the values of the DFT that match the True values of the 
    # boolean mask to the mean value of the DFT
    f_mag_spec_masked = f_mag_spec_norm.copy()
    f_mag_spec_masked[f_mask] = np.mean(f_mag_spec_norm)

    # Changing the magnitude spectrum (f_mag_spec) of the DFT is only for 
    # visualization. In order to change the DFT itself, we must change the 
    # shifted frequency component array (f_shift). Since we took the abosulte 
    # value of this array for the visualization, we had to set the masked 
    # values to the mean, but these values are centered around zero, so that is 
    # where we will set the masked values.
    # print(np.mean(f_shift))
    f_shift[f_mask] = 0

    # Inverse shift the zero frequency components back to their natural position
    f_ishift = np.fft.ifftshift(f_shift)
    # Inverse FFT back to the spatial domain
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    return img_back

def test():
    print('T E S T I N G . . .')

if __name__ == '__main__':
    test()