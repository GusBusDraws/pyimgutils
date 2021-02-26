import matplotlib.pyplot as plt
import numpy as np
from skimage import (img_as_float, filters, morphology, segmentation, measure, 
                     transform, util)


def subtract_img(img, img_to_subtract):
    """Subtract img_to_subtract from img.

    Args:
        img (np.ndarray): Image that img_to_subtract will be subtracted from. 
        img_to_subtract (np.ndarray): Image that will be subtrated from img.

    Returns:
        np.ndarray: Image that is result of subtraction.
    """
    img = img_as_float(img)
    img_to_subtract = img_as_float(img_to_subtract)

    img_diff = (img[0:min(img.shape[0], img_to_subtract.shape[0]), 
                     0:min(img.shape[1], img_to_subtract.shape[1])] 
                - img_to_subtract[0:min(img.shape[0], 
                                        img_to_subtract.shape[0]), 
                                  0:min(img.shape[1], 
                                        img_to_subtract.shape[1])])

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
        raise ValueError(f'One region expected, found {regions} regions '
                         f'greater than {region_area}.')
    else:
        minr, minc, maxr, maxc = regions[0].bbox

        img_crop = img[minr:maxr, minc:maxc]

    # Rotate and trim remaining region
    img_rotate = transform.rotate(img_crop, rotate_deg, resize=True)
    img_trim = util.crop(img_rotate, 10)

    return img_trim

def v_fft_filter(img):
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

    return img_back

def test():
    print('T E S T I N G . . .')

if __name__ == '__main__':
    test()