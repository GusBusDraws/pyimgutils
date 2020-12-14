import numpy as np
from skimage import img_as_float

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
