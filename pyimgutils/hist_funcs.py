import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from skimage import io, img_as_float, exposure


matplotlib.rcParams['font.size'] = 8


def plot_img_and_hist(image, axes, bins=256):
    """Plot an image along with its histogram and cumulative histogram.

    """
    image = img_as_float(image)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(image, cmap=plt.cm.gray)
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf

def plot_all_hist_ops(img, processing_func=None):
    """Plot original image, contrast stretched image, histogram equalized image, adaptive histogram equalized image and the associated histogram of each image.

    Args:
        img (np.ndarray or str): Image (or path to image if string) of which the histogram operations will be performed.
        processing_func (function, optional): Optional processing function that will be applied to the image. Defaults to None.
    """
    if isinstance(img, str):
        img = io.imread(img)
        img = img_as_float(img)
    elif not isinstance(img, np.ndarray):
        raise ValueError('img must be a Numpy ndarray object or a str represnting the path to an image to be loaded as a Numpy ndarray')

    if processing_func is not None:
        img = processing_func(img)

    # Contrast stretching
    p2, p98 = np.percentile(img, (2, 98))
    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))

    # Equalization
    img_eq = exposure.equalize_hist(img)

    # Adaptive Equalization
    img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)

    # Display results
    fig = plt.figure(figsize=(8, 5))
    axes = np.zeros((2, 4), dtype=np.object)
    axes[0, 0] = fig.add_subplot(2, 4, 1)
    for i in range(1, 4):
        axes[0, i] = fig.add_subplot(2, 4, 1+i, sharex=axes[0,0], sharey=axes[0,0])
    for i in range(0, 4):
        axes[1, i] = fig.add_subplot(2, 4, 5+i)

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img, axes[:, 0])
    ax_img.set_title('Low contrast image')

    y_min, y_max = ax_hist.get_ylim()
    ax_hist.set_ylabel('Number of pixels')
    ax_hist.set_yticks(np.linspace(0, y_max, 5))

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_rescale, axes[:, 1])
    ax_img.set_title('Contrast stretching')

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_eq, axes[:, 2])
    ax_img.set_title('Histogram equalization')

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_adapteq, axes[:, 3])
    ax_img.set_title('Adaptive equalization')

    ax_cdf.set_ylabel('Fraction of total intensity')
    ax_cdf.set_yticks(np.linspace(0, 1, 5))

    # prevent overlap of y-axis labels
    fig.tight_layout()
    plt.show()

def stretch_contrast(img):
    """Contrasting stretching by rescaling the intensity of the image.

    Args:
        img (np.ndarray): Image to which contract stretching will be performed.
    """

    p2, p98 = np.percentile(img, (2, 98))
    img = exposure.rescale_intensity(img, in_range=(p2, p98))

    return img

def equalize_hist(img):
    """Histogram equalization from the scikit-image package.

    Args:
        img (np.ndarray): Image to which histogram equalization will be performed.
    """
    img = exposure.equalize_hist(img)

    return img

def equalize_adapthist(img):
    """Adaptive histogram equalization from the scikit-image package.

    Args:
        img (np.ndarray): Image to which adaptive histogram equalization will be performed.
    """
    img = exposure.equalize_adapthist(img, clip_limit=0.03)

    return img