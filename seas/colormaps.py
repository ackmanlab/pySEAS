#!/usr/bin/env python3
'''
Functions for manipulating matploblib and cv2 colormaps, as well as some functions for converting between rescaled to dfof values.  On first import, loads defaults set by seas.defaults configuration into REGION_COLORMAP, DEFAULT_COLORMAP, and COMPONENT_COLORMAP.

Authors: Sydney C. Weiser
Date: 2020-03-28
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import warnings

from .defaults import config


def get_mpl_colormap(colormap_name: str):
    '''
    Convert a matplotlib colormap to a cv2-compatible colormap.

    '''
    # Initialize the matplotlib color map, convert to scalar mappable colormap
    matplotlib_colormap = plt.get_cmap(colormap_name)
    scalarmappable_colormap = plt.cm.ScalarMappable(cmap=matplotlib_colormap)

    # Obtain linear color range
    cv2_colormap = scalarmappable_colormap.to_rgba(np.linspace(0, 1, 256),
                                                   bytes=True)[:, 2::-1]

    return cv2_colormap.reshape(256, 1, 3)


# Functions for colormap manipulation
#------------------------------------


def rescaled_to_dfof(rescaled_value: float, slope: float, array_min: float):
    '''
    Convert a value in 8-bit rescaled units (0-255) to a dfof value.

    Arguments:
        rescaled_value: 
            The rescaled_value in rescaled units to convert
        slope: 
            The slope returned by video.rescale during the dfof to rescaled operation
        array_min: 
            The new minimum value returned by video.rescale during the dfof to rescaled operation

    Returns:
        dfof_value: 
            The rescaled_value converted back to dfof by the scale and array_min parameters.
    '''
    return slope * (rescaled_value - array_min)


def dfof_to_rescaled(dfof_value: float, slope: float, array_min: float):
    '''
    Convert a dfof value to 8-bit rescaled units (0-255).

    Arguments:
        dfof_value: 
            The rescaled_value in dfof units to convert
        slope: 
            The slope returned by video.rescale during the dfof to rescaled operation
        array_min: 
            The new minimum value returned by video.rescale during the dfof to rescaled operation

    Returns:
        rescaled_value: 
            The rescaled_value converted to rescaled units by the scale and array_min parameters.
    '''
    return dfof_value / slope + array_min


def save_colorbar(scale: dict, path: str, colormap=None):
    '''
    Save a plt colorbar with a given scale to a specified path.  Accepts plt or cv2 colormap objects.

    Arguments:
        scale: 
            The scale dictionary returned by rescale_movie.  Must have keys 'min' and 'max' providing the range of the rescale.
        path: 
            Where to save the file to. (supported formats: eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff)
        colormap: 
            Which colormap to save.  Should be a cv2 colormap object or name of a plt colormap.  If left as None, the default colormap will be loaded.  

    Returns:
        Nothing

    Raises:
        KeyError: 
            The min and/or max keys were not in scale dictionary
        ValueError: 
            The path provided was not supported by plt figure outputs.
    '''
    if colormap is None:
        colormap = DEFAULT_COLORMAP

    if type(colormap) is np.ndarray:
        colormap = ListedColormap(colormap.squeeze() / 256)

    ticks = np.linspace(scale['min'], scale['max'], 5).round(4)

    plt.figure(figsize=(1, 2))
    plt.imshow([[0, 0], [0, 0]],
               vmin=scale['min'],
               vmax=scale['max'],
               cmap=colormap)

    cb = plt.colorbar()
    cb.set_ticks(ticks)

    plt.cla()
    plt.axis('off')

    plt.savefig(path, bbox_inches='tight')
    plt.close()


def apply_colormap(video: np.ndarray, colormap: np.ndarray = None):
    '''
    Save a plt colorbar with a given scale to a specified path.  Accepts plt or cv2 colormap objects.

    Arguments:
        video: 
            The video to apply the colormap to.  Should be in format (t,x,y)
        colormap: 
            Which colormap to apply  Should be a cv2 colormap object.  If left as None, the default colormap will be loaded.  

    Returns:
        video_color: 
            The video with colormap applied, in format (t,x,y,c)

    Raises:
        AssertionError: The colormap was invalid.
    '''
    print('\nApplying Color Map to Movie\n-----------------------')

    if colormap is None:
        colormap = DEFAULT_COLORMAP

    assert type(colormap) is np.ndarray, 'Colormap was not cv2 compatible'
    assert colormap.shape == (256, 1, 3), 'Colormap input was not understood'

    sz = video.shape
    video_color = np.zeros((sz[0], sz[1], sz[2], 3),
                           dtype='uint8')  #create extra 4th dim for color
    for i in range(sz[0]):
        cv2.applyColorMap(video[i, :, :].astype('uint8'), colormap,
                          video_color[i, :, :, :])

    return video_color


# Get or set default parameters.
#------------------------------------

DEFAULT_COLORMAP = get_mpl_colormap(config['colormap']['videos'])
COMPONENT_COLORMAP = get_mpl_colormap(config['colormap']['components'])

CUSTOM_PASTEL_LISTED_VALUES = np.array(
    [[123, 219, 148, 255], [255, 178, 240, 255], [153, 85, 255, 255],
     [102, 191, 213, 255], [183, 200, 196, 255], [190, 237, 232, 255]]) / 255

if config['colormap']['regions'] == 'custom_pastel':
    REGION_COLORMAP = ListedColormap(CUSTOM_PASTEL_LISTED_VALUES)
else:
    REGION_COLORMAP = get_mpl_colormap(config['colormap']['components'])
