#!/usr/bin/env python3
'''
Default colormaps and parameters to be used throughout the seas module.  The file is saved to config.txt within the seas library.  On first import, the config file is loaded and can be accessed through seas.defaults.config.

Default parameters include colormaps, region assignment values, and region abbreviations.

Authors: Sydney C. Weiser
Date: 2020-04-21
'''

import configparser
import os
import numpy

DEFAULT_CONFIG = {
    'colormap': {
        'videos': 'rainbow',
        'domains': 'twilight_r',
        'components': 'coolwarm',
        'correlation': 'RdGy_r',
        'regions': 'custom_pastel'
    },
    'regions': {
        "Motor": "1",
        "Motor-Medial": "1.25",
        "Motor-Lateral": "1.5",
        "Somatosensory": "2",
        "Somatosensory-Barrel": "2.25",
        "Somatosensory-Core": "2.5",
        "Somatosensory-Secondary": "2.75",
        "Auditory": "3",
        "Visual": "4",
        "Visual-Primary": "4.25",
        "Visual-Higher": "4.75",
        "Retrosplenial": "5",
        "Olfactory": "6",
        "SuperiorColliculus": "7",
        "Clear": "NaN"
    },

    # config['regions'] = {
    #     "Motor": "1",
    #     "Somatosensory": "2",
    #     "Auditory": "3",
    #     "Visual": "4",
    #     "Retrosplenial": "5",
    #     "Olfactory": "6",
    #     "SuperiorColliculus": "7",
    # }
    'regions-abbrev': {
        "Motor": "M",
        "Motor-Medial": "Ml",
        "Motor-Lateral": "Mm",
        "Somatosensory": "S",
        "Somatosensory-Barrel": "Sb",
        "Somatosensory-Core": "Sc",
        "Somatosensory-Secondary": "Ss",
        "Auditory": "A",
        "Visual": "V",
        "Visual-Primary": "V1",
        "Visual-Higher": "V+",
        "Retrosplenial": "R",
        "Olfactory": "O",
        "SuperiorColliculus": "SC",
        "Clear": "X"
    },
}


def write_config(path, parameters='default'):
    '''
    Writes a config file to path.  The default location is the seas module directory.

    Arguments:
        path: Where to save the config file.

    Returns:
        config: the config file just saved

    Defaults values are as follows:
        'colormap':
            'videos': 'rainbow',
            'domains': 'twilight_r',
            'components': 'coolwarm',
            'correlation': 'RdGy_r',
            'regions': 'custom_pastel'
        'regions': {
            "Motor": "1",
            "Motor-Medial": "1.25",
            "Motor-Lateral": "1.5",
            "Somatosensory": "2",
            "Somatosensory-Barrel": "2.25",
            "Somatosensory-Core": "2.5",
            "Somatosensory-Secondary": "2.75",
            "Auditory": "3",
            "Visual": "4",
            "Visual-Primary": "4.25",
            "Visual-Higher": "4.75",
            "Retrosplenial": "5",
            "Olfactory": "6",
            "SuperiorColliculus": "7",
            "Clear": "NaN"
        'regions-abbrev': {
            "Motor": "M",
            "Motor-Medial": "Ml",
            "Motor-Lateral": "Mm",
            "Somatosensory": "S",
            "Somatosensory-Barrel": "Sb",
            "Somatosensory-Core": "Sc",
            "Somatosensory-Secondary": "Ss",
            "Auditory": "A",
            "Visual": "V",
            "Visual-Primary": "V1",
            "Visual-Higher": "V+",
            "Retrosplenial": "R",
            "Olfactory": "O",
            "SuperiorColliculus": "SC",
            "Clear": "X"
    '''
    config = configparser.ConfigParser()

    if parameters == 'default':
        parameters = DEFAULT_CONFIG

    for key in parameters:
        config[key] = parameters[key]

    with open(path, 'w') as configfile:
        config.write(configfile)

    return config


def load_config(path=None):
    '''
    Load the configuration file.  The default location is the seas module directory.

    Arguments:
        path: Where to load the config file from.

    Returns:
        config: the configuration parameters loaded from file.  

    Defaults values are as follows:
        'colormap':
            'videos': 'rainbow',
            'domains': 'twilight_r',
            'components': 'coolwarm',
            'correlation': 'RdGy_r',
            'regions': 'custom_pastel'
        'regions': {
            "Motor": "1",
            "Motor-Medial": "1.25",
            "Motor-Lateral": "1.5",
            "Somatosensory": "2",
            "Somatosensory-Barrel": "2.25",
            "Somatosensory-Core": "2.5",
            "Somatosensory-Secondary": "2.75",
            "Auditory": "3",
            "Visual": "4",
            "Visual-Primary": "4.25",
            "Visual-Higher": "4.75",
            "Retrosplenial": "5",
            "Olfactory": "6",
            "SuperiorColliculus": "7",
            "Clear": "NaN"
        'regions-abbrev': {
            "Motor": "M",
            "Motor-Medial": "Ml",
            "Motor-Lateral": "Mm",
            "Somatosensory": "S",
            "Somatosensory-Barrel": "Sb",
            "Somatosensory-Core": "Sc",
            "Somatosensory-Secondary": "Ss",
            "Auditory": "A",
            "Visual": "V",
            "Visual-Primary": "V1",
            "Visual-Higher": "V+",
            "Retrosplenial": "R",
            "Olfactory": "O",
            "SuperiorColliculus": "SC",
            "Clear": "X"
    '''
    config = configparser.ConfigParser()
    if path is None:
        folder = os.path.dirname(__file__)
        path = os.path.join(folder, 'config.txt')

    try:
        # print('Loaded defaults from config file:', path)
        with open(path, 'r') as configfile:
            config.read(path)

    except IOError:
        print('No Config file found.. resetting defaults to', path)

        write_config(path)

    with open(path, 'r') as configfile:
        config.read(path)

    return config


config = load_config()
