#!/usr/bin/env python3
'''
Default colormaps and regions parameters for functions 

Authors: Sydney C. Weiser
Date: 2020-04-21
'''

import configparser
import os

DEFAULTS = {
    'colormap': {
        'domains': 'twilight_r',
        'components': 'coolwarm',
        'correlation': 'RdGy_r'
    },
    'filestrings': {
        'movies': r'(?:[@-](\d{4}))?\.tif',
        'meta': r'_meta\.yaml',
        'processed': r'_ica\.hdf5',
        'roi': r'_roiset\.zip',
    },
    'expstring':{
        'single_experiment':r'(\d{6}_\d{2})',
        'experiment_span':r'^\d{6}_\d{2}-\d{2}$',
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


def write_values(path, config):
    for key in DEFAULTS:
        config[key] = DEFAULTS[key]

    with open(path, 'w') as configfile:
        config.write(configfile)


def load_defaults(path=None):

    config = configparser.ConfigParser()
    if path is None:
        folder = os.path.dirname(__file__)
        path = os.path.join(folder, 'config.ini')

    try:
        print('Loaded defaults from config file:', path)
        with open(path, 'r') as configfile:
            config.read(path)

    except IOError:
        print('No Config file found.. resetting defaults to', path)
        write_values(path, config)

    except Exception as e:
        print('Uncaught error!:')
        print(e)

    return config


config = load_defaults()
