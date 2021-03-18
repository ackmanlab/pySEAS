#!/bin/env/python3
'''
playtiffmovie.py
Usage:  
    - From command line:  
        python playmovie.py -m file1.tif file2.tif
        python playmovie.py -m testfile*
        python playmovie.py -m file*.tif
        python playmovie.py -m matfile*.mat

Authors: James B. Ackman, Sydney C. Weiser, Brain Mullen
Date: 2016-06-20 16:47:02, 2016-10-13 14:17:55, 2017-06-06 15:24
'''

#import parent directory
import os
import sys
import numpy as np
import argparse

sys.path.append('..')
import seas
from seas.experiment import Experiment
from seas.video import play, dfof

# import wholeBrain as wb
# import hdf5manager as h5
# import fileManager as fm


def main():

    pathlist = [path.name for path in args['movie']]
    print('{0} files found:'.format(len(pathlist)))

    for path in pathlist:
        print('\t' + path)
    print('\n')

    if args['downsample'] is not None:
        downsample = args['downsample'][0]
    else:
        downsample = False

    pathdir = os.path.dirname(path)

    exp = seas.experiment.Experiment(pathlist, downsample=downsample)
    if hasattr(exp, 'filtered_movie'):
        movie = exp.filtered_movie
    else:
        movie = exp.movie

        movie = dfof(movie)
        play(movie)


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-s',
                    '--saveIndex',
                    action='store_true',
                    help='add flag to save the start/stop indexes')
    ap.add_argument('-o',
                    '--overlay',
                    action='store_true',
                    help='add map overlay while playing the movie')
    ap.add_argument('-m',
                    '--movie',
                    type=argparse.FileType('r'),
                    nargs='+',
                    help='path to the image to be scanned',
                    required=True)
    ap.add_argument('-d',
                    '--downsample',
                    type=int,
                    nargs=1,
                    required=False,
                    help='factor to downsample videos by for initial loading')
    args = vars(ap.parse_args())

    if args['overlay']:
        svgpath = os.path.dirname(
            os.path.realpath(__file__)) + '/svg_maps/adult_allen_map.svg'

    main()
