import os
import sys
import numpy as np
import argparse

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import seas
from seas.experiment import Experiment
from seas.video import play, dfof

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
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
    ap.add_argument('-r',
                    '--rotate',
                    type=int,
                    nargs=1,
                    required=False,
                    help='number of CCW rotations')
    args = vars(ap.parse_args())

    pathlist = [path.name for path in args['movie']]
    print('{0} files found:'.format(len(pathlist)))

    for path in pathlist:
        print('\t' + path)
    print('\n')

    if args['downsample'] is not None:
        downsample = args['downsample'][0]
    else:
        downsample = False

    n_rotations = args['rotate'][0]

    pathdir = os.path.dirname(path)

    exp = seas.experiment.Experiment(pathlist,
                                     downsample=downsample,
                                     n_rotations=n_rotations)

    movie = dfof(exp.movie)
    play(movie)
