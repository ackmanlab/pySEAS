import os
import argparse
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from seas.gui import run_gui

ap = argparse.ArgumentParser()

ap.add_argument('-i',
                '--input',
                type=argparse.FileType('r'),
                nargs=1,
                required=True,
                help='path to the ica hdf5 file to load.')
ap.add_argument(
    '-r',
    '--rotate',
    type=int,
    default=0,
    help='number of 90 degree counterclockwise rotations for all images.')
ap.add_argument('-d',
                '--default_region',
                type=float,
                default=None,
                help='default number for domain assignment.')
args = vars(ap.parse_args())

run_gui(args['input'][0].name,
        args['rotate'],
        default_assignment=args['default_region'])
