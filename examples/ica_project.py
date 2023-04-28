import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import seas

from seas.filemanager import experiment_sorter
from seas.experiment import Experiment
from seas.hdf5manager import hdf5manager
import seas.ica
import seas.video

# testing section:
if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('-m',
                    '--movie',
                    type=argparse.FileType('r'),
                    nargs='+',
                    required=False,
                    help='path to the image to be scanned.')
    ap.add_argument(
        '-e',
        '--experiment',
        nargs=1,
        required=False,
        help='name of experiment (YYYYMMDD_EE) for loading associated files.\
            Requires folder argument -f')
    ap.add_argument('-f',
                    '--folder',
                    nargs=1,
                    required=False,
                    help='name of experiment to load associated files.  \
            Requires experiment argument -e')
    ap.add_argument('-of',
                    '--output_folder',
                    nargs=1,
                    required=False,
                    help='alternate folder for storing output')
    ap.add_argument('--rois',
                    type=argparse.FileType('r'),
                    nargs=1,
                    required=False,
                    help='path to .zip file with .rois.')
    ap.add_argument('-s',
                    '--save',
                    action='store_true',
                    help='create filtered video and comparison mp4')
    ap.add_argument('-sro',
                    '--save_roi_overlay',
                    action='store_true',
                    help='create mp4 of filtered video with roi overlay')
    ap.add_argument('-r',
                    '--rotate',
                    type=int,
                    nargs=1,
                    required=False,
                    help='number of times to rotate video ccw')
    ap.add_argument('-rr',
                    '--rotate_rois',
                    nargs=1,
                    type=int,
                    required=False,
                    help='number of times to rotate rois ccw')
    ap.add_argument('-sf',
                    '--savefigures',
                    action='store_true',
                    help='Create component page figures')
    ap.add_argument('-d',
                    '--downsample',
                    type=int,
                    nargs=1,
                    required=False,
                    help='factor to downsample videos by for initial loading')
    ap.add_argument('-dt',
                    '--downsample_time',
                    type=int,
                    nargs=1,
                    required=False,
                    help='temporally downsample videos by factor after loading')
    ap.add_argument('-n',
                    '--ncomponents',
                    type=int,
                    nargs=1,
                    required=False,
                    help='number of components to calculate')
    ap.add_argument('-tmin',
                    '--mintime',
                    type=int,
                    nargs=1,
                    required=False,
                    help='time to crop video at (frame number)')
    ap.add_argument('-tmax',
                    '--maxtime',
                    type=int,
                    nargs=1,
                    required=False,
                    help='time to crop video at (frame number)')
    ap.add_argument('-cm',
                    '--cutoffmultiplier',
                    type=int,
                    nargs=1,
                    required=False,
                    help='svd cutoff multiplier')
    ap.add_argument(
        '-fn',
        '--filternoise',
        action='store_true',
        help='treat noise components as artifacts; dont rebuild them')
    ap.add_argument('-dsf',
                    '--dsfilter',
                    action='store_true',
                    help='make 5x downsample filtered file')
    ap.add_argument(
        '-b',
        '--bound',
        action='store_true',
        help='flag to draw a bounding box to run PCA on cropped region')
    ap.add_argument('-raw',
                    '--raw_movie',
                    action='store_true',
                    help='dont calculate dfof for video')
    ap.add_argument('-avi',
                    action='store_true',
                    help='save avi videos instead of mp4s')
    ap.add_argument('-nmf',
                    '--nomeanfilter',
                    action='store_true',
                    help='dont use any mean filters when rebuilding')
    ap.add_argument('-fr',
                    '--filteringresiduals',
                    action='store_true',
                    help='calculate spatial and temporal artifact residuals')
    ap.add_argument('-ft',
                    '--filteredtiff',
                    action='store_true',
                    help='save the tiff output of the filtered movie')

    args = vars(ap.parse_args())

    # Find all movie and roi paths, load them
    if (args['folder'] is not None) and (args['experiment'] is not None):
        files = experiment_sorter(args['folder'][0],
                                  args['experiment'][0],
                                  verbose=False)
        pathlist = files['movies']

        if len(files['roi']) > 0:
            roipath = files['roi'][0]
        elif args['rois'] is not None:
            roipath = args['rois'][0].name
        else:
            roipath = None
            print('No roipath found.')

        if len(files['meta']) > 0:
            metapath = files['meta'][0]
        else:
            metapath = None
            print('No metadata found.')
    else:
        if args['movie'] is not None:
            pathlist = [path.name for path in args['movie']]
        else:
            pathlist = []

        if args['rois'] is not None:
            roipath = args['rois'][0].name
        else:
            roipath = None
        metapath = None

    if args['maxtime'] is not None:
        tmax = args['maxtime'][0]
    else:
        tmax = None

    if args['cutoffmultiplier'] is not None:
        svd_multiplier = args['cutoffmultiplier'][0]
    else:
        svd_multiplier = None

    if args['mintime'] is not None:
        tmin = args['mintime'][0]
    else:
        tmin = None

    if args['avi']:
        save_avi = True
    else:
        save_avi = False

    if args['nomeanfilter']:
        apply_mean_filter = False
    else:
        apply_mean_filter = True

    if args['output_folder'] is not None:
        output_folder = args['output_folder'][0]
        assert os.path.isdir(output_folder), 'Output folder was not valid!!'
    else:
        output_folder = None

    if args['rotate_rois'] is not None:
        rotate_rois = args['rotate_rois'][0]
    else:
        rotate_rois = 0

    print('\nVideo files ({0}):'.format(len(pathlist)))
    for path in pathlist:
        print('\t' + path)
    print('Rois:\n\t' + str(roipath))
    print('Metadata:\n\t' + str(metapath))

    if len(pathlist) > 0:

        print('Loading videos files and running ICA projection.')
        print('If there is a matching _ica.hdf5 file, ', 'it will be loaded.\n')

        if args['downsample'] is not None:
            downsample = args['downsample'][0]
            print('found spatial downsampling factor:', downsample)
        else:
            downsample = False

        if args['downsample_time'] is not None:
            dt = args['downsample_time'][0]
            print('Found time downsampling factor:', dt)
        else:
            dt = False

        if args['rotate'] is not None:
            rotate = args['rotate'][0]
        else:
            rotate = 0

        suffixes = []

        exp = Experiment(pathlist,
                         downsample=downsample,
                         downsample_t=dt,
                         n_rotations=rotate)

        if roipath is not None:
            print('Roi path found at:', roipath)
            exp.load_rois(roipath, n_roi_rotations=rotate_rois)
            exp.define_mask_boundaries()
        else:
            print('No roi path found.')

        if (tmin is not None) or (tmax is not None):
            print('previous movie shape:', exp.shape)
            exp.movie = exp.movie[tmin:tmax]
            exp.shape = exp.movie.shape
            print('new movie shape:', exp.shape)

            if tmin is not None:
                suffixes.append('tmin' + str(tmin))
            if tmax is not None:
                suffixes.append('tmax' + str(tmax))

        if args['bound']:
            exp.draw_bounding_box()
            suffixes.append('bound')

        if metapath is not None:
            print('Metadata path found at:', metapath)
            exp.load_meta(metapath)
        else:
            print('No metadata path found.')

        if args['ncomponents'] is not None:
            n_components = args['ncomponents'][0]
            suffixes.append('_{0}components'.format(n_components))
        else:
            n_components = None

        if pathlist[0].endswith('.tif') and not args['raw_movie']:
            calc_dfof = True
        else:
            calc_dfof = False

        suffix = '_'.join(suffixes)
        components = exp.ica_project(n_components=n_components,
                                     calc_dfof=calc_dfof,
                                     suffix=suffix,
                                     svd_multiplier=svd_multiplier,
                                     output_folder=output_folder)

        if output_folder is None:
            output_folder = os.path.dirname(exp.path[0])

        basename = output_folder + os.path.sep + \
            exp.name + '_ica'

        if args['filteredtiff']:
            filtered_tiff_path = basename + '_filtered.tiff'

            filtered = seas.ica.rebuild(components,
                                        apply_mean_filter=apply_mean_filter)
            seas.video.save(filtered, filtered_tiff_path)

        if args['save']:
            filtered_path = basename + '_filtered.hdf5'
            savepath = basename + '_filtercomparison'

            if save_avi:
                savepath = savepath + '.avi'
            else:
                savepath = savepath + '.mp4'

            if 'expmeta' in components.keys():
                print('Found expmeta, looking for downsample')
                ica_downsample = components['expmeta']['downsample']
                if downsample:
                    downsample = 4 // ica_downsample
                    print
                else:
                    downsample = 4

                if downsample == 0:
                    downsample = 1
            else:
                print('No downsample information found')

            seas.ica.filter_comparison(
                components,
                # filtered_path=filtered_path,
                # uncomment to save filtered movie in an hdf5 file
                savepath=savepath,
                downsample=downsample,
                apply_mean_filter=apply_mean_filter,
                include_noise=not args['filternoise'])
