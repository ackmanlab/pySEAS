import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import seas

from seas.filemanager import experiment_sorter
from seas.experiment import Experiment

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
    ap.add_argument('-i',
                    '--input',
                    type=argparse.FileType('r'),
                    nargs=1,
                    required=False,
                    help='path to the ica hdf5 file to load.')
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
    ap.add_argument(
        '--rois',
        type=argparse.FileType('r'),
        nargs=1,
        required=False,
        help='path to .zip file with .rois, or .roi file containing '
        'ROIs to associate with video object.')
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
    ap.add_argument(
        '-nf',
        '--notify',
        action='store_true',
        help='use this flag to notify on slack when code is finished')
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
    ap.add_argument('-pca',
                    '--pca',
                    action='store_true',
                    help='calculate PCA instead of ICA')
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

    if args['input'] is not None:
        inputpath = args['input'][0].name
    else:
        inputpath = None

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
        filtermean = False
    else:
        filtermean = True

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
    print('Input files:\n\t' + str(inputpath) + '\n')

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

        suffix = None

        exp = Experiment(pathlist,
                         downsample=downsample,
                         downsample_t=dt,
                         n_rotations=rotate)

        if roipath is not None:
            print('Roi path found at:', roipath)
            exp.load_rois(roipath,
                         n_roi_rotations=rotate_rois)
            exp.define_mask_boundaries()
        else:
            print('No roi path found.')

        if (tmin is not None) or (tmax is not None):
            print('previous movie shape:', exp.shape)
            exp.movie = exp.movie[tmin:tmax]
            exp.shape = exp.movie.shape
            print('new movie shape:', exp.shape)

            if tmin is not None:
                suffix = 'tmin' + str(tmin)
            if tmax is not None:
                suffix = 'tmax' + str(tmax)

        if args['notify']:
            exp.load_notifier()

        if args['bound']:
            exp.draw_bounding_box()

        if metapath is not None:
            print('Metadata path found at:', metapath)
            exp.load_meta(metapath)
        else:
            print('No metadata path found.')

        if args['ncomponents'] is not None:
            n_components = args['ncomponents'][0]
            if suffix is not None:
                suffix = suffix + '_{0}components'.format(n_components)
            else:
                suffix = '{0}components'.format(n_components)
        else:
            n_components = None

        if pathlist[0].endswith('.tif') and not args['raw_movie']:
            calc_dfof = True
        else:
            calc_dfof = False

        components = exp.ica_filter(n_components=n_components,
                                    calc_dfof=calc_dfof,
                                    suffix=suffix,
                                    svd_multiplier=svd_multiplier,
                                    output_folder=output_folder)

        if output_folder is None:
            output_folder = os.path.dirname(exp.path[0])

        basename = output_folder + os.path.sep + \
            exp.name + '_ica'

        if args['save']:
            filterpath = basename + '_filtered.hdf5'
            savepath = basename + '_ica_filtercomparison'

            if save_avi:
                savepath = savepath + '.avi'
            else:
                savepath = savepath + '.mp4'

            filtered = exp.filtered

            g = h5(filtered_path)
            g.save({'filtered': filtered})

            if 'expmeta' in components.keys():
                g.save(components['expmeta'])
            else:
                print('no expmeta found.')
                print('keys', components.keys())
            if 'filter' in components.keys():
                g.save({'filter': components['filter']})

            if 'expmeta' in components.keys():
                print('Found expmeta, looking for downsample')
                ica_downsample = components['expmeta']['downsample']
                if downsample:
                    downsample = 4 // ica_downsample
                    print
                else:
                    downsample = 4
            else:
                print('No downsample information found')

            filter_comparison(components,
                              filtered=filtered,
                              videopath=savepath,
                              downsample=downsample,
                              filterpath=filterpath,
                              filtermean=filtermean,
                              include_noise=not args['filternoise'],
                              flip=flip)

    # take the processed file, create reduced copy, and
    # create filter comparison

    if inputpath is not None:
        basename = inputpath.replace('.hdf5', '')

        if output_folder is not None:
            output_folder = os.path.dirname(exp.path[0])
            basename = os.path.join(output_folder, os.path.basename(basename))

        f = h5(inputpath)
        components = f.load(ignore='vector')

        if args['savefigures']:
            figpath = basename + '_components'
            PCfigure(components, figpath)

        if tmax is not None:
            basename = basename + '_tmax' + str(tmax)

        if tmin is not None:
            basename = basename + '_tmin' + str(tmin)

        savepath = basename + '_filtercomparison'

        if save_avi:
            savepath = savepath + '.avi'
        else:
            savepath = savepath + '.mp4'

        filterpath = basename + '_filtered.hdf5'

        if args['save']:
            ds = args['downsample']
            if ds is None:
                ds = [4]
            ds = int(ds[0])

            if 'expmeta' in components.keys():
                print('Found expmeta, looking for downsample..')

                ica_downsample = components['expmeta']['downsample']
                if ica_downsample:
                    print('Video was downsampled by', ica_downsample,
                          'before ICA')

                if ica_downsample:
                    downsample = ds // ica_downsample
                else:
                    downsample = ds

                if downsample < 1:
                    downsample = 1
            else:
                print('No downsample information found')
                downsample = ds

            print('Downsampling by:', downsample, '\n')

            filter_comparison(components,
                              videopath=savepath,
                              downsample=downsample,
                              filterpath=filterpath,
                              include_noise=not args['filternoise'],
                              t_stop=tmax,
                              t_start=tmin,
                              filtermean=filtermean,
                              flip=flip)

        if args['save_roi_overlay']:

            assert 'domain_ROIs' in f.keys(), 'ROI overlay requires'\
                'domain ROIs to be calculated!  Run wholeBrainDomainROIs.py first.'

            filterpath = basename + '_filtered.hdf5'

            if os.path.isfile(filterpath):
                g = h5(filterpath)
                filtered = g.load('filtered')
            else:
                filtered = rebuild(components,
                                   returnmeta=True,
                                   include_noise=not args['filternoise'],
                                   t_stop=tmax,
                                   t_start=tmin,
                                   filtermean=filtermean)

            domainROIs = f.load('domain_ROIs')
            edges = wb.cv2.Canny((domainROIs + 1).astype('uint8'), 1, 1)

            overlay = np.zeros(domainROIs.shape)
            overlay[np.isnan(domainROIs)] = 1
            overlay += edges

            overlaypath = basename + '_filtered_overlaymovie'

            if save_avi:
                overlaypath = overlaypath + '.avi'
            else:
                overlaypath = overlaypath + '.mp4'

            if flip:
                filtered = filtered[:, ::-1]
                overlay = np.flipud(overlay)

            wb.saveFile(overlaypath,
                        filtered,
                        overlay=overlay,
                        save_cbar=True,
                        rescale=True)

        if args['dsfilter']:
            ds = args['downsample']
            if ds is None:
                ds = [5]
            ds = int(ds[0])

            dspath = basename + '_filtered_{0}xds.hdf5'.format(ds)
            filterpath = basename + '_filtered.hdf5'
            downsampleFiltered(filterpath, dspath, downsample=ds)

        if 'rebuildmeta' in components.keys():
            f.save({'rebuildmeta': components['rebuildmeta']})

        if args['filteringresiduals']:
            assert 'artifact_components' in components, 'Assign artifact_components first'

            # invert indices and get signal components to exclude rebuilding:
            signal_components = (
                components['artifact_components'] == 0).astype('uint8')
            artifact_movie = rebuild(components,
                                     artifact_components=signal_components)

            if 'roimask' in components:
                roimask = components['roimask']
                outsideind = np.where(roimask == 0)

                artifact_movie = artifact_movie - components[
                    'mean_filtered'][:, None, None]
                artifact_movie[:, outsideind[0], outsideind[1]] = 0

            filtering_residuals = {}
            residuals_spatial = np.abs(artifact_movie).sum(axis=0)
            residuals_temporal = np.abs(artifact_movie).sum(axis=1).sum(axis=1)

            filtering_residuals['artifact_residuals'] = {
                'residuals_spatial': residuals_spatial,
                'residuals_temporal': residuals_temporal
            }

            # get total signal from video:
            original_movie = rebuild(components, artifact_components='none')

            if 'roimask' in components:
                original_movie = original_movie - components[
                    'mean_filtered'][:, None, None]
                original_movie[:, outsideind[0], outsideind[1]] = 0

            residuals_spatial = np.abs(original_movie).sum(axis=0)
            residuals_temporal = np.abs(original_movie).sum(axis=1).sum(axis=1)

            filtering_residuals['total_signal'] = {
                'residuals_spatial': residuals_spatial,
                'residuals_temporal': residuals_temporal
            }

            f.save({'filtering_residuals': filtering_residuals})
