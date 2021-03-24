import os
import sys
sys.path.append('..')

from seas.domains import get_domain_map, domain_map, rolling_mosaic_movie, mosaic_movie, get_domain_edges, save_domain_map
from seas.hdf5manager import hdf5manager
from seas.defaults import load_defaults

import argparse
import time

# Argument Parsing
# -----------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument('-i',
                '--input',
                type=argparse.FileType('r'),
                nargs=1,
                required=True,
                help='path to the processed ica file')
ap.add_argument('-svg',
                '--svgpath',
                type=argparse.FileType('r'),
                nargs=1,
                help='optional path to svg file')
ap.add_argument('-b',
                '--blur',
                type=int,
                help='calculate map with given blur value')
ap.add_argument('-map',
                '--maponly',
                action='store_true',
                help='make map only; dont calculate timecourses')
ap.add_argument('-fc',
                '--force',
                action='store_true',
                help='force re-calculation')
ap.add_argument('-mm',
                '--mosaic_movie',
                action='store_true',
                help='build mosiac movie')
ap.add_argument('-avi',
                action='store_true',
                help='save avi videos instead of mp4s')
ap.add_argument('-f', '--figures', action='store_true', help='create figures')
ap.add_argument(
    '-mc',
    '--mapcomparison',
    type=argparse.FileType('r'),
    nargs=1,
    help='hdf5 file to load with alternate map to rebuild timecourses')
ap.add_argument('--rotate', nargs=1, type=int, help='rotate movies before saving')
args = vars(ap.parse_args())

path = [path.name for path in args['input']][0]
print('Input file found:', path)

savepath = path.replace('.hdf5', '_')
savepath = savepath.replace('_reduced', '')

# Load relevant objects from processed file
# -----------------------------------------------
f = hdf5manager(path)
f.print()

rotate = args['rotate'][0]

if ('domain_ROIs' not in f.keys()) or args['force']:

    if 'artifact_components' not in f.keys():
        if not args['force']:
            print('PCA gui must be run before parcellation analyses')
            raise KeyError('Artifact components not found!')
        else:
            print('Artifact components were not found.')
            print('Running segemntation despite lack of artifact',
                  'flag due to force option')

    components = f.load([
        'eig_vec', 'artifact_components', 'cutoff', 'expmeta', 'roimask',
        'shape', 'n_components', 'eig_mix', 'noise_components', 'mean'
    ])

    if 'cutoff' in components.keys():
        cutoff = components['cutoff']
        print('found cutoff:', cutoff, '\n')
    else:
        print('cutoff not found.. using all components')
        cutoff = components['n_components']

    if args['blur'] is not None:
        blur = args['blur']
    else:
        blur = 21

    output = get_domain_map(components,
                            cutoff=cutoff,
                            # savepath=savepath + 'domainROIs.png',
                            maponly=args['maponly'],
                            blur=blur)

    f.save(output)
    domain_ROIs = output['domain_ROIs']

else:
    print('Found domain ROIs.')
    domain_ROIs = f.load('domain_ROIs')

if args['mosaic_movie']:
    domain_ROIs = f.load('domain_ROIs')
    ROI_timecourses = f.load('ROI_timecourses')
    mean = f.load('mean_filtered')
    ROI_timecourses += mean.T

    mosaicpath = savepath + 'mosaic_movie'

    if args['avi']:
        mosaicpath = mosaicpath + '.avi'
    else:
        mosaicpath = mosaicpath + '.mp4'

    try:
        mosaic_movie(domain_ROIs, ROI_timecourses, mosaicpath, n_rotations=rotate)

    except Exception as e:
        print('An error occured!')
        print('\t', e)
        rolling_mosaic_movie(domain_ROIs, ROI_timecourses, mosaicpath, n_rotations=rotate)

if args['mapcomparison'] is not None:
    mappath = args['mapcomparison'][0].name
    basename = os.path.basename(mappath).replace('.hdf5',
                                                 '').replace('_ica', '')
    print('found map comparion file:', basename)

    map_comparison = f.load('map_comparison')

    try:
        g = hdf5manager(mappath)
        alternate_ROIs = g.load('domain_ROIs')
    except Exception as e:
        print('Error loading alternate map:')
        print(e)

    components = f.load([
        'eig_vec', 'artifact_components', 'cutoff', 'expmeta', 'roimask',
        'shape', 'n_components', 'eig_mix', 'noise_components', 'mean'
    ])
    output = get_domain_rebuilt_timecourses(alternate_ROIs, components)
    output['alternate_ROIs'] = alternate_ROIs

    if 'region_assignment' in g.keys():
        output['region_assignment'] = g.load('region_assignment')

    map_comparison[basename] = output
    f.save({'map_comparison': map_comparison})

### Make Figures
if args['figures']:
    print('Making figures..')
    edges = get_domain_edges(domain_ROIs)
    try:
        blur_level = f.load('domain_blur')
    except KeyError:
        blur_level = None

    save_domain_map(domain_ROIs, savepath, blur_level, n_rotations=rotate)

    # savepath=savepath + 'domainROIs.png',


    # # domainmap
    # wb.saveFile(savepath + 'domainROIs.png', domain_ROIs.copy() + edges,
    #     apply_cmap=False, rescale=True)

    # # # edges
    # wb.saveFile(savepath + 'domainROIs_edges.png',
    #     edges, apply_cmap=False, rescale=True)

    # domain assignment from gui
    if 'region_assignment' in f.keys():
        region_assignment = f.load('region_assignment')
        from convertcmap import region_cm, region_cm_colors, get_mpl_colormap
        # region_cm_colors = (region_cm_colors * 255)

        config = load_defaults()

        # region_cm_colors = get_mpl_colormap(config['colormap']['domains'])
        # region_cm_colors = np.squeeze(region_cm_colors)
        # print(region_cm_colors.shape)

        # convert to uint8, get rid of alpha channel
        # region_cm_colors = region_cm_colors[:,:-1].astype('uint8')
        # region_cm_colors = region_cm_colors[:,::-1]# BGR to RGB

        regionmap = domain_map(domain_ROIs, values=region_assignment)
        regionmap -= 1

        region_cmap = cv2.applyColorMap(
            video.rescale_movie(regionmap, verbose=False).astype('uint8'),
            get_mpl_colormap(config['colormap']['domains']))
        # region_cmap = domainMap(regionmap, values=region_cm_colors)

        edge_indices = np.where(edges == 255)
        region_cmap[edge_indices] = 0

        # padmask = getPaddedBorders(domain_ROIs, blur,
        #                 components['expmeta']['rois'], components['expmeta']['bounding_box'])

        mask = np.ones(domain_ROIs.shape)
        ind = np.where(np.isnan(domain_ROIs))
        mask[ind] = 0
        wb.saveFile(savepath + 'domainROIs_regions.png', region_cmap, mask=mask)

        bordermap = borderLevels(domain_ROIs, region_assignment, flip=True)
        wb.saveFile(savepath + 'domainROIs_bordermap.png',
                    255 - bordermap,
                    apply_cmap=False)

        if 'map_comparison' in f.keys():
            print('map comparison found')
            map_comparison = f.load('map_comparison')
            for key in map_comparison:
                if (key == 'voronoi_ROIs') | (key == 'grid_ROIs'):
                    map_borders = getDomainEdges(
                        map_comparison[key]['alternate_ROIs'],
                        linepad=15 // 2 + 1,
                        clearbg=False)
                    wb.saveFile(savepath + key + '_bordermap.png',
                                255 - map_borders,
                                apply_cmap=False)
