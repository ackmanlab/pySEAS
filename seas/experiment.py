import numpy as np
import os

from seas.video import load, dfof, rotate, rescale
from seas.filemanager import sort_experiments, get_exp_span_string
from seas.rois import roi_loader, make_mask, get_masked_region, insert_masked_region, draw_bounding_box
from seas.hdf5manager import hdf5manager
from seas.ica import project, filter_mean
from seas.signal import sort_noise, lag_n_autocorr
from seas.waveletAnalysis import waveletAnalysis


class Experiment:
    '''
    Define the Experiment class.  Must be a (t,x,y) 3 dimensinal video 
    (no color!)
    '''

    def __init__(self,
                 pathlist,
                 loadraw=False,
                 downsample=False,
                 downsample_t=False,
                 n_rotations=0,
                 rotate_rois=False):

        print('\nInitializing Experiment\n-----------------------')
        if isinstance(pathlist, str):
            pathlist = [pathlist]

        movie = load(pathlist, downsample, downsample_t)
        assert (len(movie.shape) == 3), 'File was not a 3 dimensional video.\n'

        if np.any(np.isnan(movie)):
            # if the video was already masked
            roimask = np.zeros(movie[0].shape, dtype='uisnt8')
            roimask[np.where(~np.isnan(movie[0]))] = 1
            self.roimask = roimask

        self.downsample = downsample
        self.downsample_t = downsample_t
        self.movie = movie
        self.path = pathlist
        self.n_rotations = n_rotations

        # define default bounding box as full size video
        self.bounding_box = np.array([[0, self.movie.shape[1]],
                                      [0, self.movie.shape[2]]])
        self.shape = self.bound_movie().shape

        # if multiple experiments included, get a span string
        # (i.e. 01, 02, 03, 04 - > 01-04)
        experiments = sort_experiments(pathlist, verbose=False).keys()
        spanstring = get_exp_span_string(experiments)
        self.name = spanstring

        self.dir = os.path.dirname(pathlist[0])
        self.rotate()

    def rotate(self):
        # rotates a t,y,x movie counter-clockwise n times and
        # updates relevant parameters

        if self.n_rotations > 0:
            self.movie = rotate(self.movie, self.n_rotations)
            self.bounding_box = np.array([[0, self.movie.shape[1]],
                                          [0, self.movie.shape[2]]
                                         ])  #resets to whole movie
            self.shape = self.bound_movie().shape

    def load_rois(self, path, n_roi_rotations=0):

        rois = roi_loader(path)

        # Store in class file
        print(len(rois), 'ROIs found')

        # resize (and flip) if necessary
        if self.downsample is not False:
            print('video was downsampled.. downsampling rois.')
            for roi in rois:
                rois[roi] = rois[roi] // self.downsample

        self.rois = rois

        # Initialize Empty Mask
        roimask = np.zeros(self.shape[1:3], dtype='uint8')

        # Add mask region from all rois
        for i, roi in enumerate(rois):
            roimask += make_mask(rois[roi], self.shape[1:3])

        roimask[np.where(roimask > 1)] = 1

        if roimask.sum().sum() == 0:
            print('Roimask contains no ROI regions.  Not storing..')
            return

        self.roimask = rotate(roimask, n_roi_rotations)
        print('')

    def load_meta(self, metapath):

        print('\nLoading Metadata\n-----------------------\n')

        assert metapath.endswith('.yaml'), 'Metadata was not a valid yaml file.'
        meta = mm.readYaml(metapath)
        self.meta = meta

    def draw_bounding_box(self, required=True):
        frame = self.movie[0, :, :].copy()
        frame = rescale(frame, cap=False).astype('uint8')

        ROI = draw_bounding_box(frame, required)

        if ROI is not None:
            self.bounding_box = ROI
            self.shape = (self.shape[0], ROI[0][1] - ROI[0][0],
                          ROI[1][1] - ROI[1][0])

    def define_mask_boundaries(self):
        assert hasattr(self, 'roimask'), ('Define roimask before '
                                          'finding boundaries')

        row, cols = np.nonzero(self.roimask)
        ROI = np.array([[np.min(row), np.max(row)],
                        [np.min(cols), np.max(cols)]])

        self.bounding_box = ROI

    def bound_movie(self, movie=None, bounding_box=None):
        if bounding_box == None:
            bounding_box = self.bounding_box

        if movie is None:
            movie = self.movie

        ROI = bounding_box
        return movie[:, ROI[0][0]:ROI[0][1], ROI[1][0]:ROI[1][1]]

    def bound_mask(self, bounding_box=None):
        try:
            if bounding_box == None:
                bounding_box = self.bounding_box

            ROI = bounding_box
            return self.roimask[ROI[0][0]:ROI[0][1], ROI[1][0]:ROI[1][1]]
        except:
            return None

    def masked_movie(self, movie=None):
        assert hasattr(self, 'roimask'), (
            'Class instance does not have '
            'a roi mask.  Load a mask before attempting to call the '
            'masked movie.')

        if movie is None:
            movie = self.bound_movie()

        return movie * self.bound_mask()

    def ica_filter(self,
                   A=None,
                   savedata=True,
                   preload_thresholds=True,
                   calc_dfof=True,
                   del_movie=True,
                   n_components=None,
                   svd_multiplier=None,
                   suffix='',
                   output_folder=None,
                   filtermethod='wavelet',
                   low_cutoff=0.5):

        print('\nICA Projecting\n-----------------------')

        if savedata:
            suffix_list = []
            if len(suffix) > 0:
                suffix_list.append(suffix)

            if self.downsample:
                suffix_list.append(str(self.downsample) + 'xds')

            if self.downsample_t:
                suffix_list.append(str(self.downsample_t) + 'xtds')

            if not calc_dfof:
                suffix_list.append('raw')

            if svd_multiplier is not None:
                suffix_list.append(str(svd_multiplier) + 'svdmult')

            if output_folder is None:
                output_folder = os.path.dirname(self.path[0])

            suffix_list.append('ica.hdf5')

            suffix = '_'.join(suffix_list)

            savepath = os.path.join(output_folder,
                self.name + '_' + suffix
                )
            print('Saving ICA data to:', savepath)
        else:
            savepath = None

        if savedata:
            f = hdf5manager(savepath)
            components = f.load() # should be empty if it didn't exist yet.
        else:
            components = {}

        # Load all attributes of experiment class into expmeta dictionary
        # to keep info in ica and filtered files.
        ignore = ['movie', 'filtered', 'notifications']
        expdict = self.__dict__
        expmeta = {}
        for key in expdict:
            if key not in ignore:
                expmeta[key] = expdict[key]
        components['expmeta'] = expmeta
        print('Saving keys under expmeta in PC components:')
        for key in expmeta:
            print(key)

        if savedata:
            f.save(components)

        # calculate decomposition:
        if 'eig_vec' and 'eig_val' in components:
            # if data was already in the save path, use it
            print('Found ICA decomposition in components')
        else:

            if hasattr(self, 'roimask'):
                roimask = self.bound_mask()
            else:
                roimask = None

            if A is None:
                A = self.bound_movie()

                if calc_dfof:
                    A = dfof(A)

            if del_movie:
                print('Deleting original movie to save memory..')
                del self.movie

            #drop dimension and flip to prepare timecourse for PCA
            shape = A.shape
            t, x, y = shape
            vector = A.reshape(t, x * y)
            vector = vector.T  # now vector is (x*y, t) for PCA along x*y dimension
            print('M has been reshaped from {0} to {1}\n'.format(
                A.shape, vector.shape))

            components = project(vector,
                                 shape,
                                 roimask=roimask,
                                 n_components=n_components,
                                 svd_multiplier=svd_multiplier)
            components['expmeta'] = expmeta

            if savedata:
                f.save(components)

        if 'lag_1' not in components.keys():
            components['lag1'] = lag_n_autocorr(components['timecourses'], 1)

        if 'noise_components' not in components.keys():
            components['noise_components'], components['cutoff'] = \
                sort_noise(components['timecourses'])


        components['mean_filtered'] = filter_mean(components['mean'],
                                                  filtermethod=filtermethod,
                                                  low_cutoff=low_cutoff)
        components['mean_filter_meta'] = {
            'filtermethod': filtermethod,
            'low_cutoff': low_cutoff
        }

        if savedata:
            f.save({
                'noise_components': components['noise_components'],
                'cutoff': components['cutoff'], 
                'lag1': components['lag1']
                'mean_filtered': components['mean_filtered'],
                'mean_filter_meta': components['mean_filter_meta'],
            })
            print('Saved all data to file:')
            f.print()

        return components
