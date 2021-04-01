import numpy as np
import os

import seas.video 

from seas.video import load, dfof, rotate, rescale


from seas.filemanager import sort_experiments, get_exp_span_string, read_yaml
from seas.rois import roi_loader, make_mask, get_masked_region, insert_masked_region, draw_bounding_box
from seas.hdf5manager import hdf5manager
from seas.ica import project, filter_mean
from seas.signalanalysis import sort_noise, lag_n_autocorr
from seas.waveletAnalysis import waveletAnalysis


class Experiment:
    '''
    A class to store mesoscale calcium imaging experiment information and provide functions used for common experiment and video manipulations.
    Includes functionality for loading and rotating videos, cropping to a specific region of interest (defined by user input and/or roi files, loading and storing yaml metadata, etc.)

    Attributes:
        downsample: 
            The spatial downsampling factor
        downsample_t:
            The temporal downsampling factor
        movie:
            The loaded raw movie file 
        path:
            The pathlist of loaded videos
        n_rotations:
            The number of times the video was rotated
        rotate_rois_with_video:
            Whether rois are rotated with the video or not
        bounding_box:
            The bounding coordinates selected for the content of interest from the video file
        shape:
            The video shape
        name:
            The detected name of the experiment from the input files
        dir:
            The directory the video files reside in

    The following attributes are also available if rois are loaded:        
        n_roi_rotations:
            The number of times the rois were rotated
        rois:
            The roi dictionary loaded from FIJI RoiSet.zip file
        roimask:
            A binary masked array denoting where the movie should be masked
        meta:
            The experiment metadata loaded from a yaml file

    Functions:
        load_rois: 
            Load rois from a FIJI RoiSet.zip file
        load_meta:
            Load metadata from a yaml file
        rotate: 
            Rotate the video CCW, adjust mask and bounding box accordingly
        define_mask_boundaries:
            Auto detect the mask boundaries from the loaded roimask
        draw_bounding_box: 
            Launch a GUI to draw a bounding box to crop the movie
        bound_mask:
            Returns the mask bound to the bounding box
        bound_movie:
            Returns the movie bound to the bounding box
        ica_project:
            Perform an ICA projection to the movie

    Initialization Arguments:
        pathlist: 
            The list of paths to load raw video data from, in order.  To sort, use seas.filemanager functions.
        downsample: 
            An integer factor to spatially downsample frames with.  Implements an integer averaging spatial downsample where downsample x downsample pixels are reduced to 1.
        downsample_t: 
            An integer factor to spatially downsample frames with.  Takes the mean between sets of downsample_t frames.
        n_rotations: 
            The number of ccw rotations to rotate the video.
        rotate_rois_with_video: 
            If true, rotate all loaded rois by n_rotations as well.

    '''

    def __init__(self,
                 pathlist,
                 downsample=False,
                 downsample_t=False,
                 n_rotations=0,
                 rotate_rois_with_video=False):
        '''
        Arguments:
            pathlist: 
                The list of paths to load raw video data from, in order.  To sort, use seas.filemanager functions.
            downsample: 
                An integer factor to spatially downsample frames with.  Implements an integer averaging spatial downsample where downsample x downsample pixels are reduced to 1.
            downsample_t: 
                An integer factor to spatially downsample frames with.  Takes the mean between sets of downsample_t frames.
            n_rotations: 
                The number of ccw rotations to rotate the video.
            rotate_rois_with_video: 
                If true, rotate all loaded rois by n_rotations as well.  This parameter overrides roi rotations set by load_rois.
        '''
        print('\nInitializing Experiment\n-----------------------')
        if isinstance(pathlist, str):
            pathlist = [pathlist]

        movie = seas.video.load(pathlist, downsample, downsample_t)
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
        self.rotate_rois_with_video = rotate_rois_with_video

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
        '''
        Rotates movie by self.n_rotations, and updates the shape and bounding box to reflect this change.
        '''
        if self.n_rotations > 0:
            self.movie = rotate(self.movie, self.n_rotations)
            self.bounding_box = np.array([[0, self.movie.shape[1]],
                                          [0, self.movie.shape[2]]
                                         ])  #resets to whole movie
            self.shape = self.bound_movie().shape

    def load_rois(self, path, n_roi_rotations=0):
        '''
        Load rois set in an FIJI/ImageJ RoiSet.zip file to the experiment file, and creates a roimask based on the rois.

        Arguments:
            path: 
                The path to the .zip file.
            n_roi_rotations: 
                The number of CCW rotations to apply to the roimask after loading.  This argument is not used if rotate_rois_with_video was True when loading the experiment.
        '''
        if self.rotate_rois_with_video:
            n_roi_rotations = self.n_rotations

        rois = roi_loader(path)
        self.n_roi_rotations = n_roi_rotations

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

    def load_meta(self, meta_path):
        '''
        Load metadata to the experiment file.  This is not used in any decomposition, but provides a convenient way to save metadata along with the processed file.

        Arguments:
            meta_path: 
                The path to the metadata .yaml file.
        '''
        print('\nLoading Metadata\n-----------------------\n')

        assert metapath.endswith('.yaml'), 'Metadata was not a valid yaml file.'
        meta = read_yaml(meta_path)
        self.meta = meta

    def draw_bounding_box(self):
        '''
        Launches an opencv GUI to click and define a bounding box for the video.  Click and drag to assign the bounding box borders.
        '''
        frame = self.movie[0, :, :].copy()
        frame = rescale(frame, cap=False).astype('uint8')

        ROI = draw_bounding_box(frame, required)

        if ROI is not None:
            self.bounding_box = ROI
            self.shape = (self.shape[0], ROI[0][1] - ROI[0][0],
                          ROI[1][1] - ROI[1][0])

    def define_mask_boundaries(self):
        '''
        Updates the experiment bounding_box to go up to the edge of the rois previously loaded by load_rois.
        '''
        assert hasattr(self, 'roimask'), ('Define roimask before '
                                          'finding boundaries')

        row, cols = np.nonzero(self.roimask)
        ROI = np.array([[np.min(row), np.max(row)],
                        [np.min(cols), np.max(cols)]])

        self.bounding_box = ROI

    def bound_movie(self, movie=None, bounding_box=None):
        '''
        Returns the movie cropped by the bounding box.
        '''
        if bounding_box == None:
            bounding_box = self.bounding_box

        if movie is None:
            movie = self.movie

        ROI = bounding_box
        return movie[:, ROI[0][0]:ROI[0][1], ROI[1][0]:ROI[1][1]]

    def bound_mask(self, bounding_box=None):
        '''
        Returns the roimask cropped by the bounding box.
        '''
        try:
            if bounding_box == None:
                bounding_box = self.bounding_box

            ROI = bounding_box
            return self.roimask[ROI[0][0]:ROI[0][1], ROI[1][0]:ROI[1][1]]
        except:
            return None

    def ica_project(self,
                    movie=None,
                    savedata=True,
                    calc_dfof=True,
                    del_movie=True,
                    n_components=None,
                    svd_multiplier=None,
                    suffix='',
                    output_folder=None,
                    mean_filter_method='wavelet',
                    low_cutoff=0.5):
        '''
        Apply an ica decomposition to the experiment.  If rois and/or a bounding box have been defined, these will be used to crop the movie before filtration.

        By default, results are all saved to a [experiment]_[parameters]_ica.hdf5 file in the same directory as the original video files.

        Arguments:
            movie: 
                The movie to apply ica decomposition to.  If left blank, the movie cropped by the roimask and bounding box is used.
            save_data:
                Whether to save components to a file, or just return as a variable.
            calc_dfof:
                If true, calculate the dFoF before applying ICA decomposition.  If false, ICA is computed on the raw movie.
            del_movie:
                If true, delete the original full movie array before decomposition to save memory.
            n_components:
                A specified number of components to project. If left as None, the svd_multiplier auto component selection is used.
            svd_multiplier:
                The factor to multiply by the detected SVD noise threshold while estimating the number of ICA components to identify.  When left blank, the automatic value set in seas.ica.project is used.
            suffix:
                Optional suffix to append to the ica processed file.
            output_folder:
                By default, the results are saved to an [experiment]_ica.hdf5 file, in the same folder as the original video.  If a different folder is specified by output_folder, the ica file will be saved there instead.
            mean_filter_method: 
                Which method to use while filtering the mean.  Default is highpass wavelet filter.
            low_cutoff: 
                The lower cutoff for a highpass filter.  Default is 0.5Hz.

        Returns:
            components: A dictionary containing all the results, metadata, and information regarding the filter applied.

                mean: 
                    the original video mean
                roimask: 
                    the mask applied to the video before decomposing
                shape: 
                    the original shape of the movie array
                eig_mix: 
                    the ICA mixing matrix
                timecourses: 
                    the ICA component time series
                eig_vec: 
                    the eigenvectors
                n_components:
                    the number of components in eig_vec (reduced to only have 25% of total components as noise)
                project_meta:
                    The metadata for the ica projection
                expmeta:
                    All metadata created for this class
                lag1: 
                    the lag-1 autocorrelation
                noise_components: 
                    a vector (n components long) to store binary representation of which components were detected as noise 
                cutoff: 
                    the signal-noise cutoff value
                mean_filtered: 
                    the filtered mean
                mean_filter_meta: 
                    metadata on how the mean filter was applied

            if the n_components was automatically set, the following additional keys are also returned in components

                svd_cutoff: 
                    the number of components originally decomposed
                lag1_full: 
                    the lag-1 autocorrelation of the full set of components decomposed before cropping to only 25% noise components
                svd_multiplier: 
                    the svd multiplier value used to determine cutoff
        '''
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

            savepath = os.path.join(output_folder, self.name + '_' + suffix)
            print('Saving ICA data to:', savepath)
        else:
            savepath = None

        if savedata:
            f = hdf5manager(savepath)
            components = f.load()  # should be empty if it didn't exist yet.
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
        if 'eig_vec' and 'eig_mix' in components:
            # if data was already in the save path, use it
            print('Found ICA decomposition in components')
        else:

            if hasattr(self, 'roimask'):
                roimask = self.bound_mask()
            else:
                roimask = None

            if movie is None:
                movie = self.bound_movie()

                if calc_dfof:
                    movie = dfof(movie)

            if del_movie:
                print('Deleting original movie to save memory..')
                del self.movie

            #drop dimension and flip to prepare timecourse for ICA
            shape = movie.shape
            t, x, y = shape
            vector = movie.reshape(t, x * y)
            vector = vector.T  # now vector is (x*y, t) for ICA along x*y dimension
            print('M has been reshaped from {0} to {1}\n'.format(
                movie.shape, vector.shape))

            # run ICA projection
            ica_project_kwargs = {'vector': vector, 'shape': shape}

            if svd_multiplier is not None:
                ica_project_kwargs['svd_multiplier'] = svd_multiplier

            if roimask is not None:
                ica_project_kwargs['roimask'] = roimask

            if n_components is not None:
                ica_project_kwargs['n_components'] = n_components

            components = project(**ica_project_kwargs)
            components['expmeta'] = expmeta

            if savedata:
                f.save(components)

        # Calculate other relevant parameters
        components['mean_filtered'] = filter_mean(
            components['mean'],
            filter_method=mean_filter_method,
            low_cutoff=low_cutoff)
        components['mean_filter_meta'] = {
            'mean_filter_method': mean_filter_method,
            'low_cutoff': low_cutoff
        }

        if savedata:
            f.save({
                'noise_components': components['noise_components'],
                'cutoff': components['cutoff'],
                'lag1': components['lag1'],
                'mean_filtered': components['mean_filtered'],
                'mean_filter_meta': components['mean_filter_meta'],
            })
            print('Saved all data to file:')
            f.print()

        return components
