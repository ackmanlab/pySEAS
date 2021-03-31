#!/usr/bin/env python3
'''
Functions for creating and manipulating domain maps, created from maximum projections of independent components detected from mesoscale calcium imaging videos. 

Authors: Sydney C. Weiser
Date: 2019-06-16
'''
import numpy as np
import os
import scipy.ndimage
import cv2

from seas.hdf5manager import hdf5manager
from seas.video import save, rescale, rotate
from seas.ica import rebuild_mean_roi_timecourse, filter_mean
from seas.rois import make_mask
from seas.colormaps import save_colorbar, REGION_COLORMAP, DEFAULT_COLORMAP


def get_domain_map(components,
                   blur=21,
                   min_size_ratio=0.1,
                   map_only=True,
                   apply_filter_mean=True,
                   max_loops=2,
                   ignore_small=True):
    '''
    Creates a domain map from extracted independent components.  A pixelwise maximum projection of the blurred signal components is taken through the n_components axis, to create a flattened representation of where a domain was maximally significant across the cortical surface.  Components with multiple noncontiguous significant regions are counted as two distinct domains.

    Arguments:
        components: 
            The dictionary of components returned from seas.ica.project.  Domains are most interesting if artifacts has already been assigned through seas.gui.run_gui.
        blur: 
            An odd integer kernel Gaussian blur to run before segmenting.  Domains look smoother with larger blurs, but you can lose some smaller domains.
        map_only:
            If true, compute the map only, do not rebuild time courses under each domain.
        apply_filter_mean:
            Whether to compute the filtered mean when calculating ROI rebuild timecourses.
        min_size_ratio:
            The minimum size ratio of the mean component size to allow for a component.  If a the size of a component is under (min_size_ratio x mean_domain_size), and the next most significant domain over the pixel would result in a larger size domain, this next domain is chosen.
        max_loops:
            The number of times to check if the next most significant domain would result in a larger domain size.  To entirely disable this, set max_loops to 0.
        ignore_small:
            If True, assign undersize domains that were not reassigned during max_loops to np.nan.

    Returns:
        output: a dictionary containing the results of the operation, containing the following keys
            domain_blur:
                The Gaussian blur value used when generating the map
            component_assignment: 
                A map showing the index of which *component* was maximally significant over a given pixel.  Here, 
                This is in contrast to the domain map, where each domain is a unique integer.  
            domain_ROIs: 
                The computed np.array domain map (x,y).  Each domain is represented by a unique integer, and represents a discrete continuous unit.  Values that are masked, or where large enough domains were not detected are set to np.nan.

        if not map_only, the following are also included in the output dictionary:
            ROI_timecourses: 
                The time courses rebuilt from the video under each ROI.  The frame mean is not included in this calculation, and must be re-added from mean_filtered.
            mean_filtered: 
                The frame mean, filtered by the default method.
    '''
    print('\nExtracting Domain ROIs\n-----------------------')
    output = {}
    output['domain_blur'] = blur

    eig_vec = components['eig_vec'].copy()
    shape = components['shape']
    shape = (shape[1], shape[2])

    if 'roimask' in components.keys() and components['roimask'] is not None:
        roimask = components['roimask']
        maskind = np.where(roimask.flat == 1)[0]
    else:
        roimask = None

    if 'artifact_components' in components.keys():
        artifact_components = components['artifact_components']

        print('Switching to signal indices only for domain detection')

        if 'noise_components' in components.keys():
            noise_components = components['noise_components']

            signal_indices = np.where((artifact_components +
                                       noise_components) == 0)[0]
        else:
            print('no noise components found')
            signal_indices = np.where(artifact_components == 0)[0]
        eig_vec = eig_vec[:, signal_indices]

    if blur:
        print('blurring domains...')
        assert type(blur) is int, 'blur was not valid'
        if blur % 2 != 1:
            blur += 1

        eigenbrain = np.empty(shape)
        eigenbrain[:] = np.NAN

        for index in range(eig_vec.shape[1]):

            if roimask is not None:
                eigenbrain.flat[maskind] = eig_vec.T[index]
                blurred = cv2.GaussianBlur(eigenbrain, (blur, blur), 0)
                eig_vec.T[index] = blurred.flat[maskind]
            else:
                eigenbrain.flat = eig_vec.T[index]
                blurred = cv2.GaussianBlur(eigenbrain, (blur, blur), 0)
                eig_vec.T[index] = blurred.flat

    domain_ROIs_vector = np.argmax(np.abs(eig_vec), axis=1).astype('float16')

    if blur:
        domain_ROIs_vector[np.isnan(eig_vec[:, 0])] = np.nan

    if roimask is not None:
        domain_ROIs = np.empty(shape)
        domain_ROIs[:] = np.NAN
        domain_ROIs.flat[maskind] = domain_ROIs_vector

    else:
        domain_ROIs = np.reshape(domain_ROIs_vector, shape)

    output['component_assignment'] = domain_ROIs.copy()

    # remove small domains, separate if more than one domain per component
    ndomains = np.nanmax(domain_ROIs)
    print('domain_ROIs max:', ndomains)

    _, size = np.unique(domain_ROIs[~np.isnan(domain_ROIs)].astype('uint16'),
                        return_counts=True)

    meansize = size.mean()
    minsize = meansize * min_size_ratio

    def replaceindex():
        if n_loops < max_loops:
            if roimask is not None:
                roislice = np.delete(eig_vec[np.where(cc.flat[maskind] == n +
                                                      1)[0], :],
                                     i,
                                     axis=1)
            else:
                roislice = np.delete(eig_vec[np.where(cc.flat == n + 1)[0], :],
                                     i,
                                     axis=1)
            newindices = np.argmax(np.abs(roislice), axis=1)
            newindices[newindices > i] += 1
            domain_ROIs[np.where(cc == n + 1)] = newindices
        else:
            if ignore_small:
                domain_ROIs[np.where(cc == n + 1)] = np.nan

    n_loops = 0
    while n_loops < max_loops:
        n_found = 0
        for i in np.arange(np.nanmax(domain_ROIs) + 1, dtype='uint16'):
            roi = np.zeros(domain_ROIs.shape, dtype='uint8')
            roi[np.where(domain_ROIs == i)] = 1
            cc, n_objects = scipy.ndimage.measurements.label(roi)
            if n_objects > 1:
                objects = scipy.ndimage.measurements.find_objects(cc)
                for n, obj in enumerate(objects):
                    domain_size = np.where(cc[obj] == n + 1)[0].size
                    if domain_size < minsize:
                        n_found += 1
                        replaceindex()
            elif n_objects == 0:
                continue
            else:
                objects = scipy.ndimage.measurements.find_objects(cc)
                domain_size = np.where(roi == 1)[0].size
                if domain_size < minsize:
                    n = 0
                    obj = objects[0]
                    n_found += 1
                    replaceindex()

        n_loops += 1
        print('n undersize objects found:', n_found, '\n')

    print('n domains', np.unique(domain_ROIs[~np.isnan(domain_ROIs)]).size)
    print('nanmax:', np.nanmax(domain_ROIs))

    # split components with multiple centroids
    for i in np.arange(np.nanmax(domain_ROIs) + 1, dtype='uint16'):
        roi = np.zeros(domain_ROIs.shape, dtype='uint8')
        roi[np.where(domain_ROIs == i)] = 1
        cc, n_objects = scipy.ndimage.measurements.label(roi)
        if n_objects > 1:
            objects = scipy.ndimage.measurements.find_objects(cc)
            for n, obj in enumerate(objects):
                if n > 0:
                    ind = np.where(cc == n + 1)
                    domain_ROIs[ind] = np.nanmax(domain_ROIs) + 1

    print('n domains', np.unique(domain_ROIs[~np.isnan(domain_ROIs)]).size)
    print('nanmax:', np.nanmax(domain_ROIs))

    # adjust indexing to remove domains with no spatial footprint
    domain_offset = np.diff(np.unique(domain_ROIs[~np.isnan(domain_ROIs)]))

    adjust_indices = np.where(domain_offset > 1)[0]

    for i in adjust_indices:
        domain_ROIs[np.where(domain_ROIs > i + 1)] -= (domain_offset[i] - 1)

    domain_offset = np.diff(np.unique(domain_ROIs[~np.isnan(domain_ROIs)]))

    if ('expmeta' in components.keys()):
        if 'rois' in components['expmeta'].keys():
            padmask = get_padded_borders(
                domain_ROIs, blur, components['expmeta']['rois'],
                components['expmeta']['n_roi_rotations'],
                components['expmeta']['bounding_box'])
            domain_ROIs[np.where(padmask == 0)] = np.nan
    else:
        print('Couldnt make padded mask')

    output['domain_ROIs'] = domain_ROIs

    if not map_only:
        timecourseresults = get_domain_rebuilt_timecourses(
            domain_ROIs, components, apply_filter_mean=apply_filter_mean)
        for key in timecourseresults:
            output[key] = timecourseresults[key]
    else:
        print('not calculating domain time courses')

    return output


def save_domain_map(domain_ROIs, basepath, blur_level, n_rotations=0):
    '''
    Saves domain maps to pngs for visualization.  Two files are saved to basepath_xb.png and basepath_xb_edges.png. One is the visualization of the domain indices, saved in black and white, the other is just the edge visualization.

    Arguments:
        domain_ROIs:
        basepath: The path to save at, including everything but the file extension.
        blur_level: The Gaussian blur kernel size, only used for generating the image name
        n_rotations: The number of CCW rotations to implement before saving images

    Returns:
        Nothing.
    '''
    if blur_level is not None:
        blurname = str(blur_level)
    else:
        blurname = '?'

    savepath = basepath + blurname + 'b.png'

    edges = get_domain_edges(domain_ROIs)

    domain_ROIs = rotate(domain_ROIs, n_rotations)
    edges = rotate(edges, n_rotations)

    save(domain_ROIs.copy() + edges,
         savepath,
         apply_cmap=False,
         rescale_range=True)
    save(edges,
         savepath.replace('.png', '_edges.png'),
         apply_cmap=False,
         rescale_range=True)


def get_domain_rebuilt_timecourses(domain_ROIs,
                                   components,
                                   apply_filter_mean=True):
    '''
    Get time courses for each domain ROI.  The filtered movie is rebuilt under each ROI (one at a time).  The mean is taken under each domain ROI.  The ROI_timecourses and mean_filtered are returned in output.

    Arguments:
        domain_ROIs: 
            The domain ROI map built by get_domain_map
        components : 
            The results of seas.ica.project, including all eigenvector components.  Used for rebuilding the movie.
        apply_filter_mean:
            Whether to calculate the filtered mean 
    Returns:
        output, a dictionary with the following keys:
            ROI_timecourses: 
                A np array of shape (n_domains, t), with the mean of a given domain, i, under ROI_timecourses[i].
            mean_filtered: 
                The frame mean, of shape (t), filtered by the default filtering method. 
    '''
    output = {}
    print('\nExtracting Domain ROI Timecourses\n-----------------------')
    ROI_timecourses = rebuild_mean_roi_timecourse(components, mask=domain_ROIs)
    output['ROI_timecourses'] = ROI_timecourses

    if apply_filter_mean:
        mean_filtered = filter_mean(components['mean'])
        output['mean_filtered'] = mean_filtered

    return output


def get_domain_edges(domain_ROIs, clear_bg=False, linepad=None):
    '''
    Get the edges of the domain map using canny edge detection.

    Arguments:
        domain_ROIs: 
            The domain map built by get_domain_map.
        clear_bg: 
            True if background values should be nan, otherwise background values are 0.
        linepad:
            Line padding kernel for thicker borders.  Must be an odd integer.
            
    Returns:
        output, a dictionary with the following keys:
            ROI_timecourses: 
                A np array of shape (n_domains, t), with the mean of a given domain, i, under ROI_timecourses[i].
            mean_filtered: 
                The frame mean, of shape (t), filtered by the default filtering method. 
    '''
    # make sure domains are detected when there are more than 256 of them, by adding an offset
    edges = cv2.Canny((domain_ROIs + 1).astype('uint8'), 1, 1)
    edges += cv2.Canny((domain_ROIs + 2).astype('uint8'), 1, 1)

    if linepad is not None:
        assert type(
            linepad) is int, 'invalid line pad.  Provide an odd integer.'
        kernel = np.ones((linepad, linepad), np.uint8)
        edges = cv2.filter2D(edges, -1, kernel)

    if clear_bg:
        edges = edges.astype('float64')
        edges[np.where(edges == 0)] = np.nan

    return edges


def get_padded_borders(domain_ROIs,
                       blur,
                       rois,
                       n_roi_rotations=0,
                       bounding_box=None):
    '''
    Since a gaussian blur between a value with a number and a nan border will result in an increasingly large nan border, get this new border roi boundary.  Rois are used to separate hemispheres rather than blur together.

    Arguments:
        domain_ROIs: 
            The domain map built by get_domain_map.
        blur: 
            True if background values should be nan, otherwise background values are 0.
        rois:
            The roi dict used to generate the roimask.  This should be saved in components['expmeta'] during decomposition.
        n_roi_rotations:
            The number of times the rois were rotated when loading. 
        bounding_box:
            The bounding box applied before ICA decomposition.
            
    Returns:
        output, a dictionary with the following keys:
            ROI_timecourses: 
                A np array of shape (n_domains, t), with the mean of a given domain, i, under ROI_timecourses[i].
            mean_filtered: 
                The frame mean, of shape (t), filtered by the default filtering method. 
    '''
    shape = domain_ROIs.shape
    padmask = np.zeros((shape[0], shape[1]), dtype='uint8')

    for i, roi in enumerate(rois):
        mask = make_mask(rois[roi], shape, bounding_box)

        mask = np.pad(mask.astype('float64'),
                      1,
                      'constant',
                      constant_values=np.nan)
        mask[np.where(mask == 0)] = np.nan

        blurred = cv2.GaussianBlur(mask, (blur, blur), 0)
        blurred = blurred[1:-1, 1:-1]
        padmask[np.where(~np.isnan(blurred))] = 1

    padmask = rotate(padmask, n_roi_rotations)

    return padmask


def domain_map(domain_ROIs, values=None):
    '''
    Used to generate a domain map with a specific coloration scheme.  If values are a calculated metric, such as pearson correlation, each domain i in the domain map will be colored by its value values[i] in the input vector.

    Arguments:
        domain_ROIs: 
            The domain map built by get_domain_map.
        values: 
            An array of values to color the domain_ROIs.  It must be the same length as the number of unique indices in domain_ROIs.  If no values are provided, the domain_ROIs are returned as-is.
            
    Returns:    
        domain_ROIs_colored: 
            the domain_ROIs, with each domain reassigned from its index, to its value given by input 'values'. 
    '''
    if values is not None:
        domainmap = np.zeros(domain_ROIs.shape)
        domainmap[np.where(np.isnan(domain_ROIs))] = np.nan

        if values.ndim > 1:
            domainmap = np.tile(domainmap[:, :, None], values.shape[1])

        for i in np.arange(np.nanmax(domain_ROIs) + 1).astype('uint16'):
            domainmap[np.where(domain_ROIs == i)] = values[i]
    else:
        domainmap = domain_ROIs

    return domainmap


def mosaic_movie(domain_ROIs,
                 ROI_timecourses,
                 savepath=None,
                 t_start=None,
                 t_stop=None,
                 n_rotations=0,
                 colormap='default',
                 resize_factor=1,
                 codec=None,
                 speed=1,
                 fps=10):
    '''
    Creates a mosaic movie, where the original movie is played back as a series of domain timecourses, each displayed over its original domain.

    Arguments:
        domain_ROIs: 
            The domain map built by get_domain_map.
        ROI_timecourses:
            The time series of each ROI, built by get_domain_map if map_only is False, or reconstructed with 'get_domain_rebuilt_timecourses'
        savepath:
            Where to save the movie to.  In most cases, should be an avi or mp4 file path.
        t_start:
            Frame number to start the movie at. If not provided, movie starts at 0.
        t_stop:
            Frame number to stop the movie at.  If not provided, movie ends at last frame.
        n_rotations:
            Number of CCW rotations to apply before saving.
        colormap:
            The colormap to apply.  If not provided, loads the default from the defaults config file.
        resize_factor:
            The factor to resize by when writing the video.
            
    Returns:    
        Nothing.
    '''
    print('\nRebuilding Mosiac Movie\n-----------------------')

    if colormap == 'default':
        colormap = DEFAULT_COLORMAP

    t, x, y = (ROI_timecourses.shape[1], domain_ROIs.shape[0],
               domain_ROIs.shape[1])

    if (t_start is not None) or (t_stop is not None):
        frames = np.arange(t)
        frames = frames[t_start:t_stop]
        t = frames.size

    movie = np.zeros((domain_ROIs.size, t), dtype='float16')

    for roi in np.arange(ROI_timecourses.shape[0]):
        roi_ind = np.where(domain_ROIs.flat == roi)[0]
        movie[roi_ind, :] = ROI_timecourses[roi, t_start:t_stop]

    movie = movie.T.reshape((t, x, y))
    overlay = np.isnan(domain_ROIs).astype('uint8')

    movie = rotate(movie, n_rotations)
    overlay = rotate(overlay, n_rotations)

    if savepath is None:
        print('Finished rebuilding.  Returning movie...')
        return movie
    else:
        print('Finished rebuilding.  Saving file...')

        save(movie,
             savepath,
             rescale_range=True,
             save_cbar=True,
             overlay=overlay,
             colormap=colormap,
             resize_factor=1,
             codec=None,
             speed=1,
             fps=10)


def rolling_mosaic_movie(domain_ROIs,
                         ROI_timecourses,
                         savepath,
                         t_start=None,
                         t_stop=None,
                         n_rotations=0,
                         colormap='default',
                         resize_factor=1,
                         codec=None,
                         speed=1,
                         fps=10):
    '''
    A low memory version of mosaic_movie.  The functionality is the same, except each frame is written one at at a time.  Movie scale values may be slightly different, since they are calculated from the first frame instead of on the entire movie.

        Creates a mosaic movie, where the original movie is played back as a series of domain timecourses, each displayed over its original domain.

    Arguments:
        domain_ROIs: 
            The domain map built by get_domain_map.
        ROI_timecourses:
            The time series of each ROI, built by get_domain_map if map_only is False, or reconstructed with 'get_domain_rebuilt_timecourses'
        savepath:
            Where to save the movie to.  In most cases, should be an avi or mp4 file path.
        t_start:
            Frame number to start the movie at. If not provided, movie starts at 0.
        t_stop:
            Frame number to stop the movie at.  If not provided, movie ends at last frame.
        n_rotations:
            Number of CCW rotations to apply before saving.
        colormap:
            The colormap to apply.  If not provided, loads the default from the defaults config file.
            
    Returns:    
        Nothing.
    '''

    print('\nWriting Rolling Mosiac Movie\n-----------------------')

    if colormap == 'default':
        colormap = DEFAULT_COLORMAP

    # Initialize Parameters
    resize_factor = 1 / resize_factor
    x, y = domain_ROIs.shape
    n_domains = ROI_timecourses.shape[0]
    t = np.arange(ROI_timecourses.shape[1])
    t = t[t_start:t_stop]

    # Set up resizing factors
    w = int(x // resize_factor)
    h = int(y // resize_factor)

    # find codec to use if not specified
    if codec is None:
        if savepath.endswith('.mp4'):
            if os.name == 'posix':
                codec = 'X264'
            elif os.name == 'nt':
                codec = 'XVID'
        else:
            if os.name == 'posix':
                codec = 'MP4V'
            elif os.name == 'nt':
                codec = 'XVID'
            else:
                raise TypeError('Unknown os type: {0}'.format(os.name))

    # initialize movie writer
    display_speed = fps * speed
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(savepath, fourcc, display_speed, (h, w), isColor=True)

    def write_frame(frame):
        # rescale and convert to uint8
        frame = rescale(frame,
                        min_max=(scale['min'], scale['max']),
                        cap=False,
                        verbose=False).astype('uint8')
        frame = cv2.resize(frame, (h, w), interpolation=cv2.INTER_AREA)
        frame = rotate(frame, n_rotations)

        # apply colormap, write frame to .avi
        if colormap is not None:
            frame = cv2.applyColorMap(frame, colormap)
        else:
            frame = np.repeat(frame[:, :, None], 3, axis=2)

        out.write(frame)

    print('Saving dfof video to: ' + savepath)

    frame = np.empty((x, y))
    for f in t:

        if f % 10 == 0:
            print('on frame:', f, '/', t.size)

        frame[:] = np.nan
        for i in range(n_domains):
            frame[np.where(domain_ROIs == i)] = ROI_timecourses[i, f]

        # if first frame, calculate scaling parameters
        if (f == 0):
            mean = np.nanmean(frame)
            std = np.nanstd(frame)

            fmin = mean - 3 * std
            fmax = mean + 7 * std

            scale = {'scale': 255.0 / (fmax - fmin), 'min': fmin, 'max': fmax}

        write_frame(frame)

    out.release()
    print('Video saved to:', savepath)

    cbarpath = os.path.splitext(savepath)[0] + '_colorbar.pdf'
    print('Saving Colorbar to:' + cbarpath)
    save_colorbar(scale, cbarpath, colormap=colormap)
