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
                   filtermean=True,
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
    # del eig_vec

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
                        # print('found multi small obj', i)
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
                    # print('found single small obj', i)
                    n_found += 1
                    replaceindex()
    #
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
            padmask = get_padded_borders(domain_ROIs, blur,
                                         components['expmeta']['rois'],
                                         components['expmeta']['bounding_box'])
            domain_ROIs[np.where(padmask == 0)] = np.nan
    else:
        print('Couldnt make padded mask')

    output['domain_ROIs'] = domain_ROIs

    if not map_only:
        timecourseresults = get_domain_rebuilt_timecourses(
            domain_ROIs, components, filtermean=filtermean)
        for key in timecourseresults:
            output[key] = timecourseresults[key]
    else:
        print('not calculating domain timecourses')

    return output


def save_domain_map(domain_ROIs, basepath, blur_level, n_rotations=0):

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


def get_domain_rebuilt_timecourses(domain_ROIs, components, filtermean=True):

    output = {}
    print('\nExtracting Domain ROI Timecourses\n-----------------------')
    ROI_timecourses = rebuild_mean_roi_timecourse(components, mask=domain_ROIs)
    output['ROI_timecourses'] = ROI_timecourses

    if filtermean:
        mean_filtered = filter_mean(components['mean'])
        output['mean_filtered'] = mean_filtered

    return output


def get_domain_edges(domain_ROIs, clearbg=False, linepad=None):
    edges = cv2.Canny((domain_ROIs + 1).astype('uint8'), 1, 1)

    edges += cv2.Canny((domain_ROIs + 2).astype('uint8'), 1, 1)

    if linepad is not None:
        assert type(
            linepad) is int, 'invalid line pad.  Provide an odd integer.'
        kernel = np.ones((linepad, linepad), np.uint8)
        edges = cv2.filter2D(edges, -1, kernel)

    if clearbg:
        edges = edges.astype('float64')
        edges[np.where(edges == 0)] = np.nan

    return edges


def get_padded_borders(domain_ROIs, blur, rois, bounding_box=None):

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

    return padmask


def domain_map(domain_ROIs, values=None):

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


# def borderLevels(domain_ROIs, region_assignment, pad=15, flip=False, returnlayers=False):

#     macro_map = domain_map(domain_ROIs, values=region_assignment)
#     domain_borders = get_domain_edges(domain_ROIs, linepad=pad//2+1, clearbg=True).astype('float64')

#     colored_borders = domain_borders.copy() / 255
#     macro_borders = np.zeros(macro_map.shape)
#     macro_borders[:] = np.nan

#     macro_borders = get_domain_edges(domain_map(domain_ROIs, values=region_assignment), linepad=pad, clearbg=True)

#     region_assignment_adjusted = np.empty(region_assignment.shape)
#     for i, value in enumerate(np.unique(region_assignment[~np.isnan(region_assignment)])):
#         region_assignment_adjusted[np.where(region_assignment == float(value))] = i

#     meso_borders = get_domain_edges(domainMap(domain_ROIs, values=region_assignment_adjusted), linepad=(pad//4)*3, clearbg=True)

#     if flip:
#         domain_borders = np.flipud(domain_borders)
#         meso_borders = np.flipud(meso_borders)
#         macro_borders = np.flipud(macro_borders)

#     if returnlayers:
#         return macro_borders, meso_borders, domain_borders
#     else:
#         border_overlay = np.zeros(macro_borders.shape)
#         border_overlay[np.where(domain_borders == 255)] = 255//2
#         border_overlay[np.where(meso_borders == 255)] = 255*3//4
#         border_overlay[np.where(macro_borders == 255)] = 255
#         return border_overlay

# def getDomainProperties(domain_ROIs):
#     from skimage import measure

#     n_ROIs = (np.nanmax(domain_ROIs)+1).astype('uint16')
#     drs = (domain_ROIs.copy() + 1).astype('uint16')
#     props = measure.regionprops(drs)

#     data = {
#         'area' : np.zeros((n_ROIs,), dtype='uint64'),
#         'eccentricity' : np.zeros((n_ROIs,), dtype='float16'),
#         'centroid'  : np.zeros((n_ROIs,2), dtype='int16')
#     }

#     for i, prop_i in enumerate(props):
#         data['area'][i] = prop_i.area
#         data['eccentricity'][i] = prop_i.eccentricity
#         data['centroid'][i] = np.array(prop_i.centroid).astype('uint16')

#     return data

# def getNearestNeighbors(props, domain_ROIs, props_2, domain_ROIs_2):
#     distance = np.zeros(props['centroid'].shape[0])
#     overlap = np.zeros(props['centroid'].shape[0])
#     index = np.zeros(props['centroid'].shape[0], dtype='uint16')

#     tree = scipy.spatial.KDTree(props_2['centroid'].astype('float32'))

#     minshape = np.zeros((2,2),dtype='uint16')
#     minshape[0] = domain_ROIs.shape
#     minshape[1] = domain_ROIs_2.shape
#     minshape = minshape.min(axis=0)
#     domain_ROIs = domain_ROIs[:minshape[0], :minshape[1]]
#     domain_ROIs_2 = domain_ROIs_2[:minshape[0], :minshape[1]]

#     for i, pt in enumerate(props['centroid']):
#         d, ind = tree.query(pt, k=1)
#         index[i] = ind
#         distance[i] = d
# #         overlap[i] = np.where((domain_ROIs == i) & (domain_ROIs_2 == ind))[0].size / params['area'][i]
#         try:
#             overlap[i] = np.where((domain_ROIs == i) & (domain_ROIs_2 == ind))[0].size /\
#                 np.where((domain_ROIs == i) | (domain_ROIs_2 == ind))[0].size
#         except:
#             print('error!')
#             print('index:', i)
#             print('centroid:', pt)
#             overlap[i] = np.nan

#     return index, distance, overlap

# def getOffsetNearestNeighbors(params, domain_ROIs, props_2, domain_ROIs_2, offset, display=False, returnoffset=False):
#     distance = np.zeros(params['centroid'].shape[0])
#     overlap = np.zeros(params['centroid'].shape[0])
#     index = np.zeros(params['centroid'].shape[0], dtype='int16')

#     dx, dy = offset
#     dx = int(dx)
#     dy = int(dy)

#     # adjust inputs by offset
#     props_2_adjusted = props_2.copy()
#     props_2_adjusted['centroid'] = props_2['centroid'].copy()
#     props_2_adjusted['centroid'] += np.array([dx, dy], dtype='uint16')

#     padwidth= int(np.max(np.abs([dx,dy])))

#     domain_ROIs = np.pad(domain_ROIs.copy(), mode='constant', pad_width=padwidth, constant_values=np.nan)
#     domain_ROIs_2 = np.pad(domain_ROIs_2.copy(), mode='constant', pad_width=padwidth, constant_values=np.nan)

#     domain_ROIs_2_adjusted = np.zeros(domain_ROIs_2.shape)
#     domain_ROIs_2_adjusted[:] = np.nan

#     if dx > 0:
#         domain_ROIs_2[dx:,:] = domain_ROIs_2[:-dx,:]
#     elif dx < 0:
#         domain_ROIs_2[:dx,:] = domain_ROIs_2[-dx:,:]

#     if dy > 0:
#         domain_ROIs_2[:,dy:] = domain_ROIs_2[:,:-dy]
#     elif dy < 0:
#         domain_ROIs_2[:,:-dy] = domain_ROIs_2[:,dy:]

#     # make sure axes are even
#     minshape = np.zeros((2,2),dtype='uint16')
#     minshape[0] = domain_ROIs.shape
#     minshape[1] = domain_ROIs_2.shape
#     minshape = minshape.min(axis=0)

#     domain_ROIs = domain_ROIs[:minshape[0], :minshape[1]]
#     domain_ROIs_2 = domain_ROIs_2[:minshape[0], :minshape[1]]

#     if display:
#         plt.figure(figsize=(10,10))
#         plt.imshow(get_domain_edges(domain_ROIs, clearbg=True, linepad=10), cmap='coolwarm_r', alpha=0.75)
#         plt.imshow(get_domain_edges(domain_ROIs_2, clearbg=True, linepad=10), alpha=0.75)

# #         plt.imshow(domain_ROIs + domain_ROIs_2)
#         plt.axis('off')
#         plt.show()

#     tree = scipy.spatial.KDTree(props_2_adjusted['centroid'].astype('float32'))

#     for i, pt in enumerate(params['centroid']):
#         d, ind = tree.query(pt, k=1)
#         index[i] = ind
#         distance[i] = d
# #         overlap[i] = np.where((domain_ROIs == i) & (domain_ROIs_2 == ind))[0].size / params['area'][i]
#         try:
#             overlap[i] = np.where((domain_ROIs == i) & (domain_ROIs_2 == ind))[0].size /\
#                 np.where((domain_ROIs == i) | (domain_ROIs_2 == ind))[0].size
#         except:
#             print('error!')
#             print('index:', i)
#             print('centroid:', pt)
#             overlap[i] = np.nan

#     if returnoffset:
#         return index, distance, overlap, offset
#     else:
#         return index, distance, overlap

# def getIdealOffset(props, domain_ROIs, props_2, domain_ROIs_2, display=False):
#     try:
#         from multiprocessing import Pool
#         import itertools
#         multiprocessing = True
#     except:
#         multiprocessing = False

#     xrange = np.arange(-40,50, 10)
#     yrange = np.arange(-40,50, 10)

#     overlap = np.zeros((xrange.size, yrange.size))
#     shift = np.zeros((xrange.size, yrange.size))

#     if multiprocessing:
#         dxdy = list(itertools.product(xrange,yrange))
#         mapped_inputs = [(props, domain_ROIs, props_2, domain_ROIs_2, coords, False, True) for coords in dxdy]

#         pool = Pool()
#         outputs = pool.starmap(getOffsetNearestNeighbors, mapped_inputs)
#         pool.close()
#         pool.join()

#         for _, _, overlap_ij, offset in outputs:
#             i = np.where(xrange == offset[0])[0][0]
#             j = np.where(xrange == offset[1])[0][0]
#             overlap[i,j] = overlap_ij.mean()

#     else:
#         for i, dx in enumerate(xrange):
#             print(i+1, '/', len(xrange))
#             for j, dy in enumerate(yrange):

#                 _,_,overlap_ij = getOffsetNearestNeighbors(props, domain_ROIs,
#                     props_2, domain_ROIs_2, (dx,dy))
#                 overlap[i,j] = overlap_ij.mean()

#     if display:
#         plt.imshow(overlap, interpolation='none')
#         plt.colorbar()
#         plt.xticks(np.arange(xrange.size), xrange)
#         plt.yticks(np.arange(yrange.size), yrange)
#         plt.show()

#     ideal_shift_index = np.unravel_index(overlap.argmax(), overlap.shape)
#     ideal_shift = xrange[ideal_shift_index[0]], yrange[ideal_shift_index[1]]
# #     print('ideal_shift:', ideal_shift)

#     xrange_fine = np.arange(ideal_shift[0] - 10, ideal_shift[0] + 10)
#     yrange_fine = np.arange(ideal_shift[1] - 10, ideal_shift[1] + 10)

# #     print('xrange_fine:', xrange_fine)
# #     print('yrange_fine:', yrange_fine)

#     from  scipy.interpolate import RectBivariateSpline

#     f = RectBivariateSpline(xrange, yrange, overlap)

#     overlap_fine = np.zeros((xrange_fine.size, yrange_fine.size))
#     for i, x_i in enumerate(xrange_fine):
#         overlap_fine[i,:] = f.ev(x_i, yrange_fine)

#     if display:
#         plt.imshow(overlap_fine, interpolation='none')
#         plt.colorbar()
#         plt.xticks(np.arange(xrange_fine.size), xrange_fine)
#         plt.yticks(np.arange(yrange_fine.size), yrange_fine)
#         plt.show()

#     ideal_shift_index = np.unravel_index(overlap_fine.argmax(), overlap_fine.shape)
#     ideal_shift = xrange_fine[ideal_shift_index[0]], yrange_fine[ideal_shift_index[1]]

#     # print('ideal_shift', ideal_shift)

#     return ideal_shift

# def generateVoronoiMap(domain_ROIs):

#     from scipy.spatial import Voronoi

#     # from the internet: https://stackoverflow.com/questions/20515554/colorize-voronoi-diagram/20678647#20678647

#     def voronoi_finite_polygons_2d(vor, radius=None):
#         """
#         Reconstruct infinite voronoi regions in a 2D diagram to finite
#         regions.

#         Parameters
#         ----------
#         vor : Voronoi
#             Input diagram
#         radius : float, optional
#             Distance to 'points at infinity'.

#         Returns
#         -------
#         regions : list of tuples
#             Indices of vertices in each revised Voronoi regions.
#         vertices : list of tuples
#             Coordinates for revised Voronoi vertices. Same as coordinates
#             of input vertices, with 'points at infinity' appended to the
#             end.

#         """

#         if vor.points.shape[1] != 2:
#             raise ValueError("Requires 2D input")

#         new_regions = []
#         new_vertices = vor.vertices.tolist()

#         center = vor.points.mean(axis=0)
#         if radius is None:
#             radius = vor.points.ptp().max()

#         # Construct a map containing all ridges for a given point
#         all_ridges = {}
#         for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
#             all_ridges.setdefault(p1, []).append((p2, v1, v2))
#             all_ridges.setdefault(p2, []).append((p1, v1, v2))

#         # Reconstruct infinite regions
#         for p1, region in enumerate(vor.point_region):
#             vertices = vor.regions[region]

#             if all(v >= 0 for v in vertices):
#                 # finite region
#                 new_regions.append(vertices)
#                 continue

#             # reconstruct a non-finite region
#             ridges = all_ridges[p1]
#             new_region = [v for v in vertices if v >= 0]

#             for p2, v1, v2 in ridges:
#                 if v2 < 0:
#                     v1, v2 = v2, v1
#                 if v1 >= 0:
#                     # finite ridge: already in the region
#                     continue

#                 # Compute the missing endpoint of an infinite ridge

#                 t = vor.points[p2] - vor.points[p1] # tangent
#                 t /= np.linalg.norm(t)
#                 n = np.array([-t[1], t[0]])  # normal

#                 midpoint = vor.points[[p1, p2]].mean(axis=0)
#                 direction = np.sign(np.dot(midpoint - center, n)) * n
#                 far_point = vor.vertices[v2] + direction * radius

#                 new_region.append(len(new_vertices))
#                 new_vertices.append(far_point.tolist())

#             # sort region counterclockwise
#             vs = np.asarray([new_vertices[v] for v in new_region])
#             c = vs.mean(axis=0)
#             angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
#             new_region = np.array(new_region)[np.argsort(angles)]

#             # finish
#             new_regions.append(new_region.tolist())

#         return new_regions, np.asarray(new_vertices)

#     shape = domain_ROIs.shape
#     n = np.unique(domain_ROIs[np.where(~np.isnan(domain_ROIs))]).size

#     roimask_inv = np.zeros(shape, dtype='uint16')
#     roimask_inv[np.where(np.isnan(domain_ROIs))] = 1

#     random_indices = (np.random.random((n,2))*shape).astype('uint16')
#     out_of_bounds = np.array(1)

#     while out_of_bounds.sum() > 0:
#         out_of_bounds = roimask_inv[random_indices[:,0], random_indices[:,1]]
# #         print(out_of_bounds.sum())
#         new_random_indices = (np.random.random((out_of_bounds.sum(),2))*shape).astype('uint16')
#         random_indices[np.where(out_of_bounds == 1)] = new_random_indices

#     vor = Voronoi(random_indices)
# #     scipy.spatial.voronoi_plot_2d(vor)
# #     plt.show()

#     new_regions, new_vertices = voronoi_finite_polygons_2d(vor)

# #     for region in new_regions:
# #         polygon = new_vertices[region]
# #         plt.fill(*zip(*polygon), alpha=0.4)

# #     plt.plot(random_indices[:,0], random_indices[:,1], 'ko')
# #     plt.xlim(vor.min_bound[0] - 0.1, vor.max_bound[0] + 0.1)
# #     plt.ylim(vor.min_bound[1] - 0.1, vor.max_bound[1] + 0.1)

# #     plt.show()

#     voronoi_map = np.zeros(shape)
#     voronoi_map[:] = np.nan

#     for i, region in enumerate(new_regions):

#         polygon = new_vertices[region]
#         roi = np.zeros(shape, dtype='uint8')
#         cv2.fillPoly(roi,
#                      [polygon[:,::-1].astype('int32').reshape((-1,1,2))]
#                      ,True,1)
#         roi[np.where(roimask_inv == 1)] = 0

#         voronoi_map[np.where(roi == 1)] = i

#     return voronoi_map

# def getGridROIs(domain_ROIs, max_difference=10, method='static'):

#     print('\nMaking Grid ROIs\n-----------------------')

#     def nanunique(A):
#         return np.unique(A[np.where(~np.isnan(A))])

#     def calculateDSGridROIs(domain_ROIs, factor, method='static'):

#         if method == 'downsample':

#             # fill gaps in domains
#             binary_ROIs = (~np.isnan(domain_ROIs)).astype('uint8')
#             binary_ROIs_filled = cv2.morphologyEx(binary_ROIs, cv2.MORPH_CLOSE, np.ones((15,15), np.uint8))
#             domain_ROIs[np.where((binary_ROIs_filled == 1) & np.isnan(domain_ROIs))] = 1

#             # crops corner pieces with NaNs
#             vidROIs = np.tile(domain_ROIs[:,:,None],3)
#             vidROIs = vidROIs.swapaxes(0,2).swapaxes(1,2)
#             output = rescale(vidROIs, factor, 1, verbose=False)
#             output = output[0]
#             print('grid domains after compression:', (~np.isnan(output)).astype('uint8').sum())

#         elif method == 'static':
#             # small fractional domains on edges
#             target_shape = domain_ROIs.shape
#             rescale_factor = np.ceil(np.array(target_shape).max() / factor).astype('int')

#             small_grid = np.arange(rescale_factor**2).reshape((rescale_factor,rescale_factor))

#             intermediary = np.tile(small_grid[:,:,None],factor).reshape((small_grid.shape[0], rescale_factor*factor))
#             output = np.tile(intermediary[:,:,None],factor).swapaxes(1,2).reshape((rescale_factor*factor,rescale_factor*factor))
#             output = output[:target_shape[0], :target_shape[1]].astype('float64')
#             output[np.where(np.isnan(domain_ROIs))] = np.nan
#             print('grid domains after compression', nanunique(output.flat).size)

#         else:
#             print('method undefined!', method)
#             output = np.nan

#         return output

#     target_domains = nanunique(domain_ROIs).size
#     print('unique domains target:', target_domains)

#     factor = 50
#     in_range = False
#     attempts = 0

#     assert method in ['downsample', 'static'], 'method undefined! {0}'.format(method)

#     # loop and adjust reduction factor
#     while not in_range:
#         attempts += 1

#         output = calculateDSGridROIs(domain_ROIs, factor, method)

#         output_domains = nanunique(output).size

#         if (output_domains > target_domains) & (output_domains - target_domains < max_difference):
#             print('target reached')
#             in_range = True
#         elif (output_domains > target_domains) & (output_domains - target_domains > max_difference):
#             print('too many output domains')
#             factor += 10
#         elif (output_domains < target_domains):
#             print('too few domains')
#             factor -= 1

#         assert (attempts < 50), 'equilibrium not possible!'

#     if method == 'downsample':
#         gridind = np.where(~np.isnan(output))
#         output[gridind] = np.arange(gridind[0].size)

#         newshape = [item * factor for item in output.shape]

#         intermediary = np.tile(output[:,:,None],factor).reshape((output.shape[0], newshape[1]))
#         grid_ROIs = np.tile(intermediary[:,:,None],factor).swapaxes(1,2).reshape(newshape)

#         # pad with nan to reach original shape
#         empty_ROIs = np.empty(domain_ROIs.shape)
#         empty_ROIs[:] = np.nan
#         empty_ROIs[:grid_ROIs.shape[0], :grid_ROIs.shape[1]] = grid_ROIs
#         grid_ROIs = empty_ROIs
#     else:
#         grid_ROIs = output

#     return grid_ROIs


def rolling_mosaic_movie(domain_ROIs,
                         ROI_timecourses,
                         savepath,
                         resize_factor=1,
                         codec=None,
                         speed=1,
                         fps=10,
                         cmap='default',
                         t_start=None,
                         t_stop=None,
                         n_rotations=0):

    print('\nWriting Rolling Mosiac Movie\n-----------------------')

    if cmap == 'default':
        cmap = DEFAULT_COLORMAP

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
        if cmap is not None:
            frame = cv2.applyColorMap(frame, cmap)
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
    save_colorbar(scale, cbarpath, colormap=cmap)

    return


def mosaic_movie(domain_ROIs,
                 ROI_timecourses,
                 savepath=None,
                 t_start=None,
                 t_stop=None,
                 n_rotations=0,
                 cmap='default'):

    print('\nRebuilding Mosiac Movie\n-----------------------')

    if cmap == 'default':
        cmap = DEFAULT_COLORMAP

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
             colormap=cmap)


# def makeRegionMap(region_components, domain_ROIs, roimask=None, shape=None):

#     if shape is None:
#         assert roimask is not None, 'no shape information provided'
#         shape = roimask.shape

#     regionmap = np.zeros(shape)

#     if roimask is not None:
#         regionmap[np.where(roimask == 0)] = np.nan

#     for i, key in enumerate(region_components):

#         for region in region_components[key]:
#             regionmap[np.where(domain_ROIs == region)] = i + 5
#     return(regionmap)

# def adjustMapIndices(domain_ROIs, verbose=False):

#     if verbose:
#         print('before unique indices:', np.unique(domain_ROIs[~np.isnan(domain_ROIs)]).size)
#         print('before nanmax:', np.nanmax(domain_ROIs))

#     if np.nanmin(domain_ROIs) > 0:
#         domain_ROIs -= np.nanmin(domain_ROIs)

#     domain_offset = np.diff(np.unique(domain_ROIs[~np.isnan(domain_ROIs)]))

#     adjust_indices = np.where(domain_offset > 1)[0]

#     for i in adjust_indices:
#         domain_ROIs[np.where(domain_ROIs > i+1)] -= (domain_offset[i]) - 1

#     domain_offset = np.diff(np.unique(domain_ROIs[~np.isnan(domain_ROIs)]))

#     if verbose:
#         print('after unique indices:', np.unique(domain_ROIs[~np.isnan(domain_ROIs)]).size)
#         print('after nanmax:', np.nanmax(domain_ROIs))

#     return(domain_ROIs)

# def removeOBs(domain_ROIs, verbose=False):
#     baseindex = np.where(np.isnan(domain_ROIs).sum(axis=1) \
#                          < np.isnan(domain_ROIs).shape[1])[0][0]
#     cutoff = np.where(np.isnan(domain_ROIs).sum(axis=1)[baseindex:] \
#                       == np.isnan(domain_ROIs).shape[1])[0][0]

#     return adjustMapIndices(domain_ROIs[:cutoff], verbose=verbose)

# def removeIndices(domain_ROIs, indices, adjust_indices=True):
#     for index in indices:
#         domain_ROIs[np.where(domain_ROIs == index)] = np.nan

#     if adjust_indices:
#         domain_ROIs = adjustMapIndices(domain_ROIs)
#     return domain_ROIs
