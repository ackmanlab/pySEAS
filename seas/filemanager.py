#!/usr/bin/env python3
'''
Functions used for loading and finding files for brain analysis.

Authors: Sydney C. Weiser
Date: 2017-07-28
'''
import os
import re
import sys
import time
from subprocess import call
import yaml


def find_files(folder_path,
               match_string,
               subdirectories=False,
               suffix=False,
               regex=False,
               verbose=True):
    '''
    Finds files in folder_path that match a match_string, either at the end of the 
    path if suffix=True, or anywhere if suffix=False.
    Searches subdirectories if subdirectories = True
    '''
    assert os.path.isdir(folder_path), 'Folder input was not a valid directory'
    files = []

    if subdirectories:
        result = os.walk(folder_path)
        for i, f in enumerate(result):
            print(i)
            root, folder, file_list = f
            for file in file_list:
                print(file)
                files.append(os.path.join(root, file))
    else:
        for file in os.listdir(folder_path):
            files.append(os.path.join(folder_path, file))

    if verbose:
        print("all files found in folder '{0}':".format(folder_path))
    results = []

    for filepath in files:
        file = os.path.basename(filepath)
        if os.path.isfile(filepath):
            if verbose:
                print('\t', file)

            if regex:  # search using regular expressions
                if re.match(match_string, file):
                    results.append(filepath)
            else:  # search using string functions
                if suffix:
                    if file.endswith(match_string):
                        results.append(filepath)

                else:
                    if file.find(match_string) >= 0:
                        results.append(filepath)

    if verbose:
        print('matching files found in folder:')
        [print('\t', os.path.basename(file)) for file in results]

    return results


def movie_sorter(pathlist, verbose=True):
    '''
    Takes list of paths, sorts into experiments, and orders files by 
    extension number.  Returns dict of experiments with associated files.
    '''

    n_files = len(pathlist)
    exp_list = []
    fnum_list = []

    # only match movie files that have a specific file format
    matchstr = r'(\d{6}_\d{2})(?:[@-](\d{4}))?\.tif'

    for i, file in enumerate(pathlist):
        name = os.path.basename(file)
        match = re.match(matchstr, name)
        if match is not None:
            exp, fnum = re.match(matchstr, name).groups()
            exp_list.append(exp)
            fnum_list.append(fnum)

    experiments = {}

    for exp in set(exp_list):
        indices = [i for i, exp_i in enumerate(exp_list) if exp == exp_i]

        if len(indices) == 1:
            experiments[exp] = [pathlist[indices[0]]]
        else:
            fnum_set = [fnum_list[i] for i in indices]

            # sort file number extensions by order, get new indices
            for n, fnum in enumerate(fnum_set):
                if fnum is None:
                    fnum_set[n] = 0
                else:
                    fnum_set[n] = int(fnum)
            _, indices_sorted = zip(*sorted(zip(fnum_set, indices)))

            experiments[exp] = [pathlist[i] for i in indices_sorted]

    if verbose:
        print('\nExperiments\n-----------------------')
        for exp in experiments:
            print(exp + ':')
            [print('\t', fname) for fname in experiments[exp]]

    return experiments


def experiment_sorter(folder_path, experimentstr=None, verbose=True):
    '''
    Finds all files associated with an experiment in a particular folder, 
    organizes them by filetype: movie files, processed files, metadata files.
    '''
    assert os.path.isdir(folder_path), 'Folder input was not a valid directory'

    # experimentstr must have a specific file format

    if experimentstr is not None:
        if re.match(r'^\d{6}_\d{2}-\d{2}$', experimentstr) is not None:
            print('matching multiple experiments')
            match = re.match(r'^(\d{6})_(\d{2})-(\d{2})$', experimentstr)
            groups = match.groups()

            experimentlist = [groups[0]+'_{:02d}'.format(i) \
                    for i in range(int(groups[1]), int(groups[2])+1)]
        else:
            assert re.match(r'^\d{6}_\d{2}$', experimentstr) is not None, \
                'experimentstr input was not a valid YYMMDD_EE experiment name'
            experimentlist = [experimentstr]
    else:
        experimentlist = [r'\d{6}_\d{2}']

    files = os.listdir(folder_path)
    if verbose:
        print("all matching found in folder '{0}':".format(folder_path))

    movies = []
    meta = []
    ica = []
    processed = []
    roi = []
    dfof = []
    body = []
    oflow = []
    videodata = []

    for experimentstr in experimentlist:

        movies_unsorted = []

        moviestr = experimentstr + r'(?:[@-](\d{4}))?\.tif'
        metastr = experimentstr + r'_meta\.yaml'
        icastr = experimentstr + r'_(.*)(ica|pca)\.hdf5'
        processedstr = experimentstr + r'_(ica|pca)(.+)\.hdf5'
        roistr = experimentstr + r'_roiset\.zip'
        dfofstr = experimentstr + r'_(\d+x)_dfof\.mp4'
        bodystr = experimentstr + r'_c(\d)-body_cam\.mp4'
        oflowstr = experimentstr + r'_(\w+)OpticFlow\.hdf5'  ######
        videodatastr = experimentstr + r'_videodata\.hdf5'

        for file in files:
            filepath = os.path.join(folder_path, file)

            if verbose:
                if re.match(experimentstr, file):
                    print('\t', file)

            if re.match(moviestr, file, re.IGNORECASE):
                movies_unsorted.append(filepath)
            elif re.match(metastr, file, re.IGNORECASE):
                meta.append(filepath)
            elif re.match(icastr, file, re.IGNORECASE):
                ica.append(filepath)
            elif re.match(processedstr, file, re.IGNORECASE):
                processed.append(filepath)
            elif re.match(roistr, file, re.IGNORECASE):
                roi.append(filepath)
            elif re.match(dfofstr, file, re.IGNORECASE):
                dfof.append(filepath)
            elif re.match(bodystr, file, re.IGNORECASE):
                body.append(filepath)
            elif re.match(oflowstr, file, re.IGNORECASE):
                oflow.append(filepath)
            elif re.match(videodatastr, file, re.IGNORECASE):
                videodata.append(filepath)

        movies.extend(
            movie_sorter(movies_unsorted, verbose=False)[experimentstr])

    exp = {
        'movies': movies,
        'meta': meta,
        'processed': processed,
        'ica': ica,
        'roi': roi,
        'dfof': dfof,
        'body': body,
        'oflow': oflow,
        'videodata': videodata
    }

    if verbose:
        print('Matches:')
        for key in exp:
            if len(exp[key]) > 0:
                print('\t' + key + ':')
                [print('\t\t' + os.path.basename(item)) for item in exp[key]]

    return exp


def sort_experiments(files, experimentstr=None, verbose=True):

    if verbose:
        print('\nSorting Keys\n-----------------------')

    if experimentstr is not None:
        assert re.match(r'\d{6}_\d{2}', experimentstr) is not None, \
            'experimentstr input was not a valid YYMMDD_EE experiment name'
    else:
        experimentstr = r'(\d{6}_\d{2})'

    exps = {}

    for i, file in enumerate(files):
        match = re.match(experimentstr, os.path.basename(file))

        if match is not None:
            exp = match.groups()[0]

            if exp not in exps.keys():
                exps[exp] = [file]
            else:
                exps[exp].append(file)

    if verbose:
        for expname in exps:
            print(expname)
            [print('\t', key) for key in exps[expname]]

    return exps


def get_exp_span_string(experiments):
    # accepts list or keys object, creates string identifier to represent experiments

    if len(experiments) == 1:
        expspanstring = [get_basename(experiment) for experiment in experiments]
        return expspanstring[0]
    else:
        experimentstr = r'(\d{6})_(\d{2})'

    explist = {}

    for exp in experiments:
        match = re.match(experimentstr, exp)
        if match is not None:
            date = match.groups()[0]
            if date not in explist:
                explist[date] = []
            explist[date].append(match.groups()[1])

    [explist[date].sort() for date in explist
    ]  # in-place sort experiment numbers

    explisted = [date + '_' + '-'.join(explist[date]) for date in explist
                ]  # create spans
    expspanstring = '_'.join(explisted)

    return expspanstring


def get_basename(path):

    name = os.path.basename(path)
    name = re.sub(r'(\.)(\w){3,4}$', '', name)  # remove extension
    name = re.sub(r'([@-])(\d){4}', '', name)  # remove @0001 from path

    return name


def read_yaml(path):
    '''
    loads nested dictionaries from .yaml formated files
    '''
    meta = dict()
    with open(path, 'r') as data:
        try:
            meta = yaml.load(data)
        except yaml.YAMLError as exc:
            print(exc)

    return meta
