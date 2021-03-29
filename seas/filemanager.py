#!/usr/bin/env python3
'''
Functions used for loading and finding files for seas analysis.  Loads experiment strings from config.ini defaults

Authors: Sydney C. Weiser
Date: 2017-07-28
'''
import os
import re
import sys
import time
from subprocess import call

from seas.defaults import config

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
    matchstr = config['expstring']['single_experiment'] + config['filestrings']['movies']

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


def experiment_sorter(folder_path, experimentstr, verbose=True):
    '''
    Finds all files associated with an experiment in a particular folder, 
    organizes them by filetype: movie files, processed files, metadata files.
    '''
    assert os.path.isdir(folder_path), 'Folder input was not a valid directory'

    # experimentstr must have a specific file format

    if re.match(config['expstring']['experiment_span'], experimentstr) is not None:
        print('matching multiple experiments')
        match = re.match(config['expstring']['experiment_span'], experimentstr)
        groups = match.groups()

        experimentlist = [groups[0]+'_{:02d}'.format(i) \
                for i in range(int(groups[1]), int(groups[2])+1)]
    else:
        assert re.match(config['expstring']['single_experiment'], experimentstr) is not None, \
            'experimentstr input was not a valid YYMMDD_EE experiment name'
        experimentlist = [experimentstr]

    files = os.listdir(folder_path)
    if verbose:
        print("all matching found in folder '{0}':".format(folder_path))

    exp = dict()
    for key in config['filestrings']:
        exp[key] = []

    for experimentstr in experimentlist:
        for file in files:
            filepath = os.path.join(folder_path, file)

            if verbose:
                if re.match(experimentstr, file):
                    print('\t', file)


            for key in config['filestrings']:
                if re.match(experimentstr + config['filestrings'][key], file, re.IGNORECASE):
                    exp[key].append(filepath)
                    continue

        exp['movies'] = movie_sorter(exp['movies'], verbose=False)[experimentstr]


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
        assert re.match(config['expstring']['single_experiment'], experimentstr) is not None, \
            'experimentstr input was not a valid YYMMDD_EE experiment name'
    else:
        experimentstr = config['expstring']['single_experiment']

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
        experimentstr = config['expstring']['single_experiment']

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
