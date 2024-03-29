#!/usr/bin/env python3
'''
Tools for standardized saving/loading a class or dictionary to a .hdf5 file.
Strings are saved as attributes of the file; lists of strings are saved as tab 
delimited strings; arrays are saved as datasets.  Dicts are saved as a new folder, 
with data saved as numpy datasets.  Other objects are saved as pickle dumps.

Useage:
    * listing objects in an hdf5 file:
        f = hdf5manager(mypath)
        f.print()
    * saving data to file:
        f = hdf5manager(mypath)
        f.save(mydict)
      OR:
        f.save(myClass)
    * loading data from file:
        f = hdf5manager(mypath)
        data = f.load()
        
Authors: Sydney C. Weiser
Date: 2017-01-13
'''

import h5py
import numpy as np
import os
import pickle
from typing import List


class hdf5manager:
    '''
    A class for managing saved artifacts in the hdf5 format.
    Only temporarily opens file for read/write operations, to avoid corruption.
    Abstracts away different data types and attributes, to simplify user experience.

    Functions:
        print: Prints all objects in hdf5 file.
        keys: Lists all keys currently in hdf5 file.
        open: Opens file for manual operation.
        close: Closes file after manual operation.
        load: Load a specific object, or all objects, from file.
        delete: Delete a specific object from hdf5 file.
        save: Save data to hdf5 file.
    '''

    def __init__(self, path: str, verbose: bool = False, create: bool = True):
        '''
        Initializes the hdf5 object.
        Set the hdf5 path, and whether the file session should be verbose.
        If create=False, the file will not be created if it doesn't already exist.

        Args:
            path: Where the file should be created at, or opened from.
            verbose: Whether the function should produce verbose output.
            create: Whether the file should be created, if it doesn't already exist.
        '''

        assert (path.endswith('.hdf5') | path.endswith('.mat'))
        path = os.path.abspath(path)

        if not os.path.isfile(path) and create:
            # Create the file.
            print('Creating file at:', path)
            f = h5py.File(path, 'w')
            f.close()
        else:
            assert os.path.isfile(path), 'File does not exist'

        self.path = path
        self.verbose = verbose

        if verbose:
            self.print()

    def print(self) -> None:
        '''
        Prints the contents of the file.

        Arguments:
            None

        Returns:
            None
        '''
        path = self.path
        print()

        # If not saving or loading, open the file to read it
        if not hasattr(self, 'f'):
            print('Opening File to read...')
            f = h5py.File(path, 'r')
        else:
            f = self.f

        if len(list(f.keys())) > 0:
            print('{0} has the following keys:'.format(path))
            for file in f.keys():
                print('\t-', file)
        else:
            print('{0} has no keys.'.format(path))

        if len(list(f.attrs)) > 0:
            print('{0} has the following attributes:'.format(path))
            for attribute in f.attrs:
                print('\t-', attribute)
        else:
            print('{0} has no attributes.'.format(path))

        # If not saving or loading, close the file after finished
        if not hasattr(self, 'f'):
            print('Closing file...')
            f.close()
        print()

    def keys(self) -> List[str]:
        '''
        List all keys within the file.

        Arguments:
            None

        Returns:
            keys: All keys present with in the file.
        '''
        # If not saving or loading, open the file to read it.
        if not hasattr(self, 'f'):
            f = h5py.File(self.path, 'r')
        else:
            f = self.f

        keys = [key for key in f.attrs]
        keys.extend([key for key in f.keys()])

        if not hasattr(self, 'f'):
            f.close()

        return keys

    def open(self) -> None:
        '''
        Open the object for manual access.  
        This takes away safety / anti-corruption features, 
        and should be avoided if possible.

        The raw file can be now accessed using h5py functionsd at self.f.

        Arguments:
            None

        Returns:
            None
        '''
        path = self.path
        verbose = self.verbose

        f = h5py.File(path, 'a')
        self.f = f

        self.print()  # print all variables

        if verbose:
            print(
                'File is now open for manual accessing.\n'
                'To access a file handle, assign hdf5manager.f.[key] to a handle'
                ' and pull slices: \n'
                '\t slice = np.array(handle[0,:,1:6])\n'
                'It is also possible to write to a file this way\n'
                '\t handle[0,:,1:6] = np.zeros(x,y,z)\n')

    def close(self):
        '''
        Close the file after manual access.

        Arguments:
            None

        Returns:
            None
        '''
        self.f.close()
        del self.f

    def load(self, target: List[str] = None, ignore: List[str] = None) -> dict:
        '''
        Load a specific target, or all object from the hdf5 file.

        Arguments:
            target: Which object(s) to load from the file.
            ignore: The object(s) to ignore when loading.

        Returns:
            data: The data loaded from the file.
        '''
        path = self.path
        verbose = self.verbose

        def loadDict(f, key):
            # Load dict to key from its folder
            if verbose:
                print('\t\t-', 'loading', key, 'from file...')
            g = f[key]

            if verbose:
                print('\t\t-', key, 'has the following keys:')
            if verbose:
                print('\t\t  ', ', '.join([gkey for gkey in g.keys()]))

            data = {}
            if g.keys().__len__() > 0:
                for gkey in g.keys():
                    if type(g[gkey]) is h5py.Group:
                        data[gkey] = loadDict(g, gkey)
                    elif type(g[gkey]) is h5py.Dataset:
                        if verbose:
                            print('\t\t-', 'loading', key, 'from file...')
                        data[gkey] = np.array(g[gkey])
                    else:
                        if verbose:
                            print('key was of unknown type', type(gkey))

            if verbose:
                print('\t\t-', key, 'has the following attributes:')
            if verbose:
                print('\t\t  ', ', '.join([gkey for gkey in g.attrs]))

            for gkey in g.attrs:
                if verbose:
                    print('\t\t\t', gkey + ';', type(g.attrs[gkey]).__name__)
                if verbose:
                    print('\t\t\t-', 'loading', gkey, 'from file...')
                if type(g.attrs[gkey]) is str:
                    data[gkey] = g.attrs[gkey]
                elif type(g.attrs[gkey] is np.void):
                    out = g.attrs[gkey]
                    data[gkey] = pickle.loads(out.tobytes())
                else:
                    print('INVALID TYPE:', type(g.attrs[gkey]))

            return data

        f = h5py.File(path, 'a')  # Open file for access
        self.f = f  # set to variable so other functions know file is open

        if target is None:
            if verbose:
                print('No target key specified; loading all datasets')
            keys = f.keys()
            attrs = f.attrs
        else:
            assert (type(target) is str) or (type(target) is
                                             list), 'invalid target'
            if type(target) is str:
                target = [target]

            keys = []
            attrs = []

            for item in target:

                if (type(item) is str) & (item in f.keys()):
                    if verbose:
                        print('Target key found:', item)
                    keys.append(item)

                elif (type(item) is str) & (item in f.attrs):
                    if verbose:
                        print('Target attribute found:', item)
                    attrs.append(item)

                else:
                    print('Target was not valid:', item)

        if verbose:
            print('\nLoading datasets from hdf5 file:')
        data = {}
        for key in keys:
            if verbose:
                print('\t', key + ';', type(f[key]).__name__)

            if key == ignore:
                if verbose:
                    print('\t\t- ignoring key:', key)
            else:
                if type(f[key]) is h5py.Group:
                    data[key] = loadDict(f, key)
                elif type(f[key]) is h5py.Dataset:
                    if verbose:
                        print('\t\t-', 'loading', key, 'from file...')

                    if f[key].dtype.type is np.void:
                        data[key] = pickle.loads(np.array(f[key]).tobytes())
                    else:
                        data[key] = np.array(f[key])
                else:
                    if verbose:
                        print('\t\t- attribute was unsupported type:',
                              type(f[key]).__name__)

        for key in attrs:
            if verbose:
                print('\t', key + ';', type(f.attrs[key]).__name__)

            if key == ignore:
                if verbose:
                    print('ignoring attribute:', key)
            else:
                if verbose:
                    print('\t\t-', 'loading', key, 'from file...')
                if type(f.attrs[key]) is str:
                    data[key] = f.attrs[key]
                elif type(f.attrs[key] is np.void):
                    out = f.attrs[key]
                    data[key] = pickle.loads(out.tobytes())

        if verbose:
            print('Keys extracted from file:')
        if verbose:
            print('\t', ', '.join([key for key in data.keys()]))
        if verbose:
            print('\n\n')

        del self.f
        f.close()

        if (type(target) is list) and (len(target) == 1):
            data = data[target[0]]

        return data

    def delete(self, target: List[str]) -> None:
        '''
        Deletes a specific objects from the hdf5 file.

        Arguments:
            target: Which object(s) to delete from the file.

        Returns:
            None
        '''
        if type(target) is not list:
            target = [target]

        f = h5py.File(self.path, 'a')  # Open file for access
        self.f = f  # set to variable so other functions know file is open
        verbose = self.verbose

        for key in target:
            if key in self.keys():
                if verbose:
                    print('key found:', key)

                try:
                    del f[key]
                except:
                    del f.attrs[key]
            else:
                if verbose:
                    print('key not found:', key)

        del self.f
        f.close()

    def save(self, data: dict) -> None:
        '''
        Saves any data to the hdf5 file.
        Uses various underlying save mechanisms, 
        depending on the appropriate storage for the data type.
        If an appropriate data type is not supported, the object will be pickle dumped to file.
        In this case, be careful when loading that you have the same python libraries 
        inported as you did when the object was initially created.
        Nested dictionaries are supported, and save within hdf5 as a nested object.

        Arguments:
            data: The dictionary of data to be saved to the file.

        Returns:
            None
        '''

        # Data is a class file or dict of keys/data.
        path = self.path
        verbose = self.verbose

        # Define functions to save each type of data:
        # -------------------------------------------

        def save_dict(f, fdict, key):
            # Write dict to key as its own folder
            if verbose:
                print('\t\t-', 'writing', key, 'to file...')

            # Delete if it exists.
            if key in f:
                if verbose:
                    print('\t\t-', 'Removing', key, 'from file')
                del f[key]

            g = f.create_group(key)
            data_d = fdict

            for dkey in fdict:

                if (type(fdict[dkey]) is str):
                    save_string(g, fdict[dkey], dkey)
                elif type(fdict[dkey]) is np.ndarray:
                    save_array(g, fdict[dkey], dkey)
                elif type(fdict[dkey]) is dict:
                    save_dict(g, fdict[dkey], dkey)
                else:
                    if verbose:
                        print('\t\t- attribute was unsupported type:',
                              type(fdict[dkey]).__name__)
                    if verbose:
                        print('\t\tAttempting to save pickle dump of object')
                    try:
                        save_other(g, fdict[dkey], dkey)
                        if verbose:
                            print('\t\tSaved succesfully!')
                    except:
                        if verbose:
                            print('\t\tFailed..')

            if verbose:
                print('\t\t-', key, 'has the following keys:')
            if verbose:
                print('\t\t  ', ', '.join([dkey for dkey in g.keys()]))

            if verbose:
                print('\t\t-', key, 'has the following attributes:')
            if verbose:
                print('\t\t  ', ', '.join([dkey for dkey in g.attrs]))

        def save_string(f, string, key):
            # Write all strings as attributes of the dataset.
            if verbose:
                print('\t\t-', 'writing', key, 'to file...')
            f.attrs[key] = string

        def save_array(f, array, key):
            # Check if key exists, and if entry is the same as existing value.
            if key in f.keys():
                if (not np.array_equal(array, f[key])):
                    if verbose:
                        print(
                            '\t\t-', key, 'in saved file is inconsistent '
                            'with current version')
                    if verbose:
                        print('\t\t-', 'deleting', key, 'from file')
                    del f[key]
                    if verbose:
                        print('\t\t-', 'writing', key, 'to file...')
                    f.create_dataset(key, data=array, chunks=None)
                else:
                    if verbose:
                        print(
                            '\t\t-', key, 'in saved file is the same as '
                            'the current version')
            else:
                if verbose:
                    print('\t\t-', 'writing', key, 'to file...')
                f.create_dataset(key, data=array, chunks=None)

        def save_other(f, obj, key):
            # Compress to bytestring using pickle, save similar to string
            # Write all strings as attributes of the dataset.
            if verbose:
                print('\t\t-', 'writing', key, 'to file...')

            bstring = np.void(pickle.dumps(obj))
            try:
                f.attrs[key] = bstring
            except RuntimeError:
                if verbose:
                    print('\t\t\tEncountered RuntimeError')
                if verbose:
                    print('\t\t\tSaving pickle dump as data...')
                if key in f.keys():
                    if verbose:
                        print('Deleting previous copy of', key)
                    del f[key]
                f[key] = bstring

        # Check input data type and open file.
        if type(data) is not dict:
            # Get dictionary of all keys in class type.
            data = data.__dict__

        if verbose:
            print('Attributes found in data file:')
            for key in data.keys():
                print('\t', key, ':', type(data[key]))

        f = h5py.File(path, 'a')
        self.f = f

        if verbose:
            self.print()

        # Loop through keys and save them in hdf5 file.
        if verbose:
            print('\nSaving class attributes:')
        for key in data.keys():
            if verbose:
                print('\t', key + ';', type(data[key]).__name__)
            if (type(data[key]) is str):
                save_string(f, data[key], key)
            elif type(data[key]) is np.ndarray:
                save_array(f, data[key], key)
            elif type(data[key]) is dict:
                save_dict(f, data[key], key)
            else:
                if verbose:
                    print('\t\t- attribute was unsupported type:',
                          type(data[key]).__name__)
                if verbose:
                    print('\t\tAttempting to save pickle dump of object')
                try:
                    save_other(f, data[key], key)
                    if verbose:
                        print('\t\tSaved succesfully!')
                except:
                    print('\t\tFailed..')
                    print('\t\t\t', key, 'did not save')
                    print('\t\t\t', 'type:', type(data[key]).__name__)

        if verbose:
            print('All info saved to hdf5 file:')
            self.print()

        del self.f
        f.close()
