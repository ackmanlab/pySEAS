import sys

sys.path.append('..')

from seas.hdf5manager import hdf5manager
import argparse


def main():
    '''
    If called directly from command line, take argument passed, and try to read 
    contents if it's an .hdf5 file.
    '''

    print('\nHDF5 Manager\n-----------------------')

    ap = argparse.ArgumentParser()
    ap.add_argument('file',
                    type=argparse.FileType('r'),
                    nargs='+',
                    help='path to the hdf5 file(s)')
    ap.add_argument('-e',
                    '--extract',
                    type=str,
                    nargs='+',
                    help='key(s) to be extracted')
    ap.add_argument('-m',
                    '--merge',
                    type=argparse.FileType('r'),
                    help='merges keys in merge file into main file(s).')
    ap.add_argument('-c',
                    '--copy',
                    action='store_true',
                    help='make copy of hdf5 file')
    ap.add_argument('-d', '--delete', type=str, nargs='+', help='delete key')
    ap.add_argument('-r', '--rename', type=str, nargs=2, help='rename key')
    ap.add_argument('-i',
                    '--ignore',
                    type=str,
                    help='key to ignore while loading.  For use with copy.')
    ap.add_argument('--read',
                    type=str,
                    nargs='+',
                    help='key(s) to read to terminal.')

    args = vars(ap.parse_args())

    if len(args['file']) == 1:
        path = args['file'][0].name

        assert path.endswith('.hdf5'), 'Not a valid hdf5 file.\nExiting.\n'

        print('Found hdf5 file:', path)
        f = hdf5manager(path, verbose=True)
        # f.print()

        if args['extract'] is not None:
            print('extracting keys:', ', '.join(args['extract']), '\n')

            for key in args['extract']:
                assert key in f.keys(), '{0} was not a valid key!'.format(key)

            loaded = f.load(args['extract'])

            if type(loaded) is not dict:
                loaded = {args['extract'][0]: loaded}

            newpath = f.path.replace(
                '.hdf5', '_extract_{0}.hdf5'.format('-'.join(args['extract'])))

            print('new path:', newpath)

            hdf5manager(newpath).save(loaded)

        elif args['merge'] is not None:
            mergepath = args['merge'].name
            assert mergepath.endswith('.hdf5'), 'merge file was not valid'
            print('merging hdf5 file:', mergepath)

            mergedict = hdf5manager(mergepath).load()
            # print(mergedict)

            for key in mergedict.keys():
                print(key)

                if key in f.keys():
                    print('found in key, are you sure you want to merge? (y/n)')

                    loop = True

                    while loop:
                        response = input().lower().strip()
                        if (response == 'y') | (response == 'yes'):
                            loop = False
                            f.save({key: mergedict[key]})
                        elif (response == 'n') | (response == 'no'):
                            print('not saving', key)
                            loop = False
                        else:
                            print('invalid answer!')

                else:
                    print(key, 'not in main file.  No merge conflicts')
                    f.save({key: mergedict[key]})

        elif args['copy']:
            ignore = args['ignore']

            if ignore is not None:
                assert ignore in f.keys(), '{0} not a valid key!'.format(ignore)

            data = f.load(ignore=ignore)

            newpath = f.path.replace('.hdf5', '_copy.hdf5')
            g = hdf5manager(newpath)
            g.save(data)

        elif args['delete']:

            f.delete(args['delete'])

            print('Note: deleting keys from hdf5 file may not free up space.')
            print('Make a copy with --copy command to free up space.')

        elif args['rename']:
            print('renaming', args['rename'][0], 'to', args['rename'][1], '\n')

            key = args['rename'][0]
            assert key in f.keys(), \
                    'key was not valid: ' + key
            f.verbose = False

            data = f.load(key)
            print('data loaded:', data)

            f.save({args['rename'][1]: data})

            f.open()
            try:
                del f.f[key]
            except:
                del f.f.attrs[key]
            f.close()
            f.print()

        elif args['read']:
            f.verbose = False
            for key in args['read']:
                if key in f.keys():
                    print('key found:', key)
                    print(key + ':', f.load(key))
                else:
                    print('key not found:', key)

        else:
            print('no additional commands found')

    elif len(args['file']) > 1:
        pathlist = [file.name for file in args['file']]

        print('\nFound multiple files:')
        [print('\t', path) for path in pathlist]
        print('')

        if args['extract'] is not None:
            print('extracting keys:', ', '.join(args['extract']), '\n')

            data = load_keys(pathlist, args['extract'])

            directory = os.path.dirname(pathlist[0])
            directory = os.path.abspath(directory)
            filename = 'hdf5extract-' + '-'.join(args['extract']) + '.hdf5'

            path = os.path.join(directory, filename)
            print('path', path)

            f = hdf5manager(path)
            f.save(data)

        elif args['merge'] is not None:
            print('testing!')

            mergepath = args['merge'].name
            assert mergepath.endswith('.hdf5'), 'merge file was not valid'
            print('merging hdf5 file:', mergepath)

            mergedict = hdf5manager(mergepath).load()
            key = list(mergedict.keys())[0]
            # get a random key, test if subdict structure or one key extracted

            if type(mergedict[key]) is dict:
                subdict = True
            else:
                subdict = False
                mergekey = os.path.basename(mergepath).replace(
                    '.hdf5', '').replace('hdf5extract-', '')

            for path in pathlist:
                name = os.path.basename(path).replace('.hdf5',
                                                      '').replace('_ica', '')
                print('Merging file:', name)

                if name in mergedict.keys():
                    f = hdf5manager(path, verbose=False)

                    if subdict:
                        conflict = []
                        for key in mergedict[name]:
                            if key in f.keys():
                                conflict.append(key)

                        if len(conflict) > 0:
                            print('replace old dict with new?')

                            print('\tOriginal:')
                            print('\t', f.load(conflict))
                            print('\n\tNew:')
                            print('\t', mergedict[name])
                            print('\n\treplace?')

                            loop = True
                            while loop:
                                response = input().lower().strip()
                                if (response == 'y') | (response == 'yes'):
                                    loop = False
                                    f.save(mergedict[name])
                                elif (response == 'n') | (response == 'no'):
                                    print('not saving')
                                    loop = False
                                else:
                                    print('invalid answer!')
                        else:
                            f.save(mergedict[name])

                    else:
                        if mergekey in f.keys():
                            print(mergekey, 'was in original file.')
                            print('Are you sure you want to replace this?')

                            print('\tOriginal:')
                            print('\t', f.load(mergekey))
                            print('\n\tNew:')
                            print('\t', mergedict[name])
                            print('\n\treplace?')

                            loop = True
                            while loop:
                                response = input().lower().strip()
                                if (response == 'y') | (response == 'yes'):
                                    loop = False
                                    f.save({mergekey: mergedict[name]})
                                elif (response == 'n') | (response == 'no'):
                                    print('not saving', key)
                                    loop = False
                                else:
                                    print('invalid answer!')
                        else:
                            f.save({mergekey: mergedict[name]})

                else:
                    print(name, 'not found in merge dictionary')
                    print('skipping...')

        elif args['read']:
            for path in pathlist:
                print('\n', path)
                f = hdf5manager(path, verbose=False)
                for key in args['read']:
                    if key in f.keys():
                        print('key found:', key)
                        print(key + ':', f.load(key))
                    else:
                        print('key not found:', key)

        else:
            print('Command not defined for multiple files.')

    else:
        print('No hdf5file found')


def load_keys(pathlist, keys=None):
    '''
    If no keys are passed in, all are loaded
    '''
    if type(pathlist) is str:
        pathlist = [pathlist]
    data = dict()

    for path in pathlist:
        assert os.path.isfile(path), 'File was invalid: {0}'.format(path)
        name = os.path.basename(path).replace('.hdf5', '').replace('_ica', '')
        print('Loading File:', name)

        try:
            f = hdf5manager(path, create=False)
            filedata = f.load(keys)
        except:
            filedata = None

        data[name] = filedata

    return data


if __name__ == '__main__':
    main()
