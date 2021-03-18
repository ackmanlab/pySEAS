def convert_float(A, verbose=True):
    Amin = A.min()
    Amax = A.max()

    f16 = np.finfo('float16')
    f32 = np.finfo('float32')

    # if (Amin > f16.min) & (Amax < f16.max):
    #     print('Converting matrix to float16')
    #     A = A.astype('float16', copy=False)
    if (Amin > f32.min) & (Amax < f32.max):
        if verbose:
            print('Converting matrix to float32')
        t0 = timer()
        A = A.astype('float32', copy=False)
        if verbose:
            print('Conversion took {0} sec'.format(timer() - t0))
    else:
        print('Not converting to float32')

    return A
