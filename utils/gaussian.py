import numpy as np
from scipy import signal

def gaussian_kernel(n, std, normalised=False):
    '''
    Generates a n x n matrix with a centered gaussian
    of standard deviation std centered on it. If normalised,
    its volume equals 1.'''
    gaussian1D = signal.gaussian(n, std)
    gaussian2D = np.outer(gaussian1D, gaussian1D)
    if normalised:
        gaussian2D /= (2*np.pi*(std**2))
    return gaussian2D