import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from numpy.core.fromnumeric import _clip_dispatcher
from scipy.signal import convolve2d as conv2
import skimage
import skimage.filters

def normalize(x, vmin=None, vmax=None, clip=True):
    if vmin is None:
        vmin = np.nanmin(x)
    if vmax is None:
        vmax = np.nanmax(x)
    nrm = Normalize(vmin=vmin, vmax=vmax, clip=clip)
    return nrm(x)


def normalize_posneg(x):
    return x / np.nanmax(np.abs(x))


def falloff_fn(x, exponent=-1.2):
    """falloff function for x = 0-1
    """
    # return (x+1)**-12 # Not steep enough
    x = x**exponent - 1
    out = normalize(x)
    return out


def low_freq_noise(size, falloff_fn=falloff_fn):
    """Make low-frequency noise"""
    if isinstance(size, tuple):
        xd, yd = size
    else:
        xd = yd = size
    x, y = np.meshgrid(np.linspace(-1, 1, xd), np.linspace(-1, 1, yd))
    C = (x**2 + y**2)**0.5
    C[C == 0] = np.min(C[C != 0])
    r = np.random.rand(yd, xd)
    f = np.fft.fftshift(np.fft.fft2(r))
    fr = f * falloff_fn(C)
    rr = np.fft.ifft2(np.fft.ifftshift(fr))
    return rr.real


def grating(size, freq=10, ori=0, phi=0):
    """Make a grating image.

    Parameters
    ----------
    size : scalar or tuple (size)
        Size of grating (in pixels, optionally (x, y) diff sizes)
    freq : scalar
        Frequency of grating in cycles/image
    ori : scalar
        Orientation of grating in degrees
    phi : scalar
        Phase of grating in radians
    """
    if isinstance(size, (list, tuple)):
        xd, yd = size
    else:
        xd = yd = size
    x, y = np.meshgrid(np.linspace(-np.pi, np.pi, xd), np.linspace(-np.pi, np.pi, yd))
    theta = np.radians(ori)
    a = np.cos(theta)
    b = np.sin(theta)
    grat = np.cos(freq * (a * -x + b * y) + phi)
    return grat


def make_gauss_2d(x_grid, y_grid, x_mu, y_mu, x_std, y_std, offset=0, height=1, theta=0):
    """Make a 2-d gaussian

    Parameters
    ----------
    x_grid : array
        X grid over which to define Gaussian
    y_grid : array
        Y grid over which to define Gaussian
    x_mu : scalar
        X mean of Gaussian
    y_mu : scalar
        Y mean of Gaussian
    offset : scalar
        no idea
    height : scalar
        amplitude of gaussian
    theta : scalar 
        angle of Gaussian

    """
    if height is None:
        height = 1 / (2 * np.pi * x_std * y_std)
    a = np.cos(np.deg2rad(theta))**2/2/x_std**2 + np.sin(np.degrees(theta))**2/2/y_std**2
    b = -np.sin(np.degrees(2*theta))/4/x_std**2 + np.sin(np.degrees(2*theta))/4/y_std**2
    c = np.sin(np.degrees(theta))**2/2/x_std**2 + np.cos(np.deg2rad(theta))**2/2/y_std**2
    g = offset + height * np.exp( -(a * (x_grid - x_mu)**2 + 2 * b * (x_grid - x_mu) * (y_grid - y_mu)  + c * (y_grid - y_mu)**2))
    return g


def make_ori_noise(size, oris=4, sf_range=None, sm=7, noise_frames=24, phi=None, 
    tf_range=None, speed_range=None, contrast_range=None, integer_tfs=False, ):
    """Make oriented background noise.

    Parameters
    ----------
    size : scalar or (x, y) tuple
        size of image to be generated
    oris : scalar or array-like
        number of oriented components to be incorporated, or list/array of 
        specific orientations
    sf_range : tuple
        Allowable spatial frequency range for gratings - tuple is (min, max)
    sm : scalar
        size in pixels of smoothing kernel for blending grating layers together
    phi : scalar or list
        phase of gratings. If a list is given, a movie (with one frame per phi) will be returned.
    tf_range : tuple or list
        tuple (of length 2) specifies range of allowable temporal frequencies (in cycles per `noise_frames`)
        list (of length (n)) chooses specific temporal frequencies for each orientation
    speed_range : tuple or list
        tuple (of length 2) specifies range of allowable speeds (in fractions of image per second)
        list (of length (n)) chooses specific speeds for each orientation

    """
    # Input handling
    # Orientations
    if isinstance(oris, (tuple, list, np.ndarray)):
        n_oris = len(oris)
    else:
        n_oris = copy.copy(oris)
        # Evenly spaced 
        oris = np.linspace(0, 360, n_oris, endpoint=False)
        # or random?
        #oris = np.random.randint(low=0, high=360, size=(n_oris,))
    
    # Spatial frequencies
    if sf_range is None:
        sf_range = [7,8]
    else:
        if isinstance(sf_range, tuple):
            # range is specified
            assert len(sf_range) == 2, '`sf_range` specified as a tuple must be length 2!'
            sfs = np.random.randint(low=sf_range[0], high=sf_range[1], size=(n_oris,))
        else:
            sfs = sf_range
            assert len(sfs)==n_oris, 'Number of orientations must match number of spatial frequencies!'
    
    # Temporal frequencies or speed
    chk = [(phi is not None), (tf_range is not None), (speed_range is not None)]
    if sum(chk) == 0:
        phi = 0
    elif sum(chk) > 1:
        raise ValueError('Only one input of `phi`, `tf_range`, and `speed_range` can be defined!')
    if tf_range is not None:
        if isinstance(tf_range, tuple):
            # Define phi
            assert len(tf_range) == 2, '`tf_range` specified as a tuple must be length 2!'
            if integer_tfs:
                tfs = np.random.randint(low=tf_range[0], high=tf_range[1], size=(n_oris,))
            else:
                tfs = np.random.uniform(
                    low=tf_range[0], high=tf_range[1], size=(n_oris,))
        else:
            tfs = tf_range
            assert len(tfs)==n_oris, 'Number of orientations must match number of temporal frequencies!'
        phi = [np.linspace(0, tf * 2 * np.pi, noise_frames) for tf in tfs]
    elif speed_range is not None:
        if isinstance(speed_range, tuple):
            assert len(speed_range) == 2, '`speed_range` specified as a tuple must be length 2!'
            # Define phi
            speeds = np.random.uniform(low=speed_range[0], high=speed_range[1], size=(n_oris,))
        else:
            speeds = speed_range
            assert len(speeds)==n_oris, 'Number of orientations must match number of speeds!'
        # speed = tf / sf, so tf = speed / sf
        tfs = [np.ceil(x) for x in [speed / sf for speed, sf in zip(speeds, sfs)]]
        if integer_tfs:
            tfs = [np.ceil(x) for x in tfs]
        phi = [np.linspace(0, tf * 2 * np.pi, noise_frames) for tf in tfs]

    if not isinstance(phi, (tuple, list, np.ndarray)):
        phi = [[phi]] * n_oris

    if contrast_range is None:
        grating_contrasts = [1] * n_oris
    else:
        grating_contrasts = np.random.uniform(low=contrast_range[0], high=contrast_range[1], size=(n_oris,))
    # Get low-frequency noise masks for gratings
    lf = np.dstack([low_freq_noise(size) for x in range(n_oris)])
    # Get gratings
    gr = []
    #rfreqs = np.random.randint(low=frange[0], high=frange[1], size=(n_oris,))
    for o, sf, p, contrast in zip(oris, sfs, phi, grating_contrasts):
        tmpg = np.dstack([normalize_posneg(grating(size, ori=o, freq=sf, phi=p_)) for p_ in p])
        gr.append(normalize(tmpg[:, :, np.newaxis, :] * contrast, vmin=-1, vmax=1))
    # dimensions of `gratings` are [x, y, ori+freq, phi]
    gratings = np.concatenate(gr, axis=2)
    mx = np.argmax(lf, axis=2)
    thresh_lf_noise = np.dstack((np.double(mx == i) for i in range(n_oris)))
    lf_alpha = np.dstack([skimage.filters.gaussian(tn.T, sigma=sm) for tn in thresh_lf_noise.T])
    lf_alpha = lf_alpha[..., np.newaxis]
    noise = np.squeeze(np.sum(lf_alpha * gratings, axis=2))
    # Fix me above, maybe. Or not. 
    noise = np.moveaxis(noise, -1, 0)
    return noise

def make_color_array(n_colors, size=(100,100),):
    carray = plt.cm.hsv(np.linspace(0,1,n_colors + 1))[:-1]
    out = np.array([col[np.newaxis, np.newaxis, :] * np.ones(size + (carray.shape[1],)) 
                    for col in carray])
    return out
    
def _apply_alpha(data, alpha):
    """Apply alpha to data

    Limited use: sums all 1st dim of data (assumed to be 3D arrays of color)
    after multiplying by rows of last dim in alpha

    Assumes all last dim of alpha sums to 1"""
    out = np.array([d * np.atleast_3d(a.T) for d, a in zip(data, alpha.T)])
    return out.sum(0)
