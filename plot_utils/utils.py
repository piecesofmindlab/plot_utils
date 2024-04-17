# Plotting utils
import matplotlib
import matplotlib.pyplot as plt

from mpl_toolkits import axes_grid1
from mpl_toolkits.mplot3d import Axes3D

from matplotlib.colors import colorConverter, LinearSegmentedColormap, Normalize
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyBboxPatch
from matplotlib import transforms as mtransforms
from matplotlib import animation
from matplotlib import gridspec

import numpy as np
import itertools
import warnings
import subprocess
import six
import os
from IPython.display import display, HTML

### --- helper functions --- ###
# Convenience
rgba = colorConverter.to_rgba_array
def _sind(ang):
    '''Compute sin w/ degree input
    '''
    return np.sin(ang * np.pi/180)

def _cosd(ang):
    '''Compute cos w/ degree input
    '''
    return np.cos(ang * np.pi/180)

def _tand(ang):
    '''Compute tan w/ degree input
    '''
    return np.tan(ang * np.pi/180)

def make_colormap(L, name='tmp', n=256, sName=None, do_plot=False):
    """Wrapper for matplotlib.colors.LinearSegmentedColormap.from_list

    If a path is provided in `save`,
    """
    if isinstance(n, (list, tuple)):
        cmap = LinearSegmentedColormap.from_list(name, zip(n, L))
    else:
        cmap = LinearSegmentedColormap.from_list(name, L, n)
    # Create color map image:
    mxdpi = 300
    cmapim = np.tile(np.linspace(0, 1, mxdpi*2).reshape(1, mxdpi*2), (.4*mxdpi, 1))
    if do_plot:
       plt.matshow(cmapim, cmap=cmap)
    if not sName is None:
       plt.imsave(sName, cmapim, cmap=cmap)
    return cmap

def find_squarish_dimensions(n):
    '''Get row, column dimensions for n elememnts

    Returns (nearly) sqrt dimensions for a given number. e.g. for 23, will
    return [5, 5] and for 26 it will return [6, 5]. For creating displays of
    sets of images, mostly. Always sets x greater than y if they are not
    equal.

    Returns
    -------
    x : int
       larger dimension (if not equal)
    y : int
       smaller dimension (if not equal)
    '''
    sq = np.sqrt(n)
    if round(sq)==sq:
        # if this is a whole number - i.e. a perfect square - return perfect square
        x = sq
        y = sq
        return int(x), int(y)
    # larger value, smaller value than square
    x = [np.ceil(sq), np.ceil(sq)]
    y = [np.ceil(sq), np.floor(sq)]
    # make sure negative values will not be chosen as the minimum
    opt = [x[0] * y[0], x[1] * y[1]]
    test = np.array([o - n for o in opt])
    test[test < 0] = np.inf 
    good_option = np.argmin(test)
    x = x[good_option]
    y = y[good_option]
    return int(x), int(y)

### --- Figures --- ###
def fig_show(fnum):
    from matplotlib.pyplot import get_current_fig_manager
    plt.figure(fnum)
    get_current_fig_manager().window.raise_()

### --- Plotting functions --- ###  
# Better / more user-friendly than matplotlib functions#
def uline(xl=None, yl=None, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    if xl is None:
        xl = ax.get_xlim()
    if yl is None:
        yl = ax.get_ylim()
    mx = max([xl[1], yl[1]])
    mn = min([xl[0], yl[0]])
    ax.plot([mn, mx], [mn, mx], **kwargs)


def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

def make_grad_segs_2pt(p0, p1, npts=20, cmap=None, alph=None):
    """
    Usage: segs, cols = make_grad_segs_2pt(p0, p1, npts=20, cmap=None, alph=None)

    Make a series of segments of different colors between two points p0 and p1.
    This can be used to plot a line that is colored according to a gradient or
    other color map.

    Inputs: 
        p0, p1 : (x, y) points (ONE point at a time here!)
        npts : controls resolution of gradient line
        cmap : LinearSegmentedColormap instance, or tuple of matplotlib colors 
            from which to create a color map (LinearSegmentedColormap.from_list())
        alph : alpha values for the two points (alpha values will blend, too)
    """
    if isinstance(cmap, (tuple, list)):
       cmap = LinearSegmentedColormap.from_list('tmp', [x for x in rgba(cmap)])
    # Segments
    x_1, y_1 = p0
    x_2, y_2 = p1
    X = np.linspace(x_1, x_2, npts+1)
    Xs = X[:-1]
    Xf = X[1:]
    Y = np.linspace(y_1, y_2, npts+1)
    Ys = Y[:-1]
    Yf = Y[1:]
    segs = [[(xs, ys), (xf, yf)] for xs, ys, xf, yf in zip(Xs, Ys, Xf, Yf)]

    # Colors
    C = np.linspace(0, 1, npts)
    cols = cmap(C)
    # Optionally, with alpha values
    if not alph is None:
        a_1, a_2 = alph
        A = np.linspace(a_1, a_2, npts)
        cols[:, 3] = A
    return segs, cols
    
def shade_bg(ax, xtk, w=1, fcol=(.9, .9, .9), yl=None, vert=False, zorder=-1):
    """Shade in xtick grid (every other tick mark is gray/white)"""
    if yl is None:
        yl = plt.ylim()
    if vert:
        # interpret yl as xlim, w as height
        for xf in xtk:
            ax.fill([yl[0], yl[0], yl[1], yl[1]],
                np.array([-w/2., w/2., w/2., -w/2.])+xf,
                color=fcol, edgecolor='none', zorder=zorder)
        plt.xlim(yl)
    else:
        for xf in xtk:
            ax.fill(np.array([-w/2., w/2., w/2., -w/2.])+xf,
                [yl[0], yl[0], yl[1], yl[1]],
                color=fcol, edgecolor='none', zorder=zorder)
        plt.ylim(yl)

def gradient_lines(p0, p1, alph=None, npts=20, cmap=None, ax=None, **kw):
    """
    Draw a gradient lines between p0 and p1 using colormap cmap and alpha
    values in alph.

    Inputs: 
       p0, p1 : sets of (x, y) points to be connected by lines.
         Both should be (nPoints x 2) numpy arrays
       alph : nPoints x 2 array of alpha values for each of the two points
         in p0, p1
       npts : specifies number of segments in gradients
       cmap : matplotlib LinearSegmentedColormap instance, or tuple of color
         values to create a colormap
       ax : handle for axis into which to plot
       kw : dict passed to LineCollection. Set linestyle, linewidth, (labels?)
         etc. with this argument. FOR NOW (?) stick with single values for
         each key rathe rather than arrays

    Inspired by/partly lifted from https://gist.github.com/ivanov/5439438
    """
    segs = []
    cols = []
    if alph is None:
        alph = [(1, 1)]*p0.shape[0]
    if len(cmap) != len(p0):
        cmap = [cmap] * len(p0)
    for xy0, xy1, aa, cm in zip(p0, p1, alph, cmap):
        ss, cc = make_grad_segs_2pt(xy0, xy1, npts=npts, cmap=cm, alph=aa)
        segs+=ss
        cols+=[cc]
    cols = np.vstack(cols)
    #return segs, cols
    LC = LineCollection(segs, colors=cols, **kw)
    if ax is None:
        fig, ax = plt.subplots()
    ax.add_collection(LC)
 
def bar_xw(shape, grp_width=0.8, grp_space=0.05):
    """Gets x locations and widths for bars in a bar plot given data shape, etc

    Assumes that rows in data array (first value in shape) are clusters of bars.

    Parameters
    ----------
    shape : tuple
        shape of array to be plotted
    grp_width : scalar
        max width of bar groups; should be <= 1
    grp_space : scalar
        space between bars in a group; should be << 1
    """
    if len(shape)==1:
        n_grp, n_per_grp = shape[0], 1.
    elif len(shape)==2:
        n_grp, n_per_grp = shape
    else:
        raise ValueError("what do you WANT from me, you MONSTER! I can only deal with 2D data!")
    w = (grp_width-grp_space*(n_per_grp-1.))/n_per_grp
    x = np.arange(1, n_grp+1) # (x) cluster centers
    if n_per_grp > 1:
        x_pos = np.linspace(-grp_width/2. + w/2., grp_width/2. - w/2., n_per_grp)
        x = np.vstack([x+xp for xp in x_pos]).T
    return x, w

def ci(x, a, n=1000, axis=0):
    """Get confidence interval for mean, w/ n bootstrap samples
    
    Parameters
    ----------
    a : scalar
        confidence interval percentile from 0-100 (95 for 95% confidence interval)
        ci should GROW with larger a
    n : scalar, int
        number of resamples of data
    axis : scalar int
        BROKEN. only works for axis = 0. Do not attempt other.
    """
    if axis != 0:
        raise NotImplemnetedError("Can't use axis besides 0, sorry!")
    if isinstance(x, (list, tuple)):
        x = np.array(x)
    if np.ndim(x)==1:
        x = np.reshape(x, (len(x), 1))
    LL = x.shape[0]
    cc = [[np.mean(np.random.choice(xx, LL, replace=True)) for xx in x.T] for c in range(n)]
    cc = np.vstack(cc)
    #return cc
    a = 100.-a
    pct = np.percentile(cc, [a/2., 100-a/2.], axis=axis)
    #return [(low, up) for low, up in pct
    return pct

def ci_errorbars(x, ci, lw=2, color='k', **kwargs):
    """Plot confidence interval error bars

    Plots vertical lines from ci[n, 0] to ci[n, 1] at x[n].
    Only 1-dim for now.

    Parameters
    ----------
    x : array-like, 1D
       n-long vector of x locations to plot lines
    ci : array-like, 2D
       n x 2 array of confidence intervals (lo, hi). 
    lw, col, kwargs: plot parameters.

    Notes
    -----
    Fix to work with mlp.bar a little better! (multiple rows of xs, e.g.)
    """
    h = []
    for xx, ee, ii in zip(x, ci, range(len(x))):
        elo, ehi = ee
        h.append(plt.vlines(xx, elo, ehi, lw=lw, color=color, **kwargs))
    return h

def bar(data, err=None, color=(.5, .5, .5), xw=None, lims=None, hatch=None,
       edgecolor='k', ls='solid', lw=2, ax=None, tklab=None, is_vertical=False, labels=None):
    """Simple bar plot function

    Intended to be an improvement upon / replacement for matplotlib.pyplot.bar
    Has useful defaults. For example, feed it a matrix, it makes clustered horizontal bar plots.
    
    Parameters
    ----------
    data : array_like
       1D or 2D data to be plotted. For 1D data, plots   separate 
       bars for each entry; for 2D data, plots clusters of bars 
       (rows are clusters)
    err : array_like [optional]
       1D or 2D error bar data; same size as data
    color : list of matplotlib color specifiers [optional]
       color for each bar (or numbered bar in each cluster)
    xw : tuple (x, w)
       x is position (array, same size as data) and w is width 
       (scalar) for all bars. If set to None, this is computed to 
       give regularly-spaced bars and bar clusters

    """
    # Axis
    if not ax:
        ax = plt.subplot(111)
    if color is None:
        color = (0.5, )*3
    # Get x position, clusters
    if xw is None:
        x, w = bar_xw(data.shape)
    else:
        x, w = xw
    if x.ndim==1:
        x.shape +=(1, )
    # Optional vertical bar considerations
    if data.ndim==2:
        data = data.T
        x = x.T
        if not err is None: err = err.T
        #w = (.8-.05*(data.shape[0]-1))/data.shape[0]
        #x_pos = np.linspace(-.4+w/2, .4-w/2, data.shape[0])
        #x = np.vstack([x+xp for xp in x_pos]) 

    if is_vertical:
        xyErr='xerr'
        hw = 'height'
    else:
        xyErr='yerr'
        hw = 'width'
    # Color
    color = rgba(color)
    if len(color)<data.shape[0]:
        color = np.tile(color, (data.shape[0], 1))  
    edgecolor = rgba(edgecolor)
    if len(edgecolor)<data.shape[0]:
        edgecolor = np.tile(edgecolor, (data.shape[0], 1))
    # Line style
    if isinstance(ls, six.string_types):
        ls = (ls, )
    if len(ls)<data.shape[0]:
        ls = ls * data.shape[0]
    # handles for bar objects
    hb = []
    # Plot
    for xx, yy, ii in zip(x, data, range(len(x))):
        error_kw = {'capsize':0, 'elinewidth':lw, 'ecolor':edgecolor[ii]}
        bkwArgs = {'align':'center', hw:w, 'lw':lw, 'ls':ls[ii], 'color':color[ii], 'edgecolor':edgecolor[ii], 'error_kw':error_kw}
        if labels is not None:
            bkwArgs['label'] = labels[ii]
        if not err is None:
            bkwArgs[xyErr]=err[ii]
        if is_vertical:
            htmp = ax.barh(xx, yy, **bkwArgs)
            hb.append(htmp)
            limfn = ax.set_xlim
            tkax = ax.yaxis
        else:
            htmp = ax.bar(xx, yy, **bkwArgs)
            hb.append(htmp)
            tkax = ax.xaxis
            limfn = ax.set_ylim
    if not hatch is None:
        for hhb, pat in zip(hb, hatch):
            for b in hhb:
                if isinstance(pat, tuple):
                    p, pkw = pat
                else:
                    p = pat
                    pkw = {}
                bar_hatch(b, p, **pkw)
    if tklab is not None:
        tkax.set_ticks([r+1 for r in range(len(tklab))])
        tkax.set_ticklabels(tklab)
    if lims is not None:
        limfn(lims)
    return hb

def bar_hatch(bh, htype='/', col='k', space=0.1, lw=1, ls='-'):
    """Draw crosshatches within a bar object.
    
    Parameters
    ----------
    bh : handle
        handle for bar object to which to apply hatching
    htype : string, ['\\', '/', '|', or '-']
        Orientation of line to draw. If you want cross-hatching,
        call this function twice (e.g. with '|' and '-')
        TODO: build this functionality in with recursive calls and 
        inputs like '+', 'x'
    col : matplotlib color spec
        Color for lines
    space : float
        Space between lines.

    Other Parameters
    ----------------
    lw, ls : arguments for line formatting as in plot()

    Notes
    -----
    This function draws lines inside of your bars. The orientation of 
    the lines will depend on the aspect ratio of your plot. Thus,
    if you change the aspect ratio of a plot AFTER calling this function 
    (for example, by re-scaling the Y axis with ylim()), the "/" and "\\" 
    hatches will not be at 45 degrees.
    """
    
    ax = bh.get_axes()
    x, y, w, h = bh.get_bbox().bounds
    units_w, units_h = np.diff(ax.get_xlim()),  np.diff(ax.get_ylim()) 
    axis_w, axis_h = ax.figbox.width,  ax.figbox.height
    fig_w, fig_h = ax.figure.get_size_inches()
    W = units_w/(fig_w*axis_w)
    H = units_h/(fig_h*axis_h)
    ar = W/H
    if htype=='/':
        ys1 = np.arange(y-w/ar, y+h, space)
        ys2 = np.arange(y-w/ar, y+h, space)+w/ar
        xs1 = np.array([x]*len(ys1))
        xs2 = np.array([x+w]*len(ys1))
        d_y1 = y-np.minimum(ys1, y) # is positive; scoots ys1 up,  xs1 right
        d_x1 = d_y1*ar
        d_y2 = y+h-np.maximum(ys2, y+h) # is negative; scoots ys2 down,  xs2 left
        d_x2 = d_y2*ar
    elif htype=='-':
        ar = 1
        d_x1 = d_x2 = d_y1 = d_y2 = 0
        ys1 = ys2 = np.arange(y, y+h, space)
        xs1 = np.array([x]*len(ys1))
        xs2 = np.array([x+w]*len(ys1))
    elif htype=='\\':
        ys1 = np.arange(y, y+h+w/ar, space)
        ys2 = np.arange(y, y+h+w/ar, space)-w/ar
        xs1 = np.array([x]*len(ys1))
        xs2 = np.array([x+w]*len(ys1))
        d_y1 = y+h-np.maximum(ys1, y+h) # is negative; scoots ys1 down
        d_x1 = -d_y1*ar # ... and xs1 RIGHT (-) 
        d_y2 = y-np.minimum(ys2, y)
        d_x2 = -d_y2*ar
    elif htype=='|':
        d_x1 = d_x2 = d_y1 = d_y2 = 0
        xs1 = xs2 = np.arange(x, x+w, space*ar)
        ys1 = np.array([y]*len(xs1))
        ys2 = np.array([y+h]*len(xs1))
    elif htype=='+':
        bar_hatch(bh, htype='|', col=col, space=space, lw=lw, ls=ls)
        bar_hatch(bh, htype='-', col=col, space=space, lw=lw, ls=ls)
        return
    elif htype=='x':
        bar_hatch(bh, htype='/', col=col, space=space, lw=lw, ls=ls)
        bar_hatch(bh, htype='\\', col=col, space=space, lw=lw, ls=ls)
        return
    elif htype=='':
        return
    else:
        raise ValueError("Unknown hatch type '%s'"%htype)
    X = np.vstack((xs1+d_x1, xs2+d_x2)) # note aspect ratio
    Y = np.vstack((ys1+d_y1, ys2+d_y2))
    # line collection?
    _ = plt.plot(X, Y, color=col, lw=lw, ls=ls, solid_capstyle='round')

def histline(data, bins=25, ax=None, density=False, is_vertical=False, **kwargs):
    """Draw a line instead of plotting bars for a histogram."""
    y, be = np.histogram(data, bins, density=density)
    if density:
        y /= y.sum()
    if ax is None:
       ax = plt.gca()
    if is_vertical:
        ll = ax.plot(y, be[:-1] + np.diff(be)/2., **kwargs)
    else:
        ll = ax.plot(be[:-1] + np.diff(be)/2., y, **kwargs)
    return ll

def plot3(x, y, z, ptype='.', fig=None, ax=None, color=(0., 0., 0.), **kwargs):
    '''Slightly different syntax from normal 3D plotting in matplotlib
    '''
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot(111, projection='3d')
        #ax = Axes3D.Axes3D(fig)
    ax.plot3D(x, y, z, ptype, color=color, **kwargs)

# Plotting shapes
def circle_pos(radius, n_positions, angle_offset=0, x_center=0, y_center=0, direction='CW', duplicate_first=False):
    '''Return points in a circle. 
    
    Parameters
    ----------
    radius : scalar 
        Radius of the circle. Either one value or n_positions values.
    n_positions : int 
        number of positions around the circle
    x_center : scalar
        Defaults to 0
    y_center : scalar
        Defaults to 0
    direction : string 
        Specifies direction of points around circle - 
            'CW' - clockwise
            'CCW' - clockwise
        To ensure backward compatibility, you can also specify starting position - 
            'BotCCW' - start from bottom, go Counter-Clockwise, 
            'BotCW' - start from bottom, go Clockwise,
            'TopCCW' - top, Counter-Clockwise
            'TopCW' - top, Clockwise
        Starting with 'Top' is equivalent to angle_offset of 180 
        and will override angle_offset value if specified.
        
    '''
    if (isinstance(radius, list) and len(radius)==1) or isinstance(radius, (float, int)):
        radius = np.tile(radius, (n_positions+duplicate_first, 1))
    if direction[:3].upper() == 'TOP':
        angle_offset=180
    circ_pos = np.nan * np.ones((n_positions+duplicate_first, 2))
    angles = np.linspace(0, 360, n_positions+1)
    if direction[-3:].upper() != 'CCW':
        angles = angles[::-1]
    if not duplicate_first:
        angles = angles[:-1]
    for i, angle in enumerate(angles):
        circ_pos[i, 0] = radius[i]*_sind(angle+angle_offset) + x_center
        circ_pos[i, 1] = -radius[i]*_cosd(angle+angle_offset) + y_center
    return circ_pos

def eegplot(m, x=None, sep=6, scale=None, ax=None, **kwargs):
    """Plots the columns of a 2D matrix EEG-style (as offset traces)"""
    if ax is None:
        fig, ax = plt.subplots()
    t, n_cols = m.shape
    tt = np.arange(0, n_cols*sep, sep)
    ss = np.arange(0, t*sep, sep)
    yy, xx = np.meshgrid(tt, ss)
    if not scale is None:
        m = m / np.max(np.abs(m))
        m = m * scale
    if not x is None:
        xx = x
    ax.plot(xx, m+yy, **kwargs)
    ax.set_ylim(-sep, n_cols*sep)
    
'''
Old ver:
def eegplot(m, x=None, sep=6, **kwargs):
    """Plots the columns of a 2D matrix EEG-style (as offset traces)"""
    ii, jj = m.shape
    tt = np.arange(0, jj*sep, sep)
    ss = np.arange(0, ii*sep, sep)
    xx, yy = np.meshgrid(tt, ss)
    if not x is None:
        plt.plot(x, m+xx, **kwargs)
    else:
        plt.plot(m+xx, **kwargs)
    plt.ylim(-sep, jj*sep)
'''
### --- Modification of axes --- ###

# def setFont(ax, fontNm='Helvetica'):   
#   fSzIn = ax.get_figure().get_size_inches()
#   ax_pos = ax.get_position() # L, B, W, H
#   aSzIn = 0 # mult ax_pos x fSzIn

def set_ax_fontsz(ax, lab=None, tk=None, name=None):
    """set axis tick label size"""
    if not lab is None:
        # xlabel, ylabel
        ax.xaxis.label.set_fontsize(lab) 
        ax.yaxis.label.set_fontsize(lab)
        if hasattr(ax, 'zaxis'):
            ax.zaxis.label.set_fontsize(lab)
    if not tk is None:
        # tick labels
        ax.yaxis.set_tick_params(labelsize=tk)
        ax.xaxis.set_tick_params(labelsize=tk)
        if hasattr(ax, 'zaxis'):
            ax.zaxis.set_tick_params(labelsize=tk)
    if not name is None:
        # tick labels
        for a in ax.get_xticklabels():
            a.set_fontname(name)
        for a in ax.get_yticklabels():
            a.set_fontname(name)
        if hasattr(ax, 'zaxis'):
            for a in ax.get_zticklabels():
                a.set_fontname(name)
        #ax.yaxis.set_tick_params(fontname=name)
        #ax.xaxis.set_tick_params(fontname=name)
        ax.xaxis.label.set_fontname(name)
        ax.yaxis.label.set_fontname(name)

def fnt(size=12, weight='normal', family='sans-serif'):
    f = {'family':family,
        'weight':weight,
        'size':size}
    return f

defFont = fnt()

def prep_inkscape_svg(h):
    """
    Preps a figure / axis for saving as an SVG 
    h can be a figure or an axis
    Set font size to be 80% smaller (due to an inkscape bug!)
    (80% = inkscape default of 72 dpi / matplotlib default of 90 dpi)
    """
    # if type(h) is matplotlib.axes.AxesSubplot: # Why doesn't this work??
    if not type(h) is matplotlib.figure.Figure:
        A = [h]
        f = h.get_figure()
    else: # type(h) is matplotlib.figure.Figure:
        A = h.get_axes()
        f = h
    for ax in A:
        ToFix = [ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels() + [t for t in ax.get_children() if isinstance(t, matplotlib.text.Text)]
        L = ax.get_legend()
        if L:
            ToFix+=[L.get_texts()[0]]
            #print('Adding Legend!')
        for item in ToFix:
            fSz = float(item.get_fontsize())
            item.set_fontsize(fSz*.8)
            #print(item)
    f.set_dpi(90)

def set_y_axis():
    """set y axis to a particular position (e.g. centered)"""
    pass


def set_axis_lines(ax, color='k', lw=1):
    """Set axis lines to a given color and linewidth"""
    for side in ['top', 'bottom', 'left', 'right']:
        ax.spines[side].set_color(color)
        ax.spines[side].set_linewidth(lw)


def open_axes(ax=None):
    """Removes top, right axis borders (makes into an open graph)

    Notes
    -----
    See:
    http://www.shocksolution.com/2011/08/removing-an-axis-or-both-axes-from-a-matplotlib-plot/
    """
    if ax is None:
        ax = plt.gca()
    for loc, spine in ax.spines.items():
        #if loc in ['left', 'bottom']:
        # spine.set_position(('outward', 10)) # outward by 10 points
        if loc in ['right', 'top']: spine.set_color('none') # don't draw spine
        # else:
        # raise ValueError('unknown spine location: %s'%loc)
    # turn off ticks where there is no spine
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

### --- Distribution of axes on plot --- ###
def tile_axes(nrows, ncolumns, space_to_tile=(0., 0., 1., 1.), order_flag=1,
    gap=(0.0, ), postype='OuterPosition', return_ax=True):
    """Creates tiled subplots. 

    Maybe better than subplot or AxisGrid1 (?)

    Parameters
    ----------
    nrows : int
    ncolumns : int
    space_to_tile : tuple | (0., 0., 1., 1.)
       The rectangle within the figure you'd like to tile with axes,
       (xBotLeft yBotLeft Width Height). (.5, .5, .5, .5) would be the 
       top right corner.
    order_flag : int | [1 | 2]
       Determines order of axes. 
       order_flag = 1  |   order_flag = 2
                 |
       1 2 3 4 5   |   1 4 7 10 e
       6 7 8 9 10     |   2 5 8 11 t
       11... etc   |   3 6 9 12 c...
    gap : int or tuple
       The gap between the axes (in normalized figure units).
       If gap is specified as a 2-tuple such as (0, .05), the values 
       are interpreted as (Columngap, Rowgap).
    Other Parameters
    ----------------
    ** NOT IMPLEMENTED YET **
      postype is 'OuterPosition' or 'Position' (input argument to "axes"
       command)

    Returns
    -------
    axpos : list of lists
        Position rectangles for the number of axes specified,
    axh : axis handles
        axis handles (only returned if return_ax is True; this also
        causes function to create axes in figure)
    """
    if not isinstance(gap, (tuple, list)):
        gap = (gap, )
    if len(gap)==1:
        gap *= 2 
    elif len(gap)>2:
        raise Exception('could not understand gap size')

    ax_width = space_to_tile[2]/float(ncolumns)
    ax_height = space_to_tile[3]/float(nrows)
    nrows = int(nrows)
    ncolumns = int(ncolumns)
    HorizSpacing = np.linspace(space_to_tile[0], space_to_tile[0]+space_to_tile[2]-ax_width, ncolumns);
    VertSpacing  = np.linspace(space_to_tile[1], space_to_tile[1]+space_to_tile[3]-ax_height, nrows);
    Horiz, Vert = np.meshgrid(HorizSpacing, VertSpacing);
    if order_flag==2:
        Horiz = Horiz.T
        Vert = Vert.T
    # for 2, leave as is
    Horiz = Horiz.flatten()
    Vert = np.flipud(Vert.flatten())
    
    TmpAx_pos = np.vstack([Horiz, Vert, [ax_width]*nrows*ncolumns, [ax_height]*nrows*ncolumns]).T

    OffSetX = ax_width*gap[0]
    OffSetY = ax_height*gap[1]
    TmpAx_pos[:, 0] = TmpAx_pos[:, 0] + OffSetX;
    TmpAx_pos[:, 2] = TmpAx_pos[:, 2] - 2 * OffSetX;
    TmpAx_pos[:, 1] = TmpAx_pos[:, 1] + OffSetY;
    TmpAx_pos[:, 3] = TmpAx_pos[:, 3] - 2 * OffSetY;

    axpos = TmpAx_pos;

    if return_ax:
        axh = []
        for irc in range(int(nrows*ncolumns)):
            axh += [plt.axes(axpos[irc])]
        return axpos, axh
    else:
        return axpos

def slice_3d_matrix(volume, axis=2, figh=None, vmin=None, vmax=None, cmap=plt.cm.gray, nr=None, nc=None ):
    '''Slices 3D matrix along arbitrary axis

    Parameters
    ----------
    volume : array (3D)
    axis : int | 0, 1, [2] (optional)
       axis along which to divide the matrix into slices

    Other Parameters
    ----------------
    vmin : float [max(volume)] (optional) 
       color axis minimum
    vmax : float [min(volume)] (optional)
       color axis maximum
    cmap : matplotlib colormap instance [plt.cm.gray] (optional)
    nr : int (optional)
       number of rows
    nc : int (optional)
       number of columns
    '''
    if figh is None:
        fig = plt.figure()
    else:
        fig = plt.figure(figh)
    if nr is None or nc is None:
        nc, nr = find_squarish_dimensions(volume.shape[axis])
    if vmin is None:
        vmin = volume.min()
    if vmax is None:
        vmax = volume.max()
    ledges = np.linspace(0, 1, nc+1)[:-1]
    bedges = np.linspace(1, 0, nr+1)[1:]
    width = 1/float(nc)
    height = 1/float(nr)
    bottoms, lefts = zip(*list(itertools.product(bedges, ledges)))
    for ni, sl in enumerate(np.split(volume, volume.shape[axis], axis=axis)):
        #ax = fig.add_subplot(nr, nc, ni+1)
        ax = fig.add_axes((lefts[ni], bottoms[ni], width, height))
        ax.imshow(sl.squeeze(), vmin=vmin, vmax=vmax, interpolation="nearest", cmap=cmap)
        ax.set_xticks([])
        ax.set_yticks([])
    return fig

def _lbwh2lrtb(lbwh):
    """Left Bottom Width Height -> Left Right Top Bottom"""
    l, b, w, h = lbwh
    return [l, l+w, b, b+h]

def mosaic(data, ax=None, cmap=plt.cm.gray, vmin=None, vmax=None, nr=None, nc=None, aspect='auto', **kwargs):
    """Display image stack as contact sheet
    
    Parameters
    ----------
    data : array
        (y, x, [color], n)
    kwargs map to tile_axes()
    """
    if ax is None:
        ax = plt.gca()
    if np.ndim(data)==3:
        n, y, x = data.shape
        cmkw = dict(cmap=cmap, vmin=vmin, vmax=vmax, aspect=aspect)
    elif np.ndim(data)==4:
        n, y, x, _ = data.shape
        cmkw = dict(aspect=aspect)
    if nr is None and nc is not None:
        nr = int(n/nc)
    elif nr is not None and nc is None:
        nc = int(n/nr)
    elif nr is None and nc is None:
        # For now: assume both are none if nr is not provided
        nr, nc = find_squarish_dimensions(n)
    pos = tile_axes(nr, nc, return_ax=False, **kwargs)
    for position, frame in zip(pos, data):
        ext = _lbwh2lrtb(position)
        ax.imshow(frame,  extent=ext, **cmkw)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

def plot_ims(x, y, ims, imsz=0.08, ax=None, ylim=None, xlim=None, im_border=None, **kwargs):
    """Plot images at x, y locations
    
    Parameters
    ----------
    x : array-like
        x position for each image
    y : array-like
        y position for each image
    ims : 3 or 4-D array
        image matrix (images, x, y, [color])
    imsz : float
        fraction of horizontal axis for each image to take up.
        (vertical dimension is scaled by aspect ratio)
    ax : matplotlib axes instance
        axes into which to plot
    ylim : array-like
        y limits for plot (necessary because of oddities with imshow)
        defaults to [min(y) + imsz/2*range(y), max(y) + imsz/2*range(y)]
    xlim : array-like
        x limits for plot (see ylim)
    im_border : dict or list of dicts
        parameters to pass to add_border (if borders around images are desired)
    kwargs : dict
        passed to imshow
    """
    # Get axes 
    if ax is None:
        fig = plt.figure()
        ax = plt.gca()
    else:
        fig = ax.get_figure()
    # Handle aspect ratio of axes, images
    figw, figh = fig.get_size_inches()
    bb = ax.get_position()
    w, h = bb.width*figw, bb.height*figh
    aspect_ratio = h/w
    #print(aspect_ratio)
    # Range of image coordinates in x and y
    x_range = np.max(x)-np.min(x)
    #print(x_range)
    y_range = np.max(y)-np.min(y)
    #print(y_range)
    # Default x, y limits
    if ylim is None:
        ybuf = imsz*aspect_ratio*x_range*0.5
        ylim = [np.min(y)-ybuf, np.max(y)+ybuf]
    if xlim is None:
        xbuf = imsz*x_range
        xlim = [np.min(x)-xbuf, np.max(x)+xbuf]
    ysz = ylim[1]-ylim[0]
    xsz = xlim[1]-xlim[0]
    if 'zorder' in kwargs:
        zo = kwargs.pop('zorder')
    else:
        zo = 1
    # Plot
    imh = []
    for i, (x_, y_, im) in enumerate(zip(x, y, ims)):
        # pos is [L, R, top, bottom]
        sidex = xsz*imsz*aspect_ratio
        sidey = ysz*imsz
        pos = [x_-sidex/2., x_+sidex/2.,
               y_-sidey/2., y_+sidey/2.]
        imh.append(ax.imshow(im, extent=pos, aspect='auto', zorder=zo+i*2, **kwargs))
    # This has to be after re-setting of limits, annoyingly enough:
    if im_border is not None:
        for i, I in enumerate(imh):
            # for now, pass a dict of color=blah, lw=blah
            if isinstance(im_border, list):
                add_border(I, zorder=zo+i*2+1, **im_border[i])
            else:
                add_border(I, zorder=zo+i*2+1, **im_border)
    # re-establish limits (due to imshow wonkiness)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    return

def plot_annotation_ims(x, y, ims, ann_idx, offset=0, ax=None, ylim=None, xlim=None,
                        line_kw=None, x_im=None, y_im=None, plot_xy_line=True,
                        **pim_kw): #offs_kw=None,  #color='k', ls='-', lw=1,
    """Plot images annotating a lineplot
    
    Parameters
    ----------
    x, y : line data
    ims : 3D or 4D array
        image array, (x, y, [color], image)
    ann_idx : array-like
        index of which values in x, y, and ims to plot images for
    offset : int | tuple | array
         if int, offsetx and offsety are assumed to be equal
         if tuple, assumed to be (offsetx, offsety)
         if array, offset should be [len(ann_idx) x 2], specifying (x, y) offset 
         for each image separately
    xlim, ylim : limits
    x_im, y_im : direct specification of positions for where to plot the images, if not at 
                 (x[ann_idx], y[ann_idx]). If either is specified, it overrides `offset` 
                 for x or y.
    """
    # Get axes 
    if ax is None:
        fig = plt.figure()
        ax = plt.gca()
    else:
        fig = ax.get_figure()    
    line_kw_ = dict(color='k', lw=1, ls='-')
    if line_kw is None:
        line_kw = {}
    line_kw_.update(line_kw)
    # Plot x, y values
    if plot_xy_line:
        ax.plot(x, y, zorder=1, **line_kw_)
    # Sort the rest
    xi = x[ann_idx]
    yi = y[ann_idx]
    if isinstance(offset, int):
        offsetx, offsety = offset, offset
    elif isinstance(offset, tuple):
        offsetx, offsety = offset
    elif isinstance(offset, np.ndarray):
        offsetx, offsety = offset.T
    if x_im is None:
        x_im = xi+offsetx
    if y_im is None:
        y_im = yi+offsety

    # Images
    plot_ims(x_im, y_im, ims[ann_idx], ax=ax, ylim=ylim, xlim=xlim, zorder=2,
        **pim_kw)
    # Offset lines 
    ax.plot(np.vstack([xi, x_im]), np.vstack([yi, y_im]), zorder=1, **line_kw_)
    return

def imsave_pil(fname, im, vmin=None, vmax=None, alpha=None):
    """Save an array to an image using PIL

    Converts an array to uint8 and saves it as 
    """
    import Image as PIL
    sz = im.shape
    # Normalize image 0-1 or 0-255 ??

    if im.ndim==2:
        # X, Y only
        imc = im
    elif im.ndim==4:
        if not alpha is None:
            raise Exception('Either provide 4-D image OR alpha, not both')
        alpha = im[:, :, 3]
        im = im[:, :, :3]
    if np.max(im)<=1.0:
        im = np.cast['uint8'](im*255)
    imP = PIL.fromarray(im)
    if not alpha is None:
        if np.max(alpha)<=1.0:
            alpha = np.cast['uint8'](alpha*255)
        alpha = np.cast['uint8'](alpha*255)
        alphaP = PIL.fromarray(alpha)
        imP.putalpha(alphaP)
    imP.save(fname)

def write_gif(fstr, sname, delay=4, **kwargs):
    '''Create an animated GIF image from a list of files in a directory
    
    A wrapper for ImageMagik's "convert" function. 

    Parameters
    ----------
    fstr : str | list
        a string specifies a glob pattern; a list specifies specific file names (full paths)
    sname : str
        Full path for file to save
    delay : int
        number of 10 ms bins per frame. This is the stupid way imagemagick controls animation
        speed. For example, for animation at approximately 24 hz, (~41 ms per frame), specify
        delay=4. For 15 hz, specify 6
    See http://www.imagemagick.org/script/convert.php for more on convert commands
    '''
    
    magik_cmd = ['convert', '-delay', str(delay)]
    for k, v in kwargs.items():
       magik_cmd.append('-'+k)
       magik_cmd.append(v)
    if not isinstance(fstr, list):
       fstr = [fstr]
    magik_cmd+=fstr
    magik_cmd+=[sname]
    proc = subprocess.Popen(magik_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    out, err = proc.communicate()
    if not err=='':
        print(err)
        raise Exception('write_gif failed!') #:\n%s'%stderr)

def write_movie(fstr, sname, codec='libx264', framerate=24, **kwargs):
    """Simple wrapper for ffmpeg/avconv creation/conversion of video.

    Calls ffmpeg to create a video from a string of images. Currently,
    the images need to be numbered sequentially (fr001.png, fr002.png, etc)
    There are ways around this but they are a bigger pain in the ass than 
    I am currently willing to deal with. 

    thus fstr should be of the format fr%03d.png, unless you specify pattern_type='glob',
    in which case you can use fr*png

    for all other parameters that have no value (e.g. sameq), use param=None

    Other Parameters
    ----------------
    (all specified as keyword args)
    target : target format (e.g. dvd) - see ffmpeg help
    t : time duration. 
    sameq : same quality for input/output. use sameq=None
    start_number : 
    pattern_type : use 'glob' to specify non-numeric (but still sequential) files as input
    """
    if os.sys.platform in ('linux2'):
        vid = 'avconv'
    else:
        vid = 'ffmpeg'
    vid_cmd = [vid, '-framerate', str(framerate)]
    for k, v in kwargs.items():
        vid_cmd.append('-'+k)
        if not v is None:
            vid_cmd.append(v)
    vid_cmd+=['-i', fstr]
    vid_cmd+=[sname]
    print(vid_cmd)
    proc = subprocess.Popen(vid_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    out, err = proc.communicate()
    if not err=='':
        print('==> Error:')
        print(err)
        print("==> Output:")
        print(out)
        # raise Exception('ffmpeg failed!') #:\n%s'%stderr)

def add_border(im, color='k', lw=4, is_trim=False, round_offset=0, inset=None, zorder=4):
    """Add a border around an image.

    Parameters
    ----------
    color : colorspec
        color for border
    im : matplotlib image instance
        image to make border around
    is_trim : bool
        Clip axes to be tight around image
    inset : scalar
        how much into the image to draw the border 
        (in axis units, which will be pixels by default))

    """
    L, R, T, B = im.get_extent()
    # Horizontal / vertical
    ar = (R-L)/(B-T)
    if not inset is None:
        L+=inset*ar
        R-=inset*ar
        T-=inset
        B+=inset
    ax = im.axes
    # Translate extent to bounding box
    ww =[[L, B], [R, T]]
    bb = mtransforms.Bbox(ww)
    offset = np.abs(bb.width)*round_offset
    p_fancy = FancyBboxPatch((bb.xmin, bb.ymin),
                    abs(bb.width), abs(bb.height),
                    boxstyle="round, pad=%0.2f"%(offset),
                    fc="none",
                    ec=color,
                    lw=lw,
                    zorder=zorder)
    ax.add_patch(p_fancy)
    if is_trim:
        plt.axis('off')
        plt.setp(ax, ylim=(T+lw*offset/2., B-lw*offset/2.), xlim=(L-lw*offset/2., R+lw*offset/2.))

def unsaturate(cols, sat_factor=0.5):
    """Simple color de-saturation for rgba arrays 
    
    `cols` is a n x 4 rgba array, or iterable that can be fed to matplotlib.colors.colorConverter.to_rgba_array
    `sat_factor` is 0-1 scaling on color saturation

    Returns
    -------
    rgba : rgba array
    """
    from matplotlib import colors
    # Assure colors are in rgba array form
    cols = colorConverter.to_rgba_array(cols)
    c2 = colors.rgb_to_hsv(cols[None, :, :3])
    # Reduce saturation channel by (sat_factor)
    c2[0, :, 1]*=sat_factor
    # Convert back to RGB
    c3 = colors.hsv_to_rgb(c2)[0]
    # add alpha back
    out = np.hstack((c3, cols[:, 3:]))
    return out

def density_plot(x, y, n_bins_x=200, n_bins_y=200, xlim=None, ylim=None,
    cmap=plt.cm.Reds, max_ct=None, colorbar=True, ax=None):
    """Make a density plot of data

    Like a scatter plot, but with density counts for each pixel of a 
    2D grid (over 2 dimensions) instead of plots of individual points. 


    """
    if ax is None:
        ax = plt.gca()
    # Define limits
    if xlim is None:
        xlim = (np.nanmin(x), np.nanmax(x))
    if ylim is None:
        ylim = (np.nanmin(y), np.nanmax(y)) 
    xbe = np.linspace(xlim[0], xlim[1], n_bins_x+1)
    ybe = np.linspace(ylim[0], ylim[1], n_bins_y+1)
    # Estimate the 2D histogram
    H, xedges, yedges = np.histogram2d(x, y, bins=[xbe, ybe])
    # H needs to be rotated and flipped
    H = np.rot90(H)
    H = np.flipud(H)
    # Mask zeros
    Hmasked = np.ma.masked_where(H==0, H) # Mask pixels with a value of zero
    # Plot 2D histogram using pcolor
    if max_ct is None:
        max_ct = np.max(Hmasked)
    im = ax.pcolormesh(xedges, yedges, Hmasked, cmap=cmap, vmin=0, vmax=max_ct)
    if colorbar:
        cbar = plt.colorbar(im)
        cbar.ax.set_ylabel('Counts')

def make_image_animation(images, 
        overlay=None,
        figsize=(5,5),
        fps=30,
        extent=None,
        cmap=None,
        yticks=None,
        xticks=None,
        ylabel=None,
        xlabel=None,
        overlay_kwargs=None,
        **kwargs):
    """interval appears to be in ms
    
    Parameters
    ----------
    images : array
        array of (time, vdim, hdim, color)
    figsize : tuple, optional
        size of figure. Determines aspect ratio of movie.
    fps : int, optional
        frames per second of the animation
    extent : None, optional
        extent of the image plotted in matplotlib axis
    cmap : None, optional
        colormap, if not an RGB image
    yticks : None, optional
        Y ticks
    xticks : None, optional
        X ticks
    ylabel : None, optional
        Y axis label
    xlabel : None, optional
        X axis label
    
    Returns
    -------
    TYPE
        matplotlib animation
    """
    
    # First set up the figure, the axis, and the plot element we want to animate
    fig, ax = plt.subplots(figsize=figsize)
    # Show image & prettify
    # (Allow for multiple overlain images?)
    im = ax.imshow(images[0], extent=extent, cmap=cmap, **kwargs)
    imsz = images[0].shape
    if overlay is not None:
        if overlay_kwargs is None:
            overlay_kwargs = {}
        im_o = ax.imshow(overlay[0], extent=extent, **overlay_kwargs)
    if yticks is not None:
        ax.set_yticks(yticks)
    if xticks is not None:
        ax.set_xticks(xticks)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if (xticks is None) and (yticks is None) and (xlabel is None) and (ylabel is None):
        ax.set_position([0, 0, 1, 1])
        ax.axis('off')
    # Hide figure, we don't care
    plt.close(fig.number)
    # initialization function: plot the background of each frame
    def init():
        im.set_array(np.zeros(imsz))
        if overlay is not None:
            im_o.set_array(np.zeros(overlay[0].shape))
            return (im, im_o)
        else:
            return (im,)
    # animation function. This is called sequentially
    def animate(i):
        im.set_array(images[i])
        if overlay is not None:
            im_o.set_array(overlay[i])
            return (im, im_o)
        else:
            return (im,)
    # call the animator. blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=images.shape[0], interval=1/fps * 1000, blit=True)
    return anim

def make_subplot_image_animation(image_stacks, data=None, n_rows=None, n_cols=None, figsize=(5,5), fps=30, extent=None, cmap=None,
                         yticks=None, xticks=None, ylabel=None, xlabel=None, **kwargs):
    """interval appears to be in ms
    
    Parameters
    ----------
    images : array
        array of (time, vdim, hdim, color)
    figsize : tuple, optional
        size of figure. Determines aspect ratio of movie.
    fps : int, optional
        frames per second of the animation
    extent : None, optional
        extent of the image plotted in matplotlib axis
    cmap : None, optional
        colormap, if not an RGB image
    yticks : None, optional
        Y ticks
    xticks : None, optional
        X ticks
    ylabel : None, optional
        Y axis label
    xlabel : None, optional
        X axis label
    
    Returns
    -------
    TYPE
        matplotlib animation
    """
    if n_rows is None:
        n_rows, n_cols = find_squarish_dimensions(len(image_stacks))
    if data is None:
        data = [None] * len(image_stacks)
    n_frames = np.max([len(x) for x in image_stacks])
    # First set up the figure, the axis, and the plot element we want to animate
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])
    # Show image & prettify
    im_h = []
    for j, (ims, ax) in enumerate(zip(image_stacks, axs.flatten())):
        tmp = ims[0]
        h = ax.imshow(tmp, extent=extent, cmap=cmap, **kwargs)
        im_h.append(h)
        imsz = tmp.shape
        if yticks is not None:
            ax.set_yticks(yticks)
        if xticks is not None:
            ax.set_xticks(xticks)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if (xticks is None) and (yticks is None) and (xlabel is None) and (ylabel is None):
        #    ax.set_position([0, 0, 1, 1])
            ax.axis('off')
    # Hide figure, we don't care
    plt.close(fig.number)
    # initialization function: plot the background of each frame
    def init():
        for h in im_h:
            h.set_array(np.zeros(imsz))
        return im_h
    # animation function. This is called sequentially
    def animate(i):
        for j, h in enumerate(im_h):
            if i >= len(image_stacks[j]):
                continue
            h.set_array(image_stacks[j][i])
        return im_h
    # call the animator. blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=n_frames, interval=1/fps * 1000, blit=True)
    return anim


def show_image_animation(images, figsize=(5,5), fps=30, extent=None, cmap=None,
                         yticks=None, xticks=None, ylabel=None, xlabel=None):
    """Same as `make_image_animation`, but displays the movie rather 
    than returning an animation object. See `make_image_animation` 
    for help!"""
    anim = make_image_animation(images, figsize=figsize, fps=fps, extent=extent, cmap=cmap,
                         yticks=yticks, xticks=xticks, ylabel=ylabel, xlabel=xlabel)
    HTML(anim.to_html5_video())

def _connector(x, y, ht, lw=2, ax=None):
    """draw an upside-down u annotation line. Only works for vertical/horiz lines."""
    from matplotlib.path import Path
    import matplotlib.patches as patches    
    dy = float(y[1]-y[0])
    dx = float(x[1]-x[0])
    if dx==0:
        dx = 1e-8
    theta = np.arctan2(dy, dx)
    xx = np.cos(np.pi/2-theta)*ht
    yy = np.sin(np.pi/2-theta)*ht
    verts = [
        (x[0], y[0]), # left, bottom
        (x[0]+xx, y[0]+yy), # left, top
        (x[1]+xx, y[1]+yy), # right, top
        (x[1], y[1]), # right, bottom
        ]
    codes = [Path.MOVETO,
         Path.LINETO,
         Path.LINETO,
         Path.LINETO,
         ]
    if ax is None:
        ax = plt.gca()
    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor='none', lw=lw)
    ax.add_patch(patch)

# Custom function to draw the diff bars
def label_diff(i, j, text, X, Y, frac_offset=0.1, ht=None, ax=None, lw=2, z=10, fontsize=10):
    """Draw labels for difference in a bar graph"""
    if ax is None:
        ax = plt.gca()
    x = (X[i]+X[j])/2
    y = (1+frac_offset)*np.max(Y)
    if ht is None:
        ht = (np.max(Y)-y)*frac_ht
    dx = abs(X[i]-X[j])
    _connector([X[i], X[j]], [y, y], ht, ax=ax, lw=lw)
    ax.text(x, y+ht, text, ha='center', va='baseline', fontsize=fontsize)

# menMeans   = (5, 15, 30, 40)
# menStd     = (2, 3, 4, 5)
# ind  = np.arange(4)   # the x locations for the groups
# width= 0.7
# labels = ('A', 'B', 'C', 'D')

# # Pull the formatting out here
# bar_kwargs = {'width':width, 'color':'y', 'linewidth':2, 'zorder':5}
# err_kwargs = {'zorder':0, 'fmt':None, 'lw':2, 'ecolor':'k'}

# X = ind+width/2

# fig, ax = plt.subplots()
# ax.p1 = plt.bar(ind, menMeans, **bar_kwargs)
# ax.errs = plt.errorbar(X, menMeans, yerr=menStd, **err_kwargs)

# # Call the function
# label_diff(0, 1, 'p=0.0370', X, menMeans)
# label_diff(1, 2, 'p<0.0001', X, menMeans)
# label_diff(2, 3, 'p=0.0025', X, menMeans)


# plt.ylim(ymax=60)
# plt.xticks(X, labels, color='k')
# plt.show()


def make_overlay_anim(idx, resp=None, prf=None, stim=None, interval=16*5, normalize_response=True, extent=None):
    """interval appears to be in ms"""
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure(figsize=(4,6))
    gs = gridspec.GridSpec(3,1)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1:])
    y = resp[:, idx]
    n_trs = len(y)
    t_plot = np.arange(0, n_trs*1.5, 1.5)
    # Normalize 0-1
    if normalize_response:
        y = (y-y.min())/ (y.max()-y.min())
        yl = [-0.2, 1.2]
    else:
        yrange = y.max()-y.min()
        yl = (y.min() - yrange*0.2, y.max() + yrange*0.2)
    # Plot & prettify
    lines, = ax1.plot(t_plot, np.ones(n_trs,))
    ax1.set_ylim(yl)
    ax1.set_ylabel("Predicted\nresponse")
    ax1.set_xlabel("Time (s)")
    # Show image & prettify
    im = ax2.imshow(stim[:,:,0], extent=extent, cmap='gray')
    im2 = ax2.imshow(prf[..., idx], extent=extent, alpha=0.3, cmap='hot')
    ax2.set_yticks([-10, -5, 0, 5, 10])
    ax2.set_ylabel('Visual field\nY position (deg)')
    ax2.set_xlabel('Visual field\nX position (deg)')
    plt.tight_layout()
    # Hide figure, we don't care
    plt.close(fig.number)
    # initialization function: plot the background of each frame
    def init():
        lines.set_data([], [])
        im.set_array(np.zeros((101,101)))
        return (im, lines)
    # animation function. This is called sequentially
    def animate(i):
        lines.set_data(t_plot[:i], y[:i])
    # call the animator. blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=stim.shape[-1], interval=interval, blit=True)
    return anim


class FigureHTMLConverter:
    """Converts from plots added to figure objects to HTML video within ipython.
    """

    frames = []
    
    def add(self, fig):
        """Add new frames to video
        
        Parameters
        ----------
        fig : matplotlib.figure.Figure
            figure object which has images already added
        """
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        self.frames.append(data)
    
    def clear(self):
        """Clear all frames previously added
        """
        self.frames = []
        
    def render(self, **kwargs):
        """Turn added frames into HTML video within jupyter notebook.
        
        Parameters
        ----------
        **kwargs
            Keyword arguments for vmt.plot_utils.make_image_animation, 
            such as figsize and interval (in ms).
        """
        frames_array = np.moveaxis(np.array(self.frames),0,-1)
        anim = make_image_animation(frames_array, **kwargs)
        display(HTML(anim.to_html5_video()))


def make_dot_overlay_animation(video_data, dot_locations, dot_timestamps=None, video_timestamps=None,
                               dot_widths=None, dot_colors=None, dot_labels=None, dot_markers=None, 
                               figsize=None, fps=60, **kwargs):
    """Make an animated plot of dots moving around on a video image
    
    Useful for showing estimated gaze positions, detected marker positions, etc. Multiple dots
    indicating multiple different quantites, each with different plot (color, marker type, width)
    can be plotted with this function.

    Parameters
    ----------
    video_data : array
        stack of video_data, (time, vdim, hdim, [c]), in a format showable by plt.imshow()
    dot_locations : array
        [n_dots, n_frames, xy]: locations of dots to plot, either in normalized (0-1 for 
        both x and y) coordinates or in pixel coordinates (with pixel dimensions matching 
        the size of the video)   
    dot_timestamps : array
        [n_dots, n_dot_frames] timestamps for dots to plot; optional if a dot location is 
        specified for each frame. However, if `dot_timestamps` do not match 
        `video_timestamps`, dot_locations are resampled (simple block average) to match 
        with video frames using these timestamps.
    video_timestamps : array
        [n_frames] timestamps for video frames; optional, see `dot_timestamps`
    dot_widths : scalar or list
        size(s) for dots to plot
    dot_colors : matplotlib colorspec (e.g. 'r' or [1, 0, 0]) or list of colorspecs
        colors of dots to plot. Only allows one color across time for each dot (for now). 
    dot_labels : string or list of strings
        label per dot (for legend) NOT YET IMPLEMENTED.
    dot_markers : string or list of strings
        marker type for each dot
    figsize : tuple
        Size of figure
    fps : scalar
        fps for resulting animation

    Notes
    -----
    Good tutorial, fancy extras: https://alexgude.com/blog/matplotlib-blitting-supernova/
    """
    from functools import partial
    def prep_input(x, n):
        if not isinstance(x, (list, tuple)):
            x = [x]
        if len(x) == 1:
            x = x * n
        return x
    # Inputs
    if np.ndim(dot_locations) == 2:
        dot_locations = dot_locations[np.newaxis, :]
    if dot_markers is None:
        dot_markers = 'o'
        
    # Shape
    extent = [0, 1, 1, 0]
    # interval is milliseconds; convert fps to milliseconds per frame
    interval = 1000 / fps
    # Setup
    n_frames, y, x = video_data.shape[:3]
    im_shape = (y, x)
    aspect_ratio = x / y
    if figsize is None:
        figsize = (5 * aspect_ratio, 5)
    if np.max(dot_locations) > 1:
        dot_locations /= np.array([x, y])
    # Match up timestamps
    if video_timestamps is not None:
        mean_video_frame_time = np.mean(np.diff(video_timestamps))
        # Need for loop over dots if some dots 
        # have different timestamps than others
        tt = np.repeat(dot_timestamps[:, np.newaxis], len(video_timestamps), axis=1)
        tdif = video_timestamps - tt
        t_i, vframe_i = np.nonzero(np.abs(tdif) < (mean_video_frame_time / 2))
        # Downsample dot locations
        vframes = np.unique(vframe_i)
        print(vframes)
        n_dots_ds = len(vframes)
        print(n_dots_ds)
        dot_locations_ds = np.hstack([np.median(dot_locations[:, t_i[vframe_i==j]], axis=1)[:, None, :] for j in vframes])
        #print(dot_locations_ds.shape)
    else:
        dot_locations_ds = dot_locations
        vframes = np.arange(n_frames)
    # Plotting args
    n_dots, n_dot_frames, _ = dot_locations_ds.shape
    dot_colors = prep_input(dot_colors, n_dots)
    dot_widths = prep_input(dot_widths, n_dots)
    dot_markers = prep_input(dot_markers, n_dots)
    dot_labels = prep_input(dot_labels, n_dots)
    #print(dot_markers)
    # First set up the figure, the axis, and the plot element we want to animate
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(video_data[0], extent=extent, cmap='gray', aspect='auto')
    ax.set_xticks([])
    ax.set_yticks([])
    dots = []
    for dc, dw, dm, dl in zip(dot_colors, dot_widths, dot_markers, dot_labels):
        tmp = plt.scatter(0.5, 0.5, s=dw, c=dc, marker=dm, label=dl)
        dots.append(tmp)
    artists = (im, ) + tuple(dots)
    plt.close(fig.number)
    # initialization function: plot the background of each frame
    def init_func(fig, ax, artists):
        for d in dots:
            d.set_offsets([0.5, 0.5])
        im.set_array(np.zeros(im_shape))
        return artists 
    # animation function. This is called sequentially
    def update_func(i, artists, dot_locations, vframes):
        
        artists[0].set_array(video_data[i])
        # Also needs attention if different timecourses for different dots
        for j, artist in enumerate(artists[1:]):
            # may end up being: if i in vframes[j]:
            if i in vframes:
                _ = artist.set_offsets(dot_locations[j, i])
            else:
                _ = artist.set_offsets([-1,-1])
        return artists
    init = partial(init_func, fig=fig, ax=ax, artists=artists)
    update = partial(update_func, artists=artists, dot_locations=dot_locations_ds, vframes=vframes)
    # call the animator. blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, 
                func=update, 
                init_func=init,
                frames=n_frames, 
                interval=interval, 
                blit=True)
    return anim


def colormap_2d(
    data0,
    data1,
    cmap,
    vmin0=None,
    vmax0=None,
    vmin1=None,
    vmax1=None,
    map_to_uint8=False,
    ):
    """Map values in two dimensions to color according to a 2D color map image

    Parameters
    ----------
    data0 : array (1d)
        First dimension of data to map
    data1 : array (1d)
        Second dimension of data to map
    cmap : array (3d)
        image of values to use for 2D color map

    """
    if isinstance(cmap, str):
        try:
            import cortex as cx
        except:
            raise ImportError("You must install pycortex to use string names for 2d colormaps")
        # load pycortex 2D colormap
        cmapdir = cx.options.config.get('webgl', 'colormaps')
        colormaps = os.listdir(cmapdir)
        colormaps = sorted([c for c in colormaps if '.png' in c])
        colormaps = dict((c[:-4], os.path.join(cmapdir, c)) for c in colormaps)
        cmap = plt.imread(colormaps[cmap])

    norm0 = Normalize(vmin0, vmax0)
    norm1 = Normalize(vmin1, vmax1)

    d0 = np.clip(norm0(data0), 0, 1)
    d1 = np.clip(1 - norm1(data1), 0, 1)
    dim0 = np.round(d0 * (cmap.shape[1] - 1))
    # Nans in data seemed to cause weird interaction with conversion to uint32
    dim0 = np.nan_to_num(dim0).astype(np.uint32)
    dim1 = np.round(d1 * (cmap.shape[0] - 1))
    dim1 = np.nan_to_num(dim1).astype(np.uint32)

    colored = cmap[dim1.ravel(), dim0.ravel()]
    # May be useful to map r, g, b, a values between 0 and 255
    # to avoid problems with diff plotting functions...?
    if map_to_uint8:
        colored = (colored * 255).astype(np.uint8)
    return colored