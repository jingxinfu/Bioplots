#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################################
# Created Date : Friday February 7th 2020                                      #
# Author: Jingxin Fu (jingxinfu.tj@gmail.com)                                  #
# ----------                                                                   #
# Last Modified: Friday February 7th 2020 2:38:14 pm                           #
# Modified By: Jingxin Fu (jingxinfu.tj@gmail.com)                             #
# ----------                                                                   #
# Copyright (c) Jingxin Fu 2020                                                #
################################################################################


import os
from warnings import warn
import pkg_resources

from scipy import stats 
from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib.colors as colors
from matplotlib import markers
from matplotlib.path import Path

__all__= ["get_rdataset"]

# R Datasets API
DATA_DIR = pkg_resources.resource_filename('Bioplots', 'data/')
DATASET = pd.read_csv(os.path.join(
    DATA_DIR, 'datasets.csv')).set_index('Item')['CSV'].to_dict()

def get_rdataset(dataset_name):
    """Get R's sample datasets by name for demonstration and practice. (Internet Connection Required).

    This is a collection from **Prof. Vincent Arel-Bundock**. It consist of over 1300 datasets that were originally distributed 
    alongside the statistical software environment R and some of its add-on packages.
    Main website: https://vincentarelbundock.github.io/Rdatasets/.
    
    Parameters
    ----------
    dataset_name : str
        Name of R's sample datasets.
        The list of available names is the **Item** column in http://vincentarelbundock.github.io/Rdatasets/datasets.html
    
    Returns
    -------
    pd.DataFrame
        A R's sample dataset.
    """  

    df = pd.read_csv(DATASET[dataset_name], index_col=0)
    return df
    
def pair_wise_compare(grouped_val, groups, test='t-test', return_z_score=False, **kwargs):
    func_map = {
        't-test': stats.ttest_ind,
        'wilcoxon': stats.wilcoxon,
        'mannwhitneyu': stats.mannwhitneyu, # For unequal N of wilcoxon
    }
    result = OrderedDict()
    for i, first in enumerate(groups):
        for second in groups[i+1:]:
            if return_z_score:
                try:
                    result[(first, second)] = func_map[test](grouped_val.get_group(first),
                                                            grouped_val.get_group(
                                                                second),
                                                            **kwargs)
                except ValueError:
                    warn('All numbers are identical in mannwhitneyu',
                         category=None, stacklevel=1)
                    result[(first, second)] =(0,1)
            else:
                try:
                    result[(first, second)] = func_map[test](grouped_val.get_group(first),
                                                            grouped_val.get_group(
                                                                second),
                                                            **kwargs
                                                            )[1]
                except ValueError:
                    warn('All numbers are identical in mannwhitneyu',
                         category=None, stacklevel=1)
                    result[(first, second)] =1
                    
    return result

def fancy_scientific(x):
    ''' Turn p value to the scientific format '''
    if x is not np.nan:
        tmp = '{:.2e}'.format(x).replace('e', ' x 10^').replace('^+', '^').replace('^0', '^').replace('^-0',
                                                                                                      '^-').replace(
            'x 10^', '\\times 10^{') + '}'
        return tmp.replace('\\times 10^{0}', '')
    else:
        return 'NA'

def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 10))  # outward by 10 points
            spine.set_smart_bounds(True)
        else:
            spine.set_color('none')  # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])


def align_marker(marker, halign='center', valign='middle', fillstyle='full'):
    """
    create markers with specified alignment.
    Parameters
    ----------
    marker : a valid marker specification.
      See mpl.markers
    halign : string, float {'left', 'center', 'right'}
      Specifies the horizontal alignment of the marker. *float* values
      specify the alignment in units of the markersize/2 (0 is 'center',
      -1 is 'right', 1 is 'left').
    valign : string, float {'top', 'middle', 'bottom'}
      Specifies the vertical alignment of the marker. *float* values
      specify the alignment in units of the markersize/2 (0 is 'middle',
      -1 is 'top', 1 is 'bottom').
    Returns
    -------
    marker_array : numpy.ndarray
      A Nx2 array that specifies the marker path relative to the
      plot target point at (0, 0).
    Notes
    -----
    The mark_array can be passed directly to ax.plot and ax.scatter, e.g.::
        ax.plot(1, 1, marker=align_marker('>', 'left'))
    """

    if isinstance(halign, str):
        halign = {'right': -1.,
                  'middle': 0.,
                  'center': 0.,
                  'left': 1.,
                  }[halign]

    if isinstance(valign, str):
        valign = {'top': -1.,
                  'middle': 0.,
                  'center': 0.,
                  'bottom': 1.,
                  }[valign]
    # Define the base marker
    bm = markers.MarkerStyle(marker, fillstyle=fillstyle)

    # Get the marker path and apply the marker transform to get the
    # actual marker vertices (they should all be in a unit-square
    # centered at (0, 0))
    m_arr = bm.get_path().transformed(bm.get_transform()).vertices

    # Shift the marker vertices for the specified alignment.
    m_arr[:, 0] += halign / 2
    m_arr[:, 1] += valign / 2

    return Path(m_arr, bm.get_path().codes)


class MidpointNormalize(colors.Normalize):
    """ map continuous value to map 
    If vmin or vmax is not given, they are initialized from the 
    minimum and maximum value respectively of the first input processed. 
    That is, __call__(A) calls autoscale_None(A). 
    If clip is True and the given value falls outside the range, the returned value will be 0 or 1, 
    whichever is closer.
    """
    def __init__(self,vmin=None, vmax=None,center=None, clip=False):
        self.midpoint = center
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

