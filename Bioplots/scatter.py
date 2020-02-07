#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# License           : MIT
# Author            : Jingxin Fu <jingxinfu.tj@gmail.com>
# Date              : 06/11/2019
# Last Modified Date: 28/01/2020
# Last Modified By  : Jingxin Fu <jingxinfu.tj@gmail.com>
__doc__ = """
Scatter plot
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import pandas as pd
import numpy as np
from collections import OrderedDict
import itertools
import adjustText
from .utils import align_marker, MidpointNormalize


class _BasePlotter:
    __slot__ = ['df', 'x', 'y', 'norm']

    def __init__(self, df, x, y):
        self.x = x
        self.y = y
        self.df = df.copy()

    def _initColorMap(self, value, vmax, vmin, center):
        values = self.df[value]
        if vmin is None:
            vmin = values.min()
        if vmax is None:
            vmax = values.max()
        if center is None:
            center = values.mean()

        self.norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=center)


class MtrxPlotter(_BasePlotter):
    def __init__(self, df, x, y, value, vmin=None, vmax=None, center=None):
        super().__init__(df, x=x, y=y)
        self._initColorMap(value=value, vmax=vmax, vmin=vmin, center=center)
        self.value = value

    def plot(self, stat, alternative_col=('black', 'purple'), row_cluster=None, col_cluster=None, direction=None, ax=None, cutoff=0.05, cbar=True, bar_title=None, marker='s', cmap='RdBu_r', cbar_kws={"shrink":1}, **plot_kws):
        ''' Scatter map to show relationship between col and row

        Parameters
        ----------
        stat : str
           Name of the statistical column
        alternative_col: str
           The alternative color list for row/col clusters, Defualt is ('black','gray')
        row_cluster: str
            The column name for row cluster, Default is None
        col_cluster: str
            The column name for col cluster, Defualt is None
        direction : str, optional
           Name of the compare direction column (The default is None)
        ax : matplotlib.pyplot.axes
            Defualt is None
        cutoff : float, optional
            The cutoff of statistical significance (the default is 0.05)
        cbar : bool, optional
            Show color bar (the default is True )
        bar_title : str, optional
            Title of the color bar (the default is None, which uses the name of the value column)
        marker : str, optional
            The shape of marker shape (the default is 's')
        cmap : str, optional
            The name of colormap platter (the default is 'RdBu_r')
        Returns
        -------
        (matplotlib.pyplot.figure, matplotlib.pyplot.axes)
        '''

        n_rows = self.df[self.y].unique().size
        n_cols = self.df[self.x].unique().size
        if ax == None:
            fig, ax = plt.subplots(1, 1, figsize=(
                n_cols // 2 + n_cols % 2, n_rows // 2 + n_rows % 2))
        else:
            fig = plt.gcf()
        # ax.grid(False)
        if bar_title == None:
            bar_title = self.value

        # Color tick label for better visualization
        if not row_cluster is None:
            self.df = self.df.sort_values(by=[self.x, row_cluster])
            max_size = self.df[self.y].map(lambda x: len(x)).max()
            uniq_rows = self.df[[self.y, row_cluster]
                                ].drop_duplicates().set_index(self.y)
            color_map = {x: alternative_col[i % 2]for i, x in enumerate(
                self.df[row_cluster].unique().tolist())}
            row_alt_colors = uniq_rows[row_cluster].map(color_map).to_dict()
        else:
            row_alt_colors = None

        if not col_cluster is None:
            self.df = self.df.sort_values(by=[self.y, col_cluster])
            # max_size = self.df[self.x].map(lambda x: len(x)).max()
            # self.df[self.x] = self.df[self.x].map(lambda x: ' '*int((max_size - len(x))*1.5) +x)
            uniq_cols = self.df[[self.x, col_cluster]
                                ].drop_duplicates().set_index(self.x)
            color_map = {x: alternative_col[i % 2] for i, x in enumerate(
                self.df[col_cluster].unique().tolist())}
            col_alt_colors = uniq_cols[col_cluster].map(color_map).to_dict()

        else:
            col_alt_colors = None

        stat_col = self.df[stat]
        # freeze the x axis and y axis ticks label order
        ax.scatter(self.x, self.y, data=self.df, c='white', s=400, label='')

        for name, determinator, s, zorder in [
            ("<", (stat_col < cutoff), 400, 2),
            (">=", (stat_col >= cutoff), 100, 1)
        ]:
            direction_edgeColor = {'<': 'k', '>=': 'none'}[name]
            name = '%s %s %.2f' % (stat, name, cutoff)
            filter_df = self.df.loc[determinator, :]

            if not direction is None:
                directions = sorted(filter_df[direction].unique())

                if len(directions) != 2:
                    raise ValueError('Only handle directions with 2 unique value, direction with ({}) is invalid'.format(
                        ','.join(directions)))
                else:
                    d_map = {directions[0]: 'left', directions[1]: 'right'}

                sub_dfs = [
                    (': '.join([d, name]),
                        align_marker('s', fillstyle=d_map[d]),
                        direction_edgeColor,
                        filter_df.loc[filter_df[direction] == d, :],
                        400
                     )
                    for d in directions
                ]
            else:
                sub_dfs = [(name, marker, 'none', filter_df, s)]

            for label, m, edgecolor, sub_df, point_size in sub_dfs:
                cur = ax.scatter(
                    self.x, self.y, c=self.value,
                    data=sub_df, norm=self.norm, label=label, s=point_size, marker=m, edgecolor=edgecolor, cmap=cmap, zorder=zorder,
                    **plot_kws
                )

        ax.legend(bbox_to_anchor=(1, 1), loc=3,
                  labelspacing=1, frameon=False)

        if cbar:
            # clb = plt.colorbar(cur,shrink=.5,orientation="horizontal",pad=.1)
            # [left, bottom, width, height]
            ax2_divider = make_axes_locatable(ax)
            #if (n_rows/n_cols) > 3:
            #    cax2 = ax2_divider.append_axes("top", size="3%", pad="2%")
            #    orientation='horizontal'
            #    #p[::-1]anchor = (0.5, 1.0)
            #    #anchor = (0.5,0.0)
            #else:
            cax2 = ax2_divider.append_axes("right", size="3%", pad="2%")
            orientation='vertical'
            #panchor = (1.0, 0.5)
            #anchor = (0.0,0.5)

            clb = plt.colorbar(cur,cax=cax2 ,**cbar_kws,orientation=orientation,
                 #              anchor=anchor, panchor=panchor,
                               fraction=0.036*(n_rows/n_cols))
            clb.ax.set_ylabel(bar_title)

        ax.set_xbound(-.5, n_cols)
        ax.set_ybound(-.5, n_rows)
        #ax.spines['bottom'].set_position(('outward', 10))
        #ax.spines['left'].set_position(('outward', 10))
        if n_cols > 7 or self.df[self.x].map(lambda x: len(x)).min() > 7:
            ax.tick_params(axis='x', rotation=90)

        # x_lim = ax.get_xlim()
        # y_lim = ax.get_ylim()
        # ax.set_xlim(x_lim[0]+1,x_lim[1]-1)
        # ax.set_ylim(y_lim[0]+.5,y_lim[1]-.5)

        # color ticks label
        if not row_alt_colors is None:
            # We need to draw the canvas, otherwise the labels won't be positioned and
            # won't have values yet.
            fig.canvas.draw()
            ylbls = ax.get_yticklabels()
            original = row_alt_colors[ylbls[0].get_text()]
            i = 0
            for lbl in ylbls:
                c = row_alt_colors[lbl.get_text()]
                if c != original:
                    ax.axhline(y=i-.5,xmin=ax.get_xlim()[0]-0.5,c='black',lw=5)
                    original = c
                # lbl.set_backgroundcolor(c)
                lbl.set_color(c)
                i += 1

        if not col_alt_colors is None:
            # We need to draw the canvas, otherwise the labels won't be positioned and
            # won't have values yet.
            fig.canvas.draw()

            xlbls = ax.get_xticklabels()
            original = col_alt_colors[xlbls[0].get_text()]
            i = 0
            for lbl in xlbls:
                c = col_alt_colors[lbl.get_text()]
                if c != original:
                    ax.axvline(x=i-.5,ymin=ax.get_ylim()[0]-0.5,c='black',lw=5)
                    original = c

                # lbl.set_backgroundcolor(c)
                lbl.set_color(c)

        return fig, ax


class AnnoPlotter(_BasePlotter):
    def __init__(self, df, x, y):
        super().__init__(df, x, y)

    def set_color_circle(self, ax, platter):

        if isinstance(platter, str):
            ax.set_prop_cycle(color=plt.get_cmap(platter).colors)
        elif isinstance(platter, list):
            ax.set_prop_cycle(color=platter)
        else:
            raise ValueError('Only string and list is available for platter.')

    def plot(self, color=None, marker=None, l_marker=(',', '+', '.', 'o', '*'), platter='Dark2', ax=None,
             vmax=None, vmin=None, center=None, bar_title=None, cmap='RdBu_r',
             label=None, **kwScatter):
        '''  Plotter with annotation

        Parameters
        ----------
        color : str, optional
            The name of the column to render points color on (the default is None)
        marker : str, optional
            The name of the column to render points shape on (the default is None)
        l_marker : tuple, optional
            The list of shapes want to use (the default is (',', '+', '.', 'o', '*'))
        platter : str, optional
            The name of color platter want to use (the default is 'Dark2')
        ax : matplotlib.pyplot.axes, optional
            (the default is None)
        vmax : float, optional
            The maximun normalize value (the default is None, which uses the maximun value on the color column )
        vmin : float, optional
            The minimum normalize value (the default is None, which which uses the minimum value on the color column)
        center : float, optional
            The center normalize value (the default is None, which which uses the center value on the color column)
        bar_title : str, optional
            The title of the color bar (the default is None, which uses the name of color column)
        cmap : str, optional
            The name of colormap platter (the default is 'RdBu_r')
        label : pandas.Series, optional
            The text label to annotate poinits, with corresponding index name in df.
            Default is None.

        Returns
        -------
        (matplotlib.pyplot.figure, matplotlib.pyplot.axes)
        '''

        if ax is None:
            fig, ax = plt.subplots(1, 1, dpi=100)
        else:
            fig = plt.gcf()
        # Parameter checking
        ## Choose color platter, the input variable must be str (name of platter)
        ## or a list of customized colors
        self.set_color_circle(ax, platter)
        l_marker = itertools.cycle(l_marker)
        # Case 1: Plotter when both of color and marker is True
        ## subset data frame by marker
        ## - Continuous plot:
        #   plot subset data with continuous change color and corresponding marker shape
        ## - Quantative plot:
        ##  subset the subseted data by color
        ##  plot points with corresponding color and marker shape
        if not color is None and not marker is None:

            if self.df[color].dtype.kind == 'f':
                self._initColorMap(value=color, vmax=vmax,
                                   vmin=vmin, center=center)
            else:
                color_groups = sorted(self.df[color].unique())

            marker_groups = sorted(self.df[marker].unique())
            for m_group in marker_groups:
                select_rows = (self.df[marker] == m_group)
                sub_df = self.df.loc[select_rows, :]
                m = next(l_marker)
                if self.df[color].dtype.kind == 'f':
                    # continunous
                    cur = ax.scatter(x=self.x, y=self.y, c=color, data=sub_df, label=m_group,
                                     marker=m, norm=self.norm, cmap=cmap, **kwScatter)
                else:
                    # quantative
                    # reset color cycle
                    self.set_color_circle(ax, platter)
                    for c_group in color_groups:
                        plot_df = sub_df.loc[sub_df[color] == c_group, :]
                        if plot_df.shape[0] > 0:
                            ax.scatter(x=self.x, y=self.y, data=plot_df, label=' : '.join([str(m_group), str(c_group)]),
                                       marker=m, **kwScatter)

        # Case 2: Plotter when only color is True
        ## - Continuous plot:
        ##   plot data with continuous change color
        ## - Quantative plot:
        ##  subset the subseted data by color
        ##  plot points with corresponding color
        elif not color is None:

            if self.df[color].dtype.kind == 'f':
                # continunous
                self._initColorMap(value=color, vmax=vmax,
                                   vmin=vmin, center=center)
                cur = ax.scatter(x=self.x, y=self.y, c=color, data=self.df,
                                 norm=self.norm, cmap=cmap, **kwScatter)
            else:
                # quantative
                color_groups = sorted(self.df[color].unique())
                for c_group in color_groups:
                    select_rows = (self.df[color] == c_group)
                    sub_df = self.df.loc[select_rows, :]
                    ax.scatter(x=self.x, y=self.y, data=sub_df,
                               label=c_group, **kwScatter)

        # Case 3: Plotter when only marker is True
        ##  subset the subseted data by marker
        ##  plot points with corresponding marker shape
        elif not marker is None:
            marker_groups = sorted(self.df[marker].unique())
            for m_group in marker_groups:
                select_rows = (self.df[marker] == m_group)
                sub_df = self.df.loc[select_rows, :]
                ax.scatter(x=self.x, y=self.y, data=sub_df, label=m_group,
                           marker=next(l_marker), **kwScatter)

        # Case 4: Plotter when neither color or marker is True
        ##  simple scatter plot
        else:
            ax.scatter(x=self.x, y=self.y, data=self.df, **kwScatter)

        if not color is None and self.df[color].dtype.kind == 'f':
            clb = plt.colorbar(cur, shrink=.4)
            if bar_title == None:
                clb.ax.set_title(color)
            else:
                clb.ax.set_title(bar_title)
        else:
            ax.legend(loc=(1.01, 0.5))

        ax.set(xlabel=self.x, ylabel=self.y)

        return fig, ax


def scatterMtrx(df, x, y, value, stat, vmin=None, vmax=None, center=None, ax=None, direction=None, cutoff=0.05, cbar=True, bar_title=None, marker='s', cmap='RdBu_r', cbar_kws={"shrink": 1, "pad": .01}, **kwPlot):
    ''' Scatter map to show relationship between col and row

    Parameters
    ----------
    stat : str
        Name of the statistical column
    direction : str, optional
        Name of the compare direction column (The default is None)
    ax : matplotlib.pyplot.axes
        Defualt is None
    cutoff : float, optional
        The cutoff of statistical significance (the default is 0.05)
    cbar : bool, optional
        Show color bar (the default is True )
    bar_title : str, optional
        Title of the color bar (the default is None, which uses the name of the value column)
    marker : str, optional
        The shape of marker shape (the default is 's')
    cmap : str, optional
        The name of colormap platter (the default is 'RdBu_r')
    Returns
    -------
    (matplotlib.pyplot.figure, matplotlib.pyplot.axes)
    '''
    plotter = MtrxPlotter(df=df, x=x, y=y, value=value,
                          vmin=vmin, vmax=vmax, center=center)
    fig, ax = plotter.plot(stat, direction=direction, ax=ax, cutoff=cutoff,
                           cbar=cbar, bar_title=bar_title, marker=marker, cmap=cmap, cbar_kws=cbar_kws, **kwPlot)

    return fig, ax


def volcano(
        df, stat, lfc, stat_cutoff, lfc_cutoff,
        num_show_labels=0, label_col=None, show_text_box=False,
        xlabel='Log fold change', ylabel=r'$ -log_{10}(P value)$', ax=None,
        map_platter={"-1": '#2166AC', "0": 'grey', "1": '#B2182B'},
        text_props=dict(weight='normal'),
        text_adjust={}, **kwargs):
    ''' Volcano plot

    Parameters
    ----------
    df : pandas.DataFrame
        A data frame contains need columns
    stat : str
        Name of the statistics column (e.g pvalue, must not be -np.log10 transformed )
    lfc : str
        Name of the lfc column
    stat_cutoff : float
        significance cutoff (must be -np.log10 transformed)
    lfc_cutoff : float
        lfc cutoff, only show points larger than abs(lfc) and points lower than -abs(lfc)
    num_show_lables: int
        How many labels shows on the plot (sorted by Pvalue and LFC)
    label_col: str
        The label column name
    show_text_box: boolean
        Whether draw box on the label text
    ax : matplotlib.pyplot.axes, optional
       Axes to draw to plot


    Returns
    -------
    [type]
        [description]
    '''

    # Format data frame, generate color groups to drop
    ## Only show colors for significant points
    df = df.copy()
    df[stat] = -np.log10(df[stat])
    sig_rows = (df[stat] > stat_cutoff)
    df['change'] = "0"
    ##Significantly down
    df.loc[(df[lfc] < -abs(lfc_cutoff)) & sig_rows, 'change'] = "-1"
    ##Significantly up
    df.loc[(df[lfc] >= abs(lfc_cutoff)) & sig_rows, 'change'] = "1"
    platter = [map_platter[x] for x in sorted(
        df['change'].unique().tolist()) if x in map_platter.keys()]

    ## Draw plots
    plotter = AnnoPlotter(df=df, x=lfc, y=stat)
    # Assign color to -1,0,1
    fig, ax = plotter.plot(color='change', platter=platter, ax=ax, **kwargs)
    ax.get_legend().remove()

    # Add reference lines
   # ax.axvline(-abs(lfc_cutoff), color='k', linestyle='--')
   # ax.axvline(abs(lfc_cutoff), color='k', linestyle='--')
   # ax.axhline(stat_cutoff, color='k', linestyle='--')

    # add subtitle
    ax.set(xlabel=xlabel, ylabel=ylabel)

    ## Add labels
    defaul_text_props = dict()
    defaul_text_props.update(text_props)

    df = df.loc[df['change'] != "0", :]
    if num_show_labels > 0 and df.shape[0] > 0:
        # annotlabel = df.reindex(df[lfc].sort_values().index)
        annotlabel = df.sort_values(by=lfc)
        if label_col != None:
            annotlabel.index = annotlabel[label_col]
        if num_show_labels < annotlabel.shape[0]:
            pos_annot = annotlabel.loc[annotlabel['change'] == '1', :]
            neg_annot = annotlabel.loc[annotlabel['change'] == '-1', :]
            n_pos = num_show_labels if num_show_labels < pos_annot.shape[0] else pos_annot.shape[0]
            n_neg = num_show_labels if num_show_labels < neg_annot.shape[0] else neg_annot.shape[0]
            annotlabel = pd.concat(
                [pos_annot.tail(n_pos), neg_annot.head(n_neg)])
        annotlabel['c'] = annotlabel['change'].map(
            {"1": '#B2182B', "-1": '#2166AC'})

        if show_text_box:
            texts = [
                ax.text(pos[lfc], pos[stat], label, defaul_text_props, color='k', bbox=dict(
                    boxstyle='round', facecolor=pos['c'], alpha=0.5))
                for label, pos in annotlabel[[lfc, stat, 'c']].to_dict('index').items()
            ]
        else:
            texts = [
                ax.text(pos[lfc], pos[stat], label,
                        defaul_text_props, color='k')  # pos['c'])
                for label, pos in annotlabel[[lfc, stat, 'c']].to_dict('index').items()
            ]
        adjustText.adjust_text(texts, **text_adjust)

    return fig, ax


def volcano_general(
        df, stat, lfc, stat_cutoff, lfc_cutoff, xlabel='Rho', ylabel='Value',
        num_show_labels=0, label_col=None, show_text_box=True,
        ax=None,
        map_platter={"-1": 'navy', "0": 'lightgrey', "1": 'maroon'},
        text_props=dict(weight='normal'),
        text_adjust={}, **kwargs):

    # Format data frame, generate color groups to drop
    ## Only show colors for significant points
    df = df.copy()
    sig_rows = (abs(df[stat]) > stat_cutoff)
    df['change'] = "0"
    ##Significantly down
    df.loc[(df[lfc] < -abs(lfc_cutoff)) & sig_rows, 'change'] = "-1"
    ##Significantly up
    df.loc[(df[lfc] >= abs(lfc_cutoff)) & sig_rows, 'change'] = "1"
    platter = [map_platter[x] for x in sorted(
        df['change'].unique().tolist()) if x in map_platter.keys()]

    ## Draw plots
    plotter = AnnoPlotter(df=df, x=lfc, y=stat)
    # Assign color to -1,0,1
    fig, ax = plotter.plot(color='change', platter=platter, ax=ax, **kwargs)
    ax.get_legend().remove()

    # Add reference lines
    ax.axvline(-abs(lfc_cutoff), color='k', linestyle='--')
    ax.axvline(abs(lfc_cutoff), color='k', linestyle='--')
    ax.axhline(stat_cutoff, color='k', linestyle='--')
    ax.axhline(-stat_cutoff, color='k', linestyle='--')

    # add subtitle
    ax.set(xlabel=xlabel, ylabel=ylabel)

    # add spines arround (required by collaborator)

    ## Add labels
    defaul_text_props = dict()
    defaul_text_props.update(text_props)

    df = df.loc[df['change'] != "0", :]
    if num_show_labels > 0 and df.shape[0] > 0:
        annotlabel = df.reindex(df[lfc].sort_values().index)
        if label_col != None:
            annotlabel.index = annotlabel[label_col]
        if num_show_labels < annotlabel.shape[0]:
            pos_annot = annotlabel.loc[annotlabel['change'] == '1', :]
            neg_annot = annotlabel.loc[annotlabel['change'] == '-1', :]
            n_pos = num_show_labels if num_show_labels < pos_annot.shape[0] else pos_annot.shape[0]
            n_neg = num_show_labels if num_show_labels < neg_annot.shape[0] else neg_annot.shape[0]
            annotlabel = pd.concat(
                [pos_annot.tail(n_pos), neg_annot.head(n_neg)])
        annotlabel['c'] = annotlabel['change'].map(
            {"1": 'maroon', "-1": 'navy'})

        if show_text_box:
            texts = [
                ax.text(pos[lfc], pos[stat], label, defaul_text_props, color='k', bbox=dict(
                    boxstyle='round', facecolor=pos['c'], alpha=0.5))
                for label, pos in annotlabel[[lfc, stat, 'c']].to_dict('index').items()
            ]
        else:
            texts = [
                ax.text(pos[lfc], pos[stat], label,
                        defaul_text_props, color=pos['c'])
                for label, pos in annotlabel[[lfc, stat, 'c']].to_dict('index').items()
            ]
        adjustText.adjust_text(texts, **text_adjust)

    return fig, ax


def annoScatter(df, x, y, color=None, marker=None, text_label=None,
                text_adjust={}, text_props=dict(weight='normal'), **kwargs):

    plotter = AnnoPlotter(df=df, x=x, y=y)
    fig, ax = plotter.plot(color=color, marker=marker, **kwargs)

    ## Add labels
    defaul_text_props = dict()
    defaul_text_props.update(text_props)
    if not text_label is None:
        texts = [
            ax.text(df.loc[pos, x], df.loc[pos, y],
                    text_label[pos], defaul_text_props)
            for pos in text_label.index
        ]
        adjustText.adjust_text(texts, **text_adjust)

    return fig, ax
