#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# License           : MIT
# Author            : Jingxin Fu <jingxinfu.tj@gmail.com>
# Date              : 16/10/2019
# Last Modified Date: 06/02/2020
# Last Modified By  : Jingxin Fu <jingxinfu.tj@gmail.com>
import os
import seaborn as sns
from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from Tigger.visual import COLORS,fancy_scientific,calStats

from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')


__all__ = ['boxplot', 'violinplot', 'multi']


class _BasePlotter(object):
    __slots__ = [
        'df', 'x', 'y', 'widths', 'orient', 'order', 'color_palette', 'widths',
        'vals', 'groups', 'grouped_vals', 'color', 'vals_name'
    ]

    def __init__(self, df, x, y, widths=.5, order='median', color_palette='Set1'):
        # self.color_palette = ['indianred', 'steelblue',
        #                        'gold', 'lightgray', 'coral', 'k', 'seagreen']
        self.color_palette = color_palette
        self.df = df
        self.order = order
        self.check_variables(x, y)
        self.widths = widths

    def set_color_circle(self, ax, platter):
        if isinstance(platter, str):
            ax.set_prop_cycle(color=plt.get_cmap(platter).colors)
        elif isinstance(platter, list):
            ax.set_prop_cycle(color=platter)
        else:
            raise ValueError('Only string and list is available for platter.')

    def infer_orient(self, x, y):

        if self.df[x].dtype in ['float', 'int']:
            x, y = y, x
            self.orient = 'h'
        elif self.df[y].dtype in ['float', 'int']:
            self.orient = 'v'
        else:
            raise ValueError(
                'Neither {} is numberic column.'.format(' and '.join([x, y])))
        self.vals = self.df[y]
        self.groups = self.df[x]
        self.grouped_vals = self.df.groupby(x)
        self.vals_name = y

    def check_variables(self, x, y):
        if not isinstance(self.df, pd.DataFrame):
            raise ValueError('Please assign a pandas.DataFrame object to df.')

        ## check val and group element
        for v in [x, y]:
            if not isinstance(v, str):
                raise KeyError(
                    'Please using column name (str variable) as index.')
            if not v in self.df.columns:
                raise KeyError(
                    'Cannot find column name {} in the given dataframe.'.format(v))

        ## infer orientation
        self.infer_orient(x, y)
        ## Check order
        if isinstance(self.order, list):
            exist_group = sorted(self.groups.unique().tolist())
            exist_order_list = [x for x in self.order if x in exist_group]
            self.order = exist_order_list + \
                [x for x in exist_group if not x in self.order]
            # if sorted(self.order) != exist_group:
            #     return ValueError('''
            #     Unequal elements! Pleae input order list with following elements:
            #     {}
            #     '''.format(','.join(exist_group)))

        elif isinstance(self.order, str):
            if self.order == 'median':
                self.order = self.grouped_vals[self.vals_name].median(
                ).sort_values().index.tolist()
            elif self.order == 'name':
                self.order = sorted(self.groups.unique())
            else:
                raise KeyError('''
                Unknown str order options. Valid options:
                1. "median": order by the median.
                2. "name": order by group name.
                ''')
        else:
            raise KeyError('''
            Unknown str order options.Please either specify group orders
                1. by inputting a list
                2. or a str variable("median": order by the median. "name": order by group name.)
                ''')

        ## assign colors
        # self.color = self.color_palette[:len(self.order)]

    def add_stat_annotation(self, ax, stat_anno, test='t-test', stat_display_cutoff=.05, sig_label=True, val_lim=None):
        """ Add statistic information to the box(violin) plot

        Parameters
        ----------
        ax : axes
        stat_anno : bool or dict
            bool: indicate whether add the static info
            dict: {
                (group1 name,group2 name): pvalue
                ...
            }
        test : str, optional
            the statistic test you want to use, by default 't-test'
            Option:
            't-test' and 'wilcoxon'
        sig_label : bool, optional
            Whether use the significant label instead of real number, by default True
        val_lim : list, optional
            [bottom line for the plot, top line for the plot], by default None
        """
        if isinstance(stat_anno, bool):
            if stat_anno:
                stat_anno = calStats(
                    self.grouped_vals[self.vals_name], self.order, test=test)
            else:
                return
        elif not isinstance(stat_anno, dict):
            return

        if val_lim == None:
            val_lim = [self.vals.min(), self.vals.max()]

        val_ranges = val_lim[1] - val_lim[0]
        val_loc = val_lim[1] * 1.1
        line_step = val_ranges * .05
        text_step = val_ranges * .07
        anno_step = val_ranges * .15
        # text relative position
        if self.orient == 'v':
            ha = 'center'
            va = 'bottom'
        else:
            ha = 'left'
            va = 'center'
        stat_anno = { k:v for k,v in stat_anno.items() if v < 0.05}
        for k, v in stat_anno.items():
            # group variable location
            g1_loc = self.order.index(k[0])+1
            g2_loc = self.order.index(k[1])+1
            g_loc = (g2_loc - (g2_loc-g1_loc)/2.0)

            # statistics text
            if v < .01:
                text_color = 'k'
            elif v < .05:
                text_color = 'k'
            else:
                text_color = 'k'

            # statistics postion
            line_x = [g1_loc]*2 + [g2_loc] * 2
            line_y = [val_loc] + [val_loc+line_step]*2 + [val_loc]
            text_x = g_loc
            text_y = val_loc+text_step

            if self.orient == 'h':
                line_x, line_y = line_y, line_x
                text_x, text_y = text_y, text_x

            if sig_label:  # show sigfinicance label
                if v <= .0001:
                    stat_str = '****'
                elif v <= .001:
                    stat_str = '***'
                elif v <= .01:
                    stat_str = '**'
                elif v <= .05:
                    stat_str = '*'
                else:
                    stat_str = 'NS'
            else:
                stat_str = r'${}$'.format(fancy_scientific(v))

            if v <= stat_display_cutoff:
                ax.text(text_x, text_y, stat_str,
                        color=text_color, ha=ha, va=va)
                ax.plot(line_x, line_y, lw=1.5, c='grey')
            val_loc += anno_step
        # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    def add_scatter(self, boxDat, ax, color_dict=None, **kwargs):
        dodge = self.widths*.3
        if color_dict is None:  # is Not : Customize color platter to correpsonding label
            self.set_color_circle(ax, self.color_palette)

        for i, col in enumerate(boxDat):
            y = col
            x = np.random.uniform(-dodge, dodge, size=len(y)) + i+1
            if self.orient == 'h':
                x, y = y, x
            label = self.order[i]
            if color_dict is None:
                ax.scatter(x, y, label=label, **kwargs)
            else:
                ax.scatter(x, y, label=label, c=color_dict[label], **kwargs)

        handles, labels = ax.get_legend_handles_labels()
        return handles, labels

    def _restype(self, ax, grid):
        ## Add grid
        if grid == True:
            if self.orient == 'v':
                ax.xaxis.grid()
            else:
                ax.yaxis.grid()


class BoxPlotter(_BasePlotter):

    def __init__(self, df, x, y, widths=.5, platter='Dark2', order='median'):
        super().__init__(df, x, y, widths=widths, color_palette=platter, order=order)

    def plot(
            self, ax,
            add_scatter=True, show_legend=False, color_dict=None,
            stat_anno=True, stat_test='t-test',
            show_tick_name=True, box_kw={},
            stat_display_cutoff=.05,
            grid=False, **scatter_kw):

        if ax == None:
            ax = plt.gca()

        ### Box Part
        props = dict(
            boxprops=dict(color='k'),
            medianprops=dict(linestyle='-', linewidth=2.5, color='k'),
            widths=self.widths,
            showfliers=False,
        )
        for obj, v in box_kw.items():
            props[obj] = v

        boxDat = [np.asarray(self.grouped_vals.get_group(g)[self.vals_name].dropna())
                  for g in self.order]
        # determine box orient
        vert = self.orient == "v"
        ax.boxplot(boxDat, vert=vert, **props)

        # statistic annotation
        self.add_stat_annotation(
            ax=ax, stat_anno=stat_anno, test=stat_test, stat_display_cutoff=stat_display_cutoff)

        if show_tick_name:
            tick_labels = self.order
        else:
            tick_labels = []

        if self.orient == 'v':
            ax.set_xticklabels(tick_labels)
            ax.set_ylabel(self.vals_name)
        else:
            ax.set_xlabel(self.vals_name)
            ax.set_yticklabels(tick_labels, rotation=90)

        ### Scatter Part
        ## annotate stat information on boxplot
        if add_scatter:
            handles, labels = self.add_scatter(ax=ax, boxDat=boxDat, color_dict=color_dict,
                                               **scatter_kw)
            if show_legend:
                ax.legend(handles=handles, loc=(1.01, 0.5), labels=labels)
        else:
            handles, labels = None, None

        ### Overall layout
        self._restype(ax=ax, grid=grid)

        return ax, handles, labels


class ViolinPlotter(_BasePlotter):

    def __init__(self, df, x, y, widths=.5, platter='Dark2', ax=None, order='median'):
        super().__init__(df, x, y, widths=widths, color_palette=platter, order=order)

    def set_axis_style(self, ax, labels):
        if self.orient == 'v':
            ax.get_xaxis().set_tick_params(direction='out')
            ax.xaxis.set_ticks_position('bottom')
            ax.set_xticks(np.arange(1, len(labels) + 1))
            ax.set_xticklabels(labels)
            ax.set_xlim(0.25, len(labels) + 0.75)
        else:
            ax.get_yaxis().set_tick_params(direction='out')
            ax.yaxis.set_ticks_position('left')
            ax.set_yticks(np.arange(1, len(labels) + 1))
            ax.set_yticklabels(labels)
            ax.set_ylim(0.25, len(labels) + 0.75)

    def calStatLine(self):
        medians = self.grouped_vals.quantile(.5)[self.order].values
        q1 = self.grouped_vals.quantile(.25)[self.order].values
        q3 = self.grouped_vals.quantile(.75)[self.order].values
        iqr = np.array([q1, q3])
        upper_adjacent_value = q3 + (q3 - q1) * 1.5
        lower_adjacent_value = q1 - (q3 - q1) * 1.5
        return medians, iqr, np.array([lower_adjacent_value, upper_adjacent_value])

    def plot(self, ax, stat_anno=True, test='t-test', grid=False, draw_whiskers=False, violin_kw={}):
        if ax == None:
            ax = plt.gca()
        boxDat = [np.asarray(self.grouped_vals.get_group(g).dropna())
                  for g in self.order]
        self.set_axis_style(ax, labels=self.order)
        ### violin_kw Default
        props = dict(
            widths=self.widths * len(self.order),
        )
        for obj, v in violin_kw.items():
            props[obj] = v

        vert = self.orient == "v"
        parts = ax.violinplot(boxDat, vert=vert, showmeans=False, showmedians=False,
                              showextrema=False, **props)
        for pc in parts['bodies']:
            # pc.set_facecolor(c)
            pc.set_edgecolor('black')
            pc.set_alpha(.5)

        ## Add statistical lines
        medians, iqr, whiskers = self.calStatLine()
        inds = range(1, len(medians)+1)
        if draw_whiskers is True:
            x_list = [[inds, inds], [inds, inds], [inds]]
            y_list = [iqr, whiskers, medians]
            name_list = ['iqr', 'whiskers', 'medians']
            lw_list = [5, 1, None]
        else:
            x_list = [[inds, inds], [inds]]
            y_list = [iqr, medians]
            name_list = ['iqr',  'medians']
            lw_list = [5, None]
        # plot IQR , whiskers, and median circles
        for x, y, n, lw in zip(x_list, y_list, name_list, lw_list):
            if self.orient == 'h':
                x, y = y, x

            if n != 'medians':
                ax.plot(x, y, 'k-', lw=lw)
            else:
                ax.scatter(x, y, marker='o', c='white', s=5, zorder=3)

        if stat_anno == True:
           self.add_stat_annotation(ax=ax, test=test)

        self._restype(ax=ax, grid=grid)

        return ax


def boxplot(df, x, y, widths=.5, order='median', platter='Set1',
            ax=None, add_scatter=True, show_legend=False, color_dict=None,
            stat_anno=True, stat_test='t-test',
            show_tick_name=True, box_kw={},
            stat_display_cutoff=.05,
            grid=False, **scatter_kw
            ):
    """

    Parameters
    ----------
    df : pd.DataFrame
    x : str
        group(vertical boxplot) or value column name in the data frame.

    y : str
        group(horizontal boxplot) or value column name in the data frame.
    widths : float, optional
        relative width for box, by default .5
    order : str or list, optional
        group order, by default 'median': groups are ordered by median
    platter : str, optional
        color platter , by default 'Set1'
    ax : matplotlib.pyplot.axes, optional
        axes obj, by default None
    add_scatter : bool, optional
        Whether draw scatter points over box,by default True
    show_legend : bool, optional
        whether show legend on the plot, by default False
    color_dict : dict, optional
        specify color map fpr each group, by default None
    stat_anno : bool or dict
            By default: true
            bool: indicate whether add the static info
            dict: {
                (group1 name,group2 name): pvalue
                ...
            }
    stat_test : str, optional
        the statistic test you want to use, by default 't-test'
        Option:
            't-test' and 'wilcoxon'
    show_tick_name : bool, optional
        whether show the group name, by default True
    box_kw : dict, optional
        kw for plt.boxplot, by default {}
    stat_display_cutoff: float,optional
        the maximum cutoff to display statistic annotation, by default .05
    grid : bool, optional
        whether draw grid, by default True
    Returns
    -------
    ax : matplotlib.pyplot.axes
    handles: legend handles
    labels: corresponding legend labels
    plotter: _basePlotter obj
    """
    plotter = BoxPlotter(df=df, x=x, y=y, widths=widths,
                         platter=platter, order=order)
    ax, handles, labels = plotter.plot(
        ax, add_scatter=add_scatter, show_legend=show_legend, color_dict=color_dict,
        stat_anno=stat_anno, stat_test=stat_test, show_tick_name=show_tick_name, box_kw=box_kw,
        stat_display_cutoff=stat_display_cutoff,
        grid=False, **scatter_kw)

    return ax, handles, labels, plotter


def violinplot(df, x, y, widths=.5, order='median', platter='Set1',
               ax=None, show_legend=False, color_dict=None,
               stat_anno=True, stat_test='t-test',
               grid=False, violin_kw={}, show_tick_name=False,
               ):

    plotter = ViolinPlotter(df=df, x=x, y=y, widths=widths,
                            platter=platter, order=order)
    ax, handles, labels = plotter.plot(ax, show_legend=show_legend, color_dict=color_dict,
                                       stat_anno=stat_anno, stat_test=stat_test, grid=grid, violin_kw=violin_kw, show_tick_name=show_tick_name)
    return ax, handles, labels, plotter


def multi(func, df, x, y, factor, factor_order='median', stat_result=None, facet=None, sharex=False, sharey=False, fig_kws={}, platter='Set2', **kwargs):

    # figure out the subgroup name
    if df[x].dtype in ['float', 'int']:
        hue_col, value_col = y, x
    else:
        hue_col, value_col = x, y

    # order each group by correspinding input order
    if factor_order == 'median':
        factor_list = df.groupby(factor)[value_col].median(
        ).sort_values(ascending=True).index.tolist()
    elif factor_order == 'name':
        factor_list = df[factor].unique().sort_values(
            ascending=True).index.tolist()
    elif isinstance(factor_order,list):
        factor_list = factor_order
    elif factor_order is None:
        factor_list = df[factor].unique().tolist()
    else:
        raise ValueError(
            "Don't know how to order group by %s.Optional parameter for factor_order are: None, 'name', and 'median'" % factor_order)
    # way to make sure the legend are right

    if not 'color_dict' in kwargs.keys():
        # Digest order information
        order_list = df[hue_col].unique()
        if 'order' in kwargs.keys() and isinstance(kwargs['order'], list):
                order_list = kwargs['order'] + \
                    [x for x in order_list if not x in kwargs['order']]

        kwargs['color_dict'] = {
            f: color for f, color in zip(
                order_list,
                plt.get_cmap(platter).colors[:len(order_list)]
            )
        }

    if facet == None:
        facet = (1, len(factor_list))
    fig, axs = plt.subplots(*facet, sharex=sharex, sharey=sharey, **fig_kws)
    axs = axs.flatten()
    all_labels = []
    all_handles = []

    for ax, f in zip(axs[:len(factor_list)], factor_list):
        sub_dat = df.loc[df[factor] == f, :]
        sub_dat = sub_dat.dropna()
        if not stat_result is None:
            kwargs['stat_anno'] = stat_result[f]
        ax, handles, labels, plotter = func(
            df=sub_dat, x=x, y=y, ax=ax, **kwargs)
        if plotter.orient == 'v':
            ax.set_xlabel(f)  # ,rotation=90,ha='center',va='top')
        else:
            ax.set_ylabel(f, rotation=0, ha='right', va='center')
        for hdl, lbl in zip(handles, labels):
            if not lbl in all_labels:
                all_labels.append(lbl)
                hdl.set_alpha(1)
                all_handles.append(hdl)

    # put legend on top right corner
    if len(all_labels) > 1:
        axs[facet[1]-1].legend(handles=all_handles, loc=(1.01, 0),
                               labels=all_labels, markerscale=5, prop={'size': 15})

    for empty_ax in axs[len(factor_list):]:
        empty_ax.set_axis_off()

    return fig


def multiFlex(func, df, x, y, factor, factor_order='median', stat_result=None, fig_kws={}, platter='Set1', **kwargs):

    # figure out the subgroup name
    if df[x].dtype in ['float', 'int']:
        hue_col, value_col = y, x
        horizontal = True
    else:
        horizontal = False
        hue_col, value_col = x, y
    # figure out size for each subgroup
    num_subgroup = df[[hue_col, factor]].drop_duplicates()[
        factor].value_counts().to_frame()
    num_subgroup.columns = ['Num']
    # order each group by correspinding input order
    if factor_order == 'median':
        factor_list = df.groupby(factor)[value_col].median(
        ).sort_values(ascending=True).index.tolist()
    elif factor_order == 'name':
        factor_list = df[factor].unique().sort_values(
            ascending=True).index.tolist()
    elif factor_order is None:
        factor_list = df[factor].unique().tolist()
    else:
        raise ValueError(
            "Don't know how to order group by %s.Optional parameter for factor_order are: None, 'name', and 'median'" % factor_order)
    # way to make sure the legend are right

    if not 'color_dict' in kwargs.keys():
        # Digest order information
        order_list = df[hue_col].unique()
        if 'order' in kwargs.keys() and isinstance(kwargs['order'], list):
                order_list = kwargs['order'] + \
                    [x for x in order_list if not x in kwargs['order']]

        kwargs['color_dict'] = {
            f: color for f, color in zip(
                order_list,
                plt.get_cmap(platter).colors[:len(order_list)]
            )
        }
    fig = plt.figure(**fig_kws)
    num_box = num_subgroup['Num'].sum()
    num_subgroup = num_subgroup.loc[factor_list, :]  # re-order list
    num_subgroup['End'] = num_subgroup['Num'].cumsum()

    if not horizontal:
        gs = gridspec.GridSpec(1, num_box)

    else:
        gs = gridspec.GridSpec(num_box, 1)

    all_labels = []
    all_handles = []
    for i, f in enumerate(factor_list):
        row = num_subgroup.loc[f, :]
        if horizontal:
            ax_prev = ax if i > 0 else None
            ax = fig.add_subplot(
                gs[row['End']-row['Num']:row['End'], :], sharex=ax_prev)
            if i == 0:
                label_ax = ax
        else:
            ax_prev = ax if i > 0 else None
            ax = fig.add_subplot(
                gs[:, row['End']-row['Num']:row['End']], sharey=ax_prev)
            if (i+1) == len(factor_list):
                label_ax = ax

        sub_dat = df.loc[df[factor] == f, :]
        sub_dat = sub_dat.dropna()
        if not stat_result is None:
            kwargs['stat_anno'] = stat_result[f]
        ax, handles, labels, plotter = func(
            df=sub_dat, x=x, y=y, ax=ax, **kwargs)
        if plotter.orient == 'v':
            ax.set_xlabel(f, rotation=90, ha='center', va='top')
            if i > 0:
                ax.get_yaxis().set_visible(False)
            else:
                ax.set(ylabel=value_col)
        else:
            if i < (len(factor_list) - 1):
                ax.get_xaxis().set_visible(False)
            else:
                ax.set(xlabel=value_col)
            ax.set_ylabel(f, rotation=0, ha='right', va='center')
        for hdl, lbl in zip(handles, labels):
            if not lbl in all_labels:
                all_labels.append(lbl)
                hdl.set_alpha(1)
                all_handles.append(hdl)

    # put legend on top right corner
    if len(all_labels) > 1:
        label_ax.legend(handles=all_handles, loc=(1.05, 0),
                        labels=all_labels, markerscale=5, prop={'size': 15})

    #for empty_ax in axs[len(factor_list):]:
    #    empty_ax.set_axis_off()

    return fig


def snsScatter(sub_dt, groupName, var1, var2, xlim1, xlim2, Xlab, topic, namep, output):
    markers = 'o'
    if 'Low' in groupName:
        markers = 'X'
    sampleNumber = sub_dt.shape[0]
    corr = stats.pearsonr(sub_dt[var1], sub_dt[var2])
    corr = [np.round(c, 2) for c in corr]
    text = 'r=%s, p=%s' % (corr[0], corr[1])
    sns.set(color_codes=True)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax = sns.scatterplot(x=var1, y=var2, hue="Group", style="Group",
                         data=sub_dt, markers=markers)
    ax = sns.regplot(x=var1, y=var2, data=sub_dt)
    ax.text(0.7, 0.6, text, fontsize=12,  ha='center',
            va='center', transform=ax.transAxes)
    ax.set(title=str(sampleNumber) + ' cell lines',
           xlabel=Xlab, ylabel=namep+' Expression')
    ax.set(xlim=(xlim1*0.8, xlim2*1.2))

    fig.savefig(os.path.join(
        output, 'BiomarkerCorrelation', topic+''+groupName.split(' ')[1]+'.png'),
        dpi=300, bbox_inches='tight')
    plt.close('all')
