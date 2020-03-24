#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# License           : MIT
# Author            : Jingxin Fu <jingxinfu.tj@gmail.com>
# Date              : 16/10/2019
# Last Modified Date: 06/02/2020
# Last Modified By  : Jingxin Fu <jingxinfu.tj@gmail.com>
import os
from collections import OrderedDict
from textwrap import dedent

import pandas as pd
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt

from Bioplots.utils import fancy_scientific, pair_wise_compare

__all__ = ['box', 'violin', 'bar', 'lollipop']

class _Base(object):
    __slots__ = [
        'df','vertical',
        'group','subgroup','value'
        'rm_empty_space',
        'plot_props',
        'colors',
        'order','subgroup_order',
        'gps',
    ]

    def __init__(self, df, x, y, subgroup=None, order='median', subgroup_order='median', rm_empty_space=False, colors='Set1', stat_anno_median_max=False):
        self.df = df.copy()
        self.order = order # group order
        self.subgroup_order = subgroup_order # subgroup_order 
        self.rm_empty_space = rm_empty_space
        self.subgroup = subgroup # subgroup_order 
        self.colors = colors
        self._width = .7
        self._stat_anno_median_max = stat_anno_median_max
        # Establish plotting data
        self._establish_data(x,y)

    # Variable Checking 
    def _check_variables(self, x, y):
        """Check variables are correct"""

        if not isinstance(self.df, pd.DataFrame):
            raise ValueError('Please assign a pandas.DataFrame object to df.')
        ## check val and group element
        vals = [x,y]
        if self.subgroup:
            vals.append(self.subgroup)

        for v in vals:
            if not isinstance(v, str):
                raise KeyError(
                    'Please using column name (str variable) as index.')
            if not v in self.df.columns:
                raise KeyError(
                    'Cannot find column name {} in the given dataframe.'.format(v))

    def _infer_orient(self, x, y):
        """Determine the oriention of the plot"""

        if self.df[x].dtype in ['float', 'int']:
            x, y = y, x
            self.vertical = False
        elif self.df[y].dtype in ['float', 'int']:
            self.vertical = True
        else:
            raise ValueError(
                'Neither {} is numberic column.'.format(' and '.join([x, y])))
        return x,y

    def _order_check(self,group_name,group_order):
        """ check group orders """
        groups = self.df[group_name].unique().tolist() 
        if isinstance(group_order, list):
            group_order = group_order + [x for x in groups if not x in group_order]
        elif isinstance(group_order, str):
            if group_order == 'median':
                group_order = self.df.groupby(group_name)[self.value].median().sort_values().index.tolist()
            elif group_order== 'name':
                group_order = sorted(groups)
            else:
                raise KeyError('''Unknown str order options. Valid options: 
            1. "median": order by the median.
            2. "name": order by group name.
            ''')
        else:
            raise KeyError('''
        Unknown str order options.Please either specify group orders
        1. by inputting a list
        2. or a str variable("median": order by the median. "name": order by group name.)
        ''')

        return group_order

    def _map_colors(self,color_keys):
        if isinstance(self.colors,str):
            color_list = plt.get_cmap(self.colors).colors
        elif len(color_keys) > len(self.colors):
            raise ValueError('Please assign color to every group. %d color found but there are %d groups.' \
                % (len(color_keys),len(self.colors)))
        elif isinstance(self.colors,list):
            color_list = self.colors
        elif isinstance(self.colors,dict):
            color_list = [self.colors[k] for k in color_keys] # To order colors
        
        mapped_colors = OrderedDict()
        for k,v in zip(color_keys,color_list):
            mapped_colors[k] = v

        self.colors = mapped_colors
            
    def _establish_data(self,x,y):
        """ Generate orders and groupby objects for plotting """
        self._check_variables(x,y)
        self.group,self.value = self._infer_orient(x,y)

        ## Check order
        self.order = self._order_check(self.group, self.order)
        self.df[self.group] = pd.Categorical(self.df[self.group], ordered=True, categories=self.order)
        
        ## Grouping data
        if self.subgroup: 
            ## Check subgroup order
            self.subgroup_order = self._order_check(self.subgroup,self.subgroup_order)
            self.df[self.subgroup] = pd.Categorical(self.df[self.subgroup], ordered=True, categories=self.subgroup_order)
            group_names = [self.group,self.subgroup]

            self._map_colors(color_keys=self.subgroup_order)
        else:
            self._map_colors(color_keys=self.order)
            group_names = [self.group]

        self.gps = self.df.groupby(group_names)[self.value]


    @property
    def subgroup_offsets(self):
        """A list of center positions for plots when hue nesting is used."""
        n_levels = len(self.subgroup_order)
        each_width = self._width / n_levels
        offsets = np.linspace(each_width, self._width, n_levels)
        offsets -= offsets.mean()
        return offsets

    @property
    def width(self):
        """A float with the width of plot elements when group with same witdth is used."""
        if self.subgroup and not self.rm_empty_space:
            width = self._width / len(self.subgroup_order) * .9
        else:
            width = self._width

        return width
    
    def annotate_axes(self, ax, tick_locations, num_box):
        """Add labels and ticks to axes"""
        if self.vertical:
            xlabel, ylabel = self.group, self.value
            ax.set_xticks(tick_locations)
            ax.set_xticklabels(self.order)
            ax.set_xlim(-.5, num_box - .5, auto=None)
        else:
            xlabel, ylabel = self.value, self.group
            ax.set_yticks(tick_locations)
            ax.set_yticklabels(self.order)
            ax.set_ylim(-.5, num_box - .5, auto=None)

        ax.set(xlabel=xlabel,ylabel=ylabel)
    # Annotation Setting 
    def add_stat_annotation(self, ax, stat_anno,group_locus,stat_test='t-test',stat_display_cutoff=.05, stat_anno_by_star=True,color_stat_sig=True):
        """ Add statistic information to the box(violin) plot
        Parameters
        ----------
        ax : axes
        stat_anno : dict
            dict: {
                (group1 name,group2 name): pvalue
                ...
            }
        stat_anno_by_star : bool, optional
            Whether use the significant label instead of real number, by default True
        stat_display_cutoff: float,optional
            Cutoff of visible statistic annotation
        color_stat_sig: bool, optional
            Whether color statistical text by their significance
   
        """
        
        if stat_anno: 
            # Do statistic annotation
            if isinstance(stat_anno, bool):
                groups = list(group_locus.keys())
                if len(groups) < 2:
                    return
                stat_anno = pair_wise_compare(self.gps, test=stat_test, groups=groups)
                
            elif not isinstance(stat_anno, dict):
                raise ValueError('stat_anno only support input types as dict or bool')
        else: 
            # Don't do statistic annotation
            return 
        
        if self._stat_anno_median_max:
            val_lim = [self.df[self.value].min(), self.gps.quantile(.75).max()]
        else:
            val_lim = [self.df[self.value].min(), self.df[self.value].max()]
        
        # Value range setting
        val_ranges = val_lim[1] - val_lim[0]
        val_loc = val_lim[1] + val_ranges*.1
        min_loc = val_lim[0] - val_ranges*.1

        # Iterative step length setting
        line_step = val_ranges * .05
        text_step = val_ranges * .07
        anno_step = val_ranges * .15
        # text relative position
        if self.vertical:
            ha = 'center'
            va = 'bottom'
        else:
            ha = 'left'
            va = 'center'
            

        # stat_anno = {k: v for k, v in stat_anno.items() if v <
        #              stat_display_cutoff}
        for k, v in stat_anno.items():
            # group variable location
            g1_loc = group_locus[k[0]]
            g2_loc = group_locus[k[1]]
            g_loc = (g2_loc - (g2_loc-g1_loc)/2.0)
            # statistics postion
            line_x = [g1_loc]*2 + [g2_loc] * 2
            line_y = [val_loc] + [val_loc+line_step]*2 + [val_loc]
            text_x = g_loc
            text_y = val_loc+text_step

            if not self.vertical:
                line_x, line_y = line_y, line_x
                text_x, text_y = text_y, text_x

            # statistics text
            if color_stat_sig:
                if v < .01:
                    text_color = 'indianred'
                elif v < .05:
                    text_color = 'gold'
                else:
                    text_color = 'k'
            else:
                text_color = 'k'
          
            # show sigfinicance label
            if stat_anno_by_star:  
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

            ax.text(text_x, text_y, stat_str,
                    color=text_color, ha=ha, va=va)
            ax.plot(line_x, line_y, lw=1, c='k')
            val_loc += anno_step

        # Update figure value limits
        if self.vertical:
            ax.set_ylim([min_loc, val_loc+text_step])
        else:
            ax.set_xlim([min_loc, val_loc+text_step])

    
    def add_scatter(self, ax, boxData, center_position, color, scatter_kws):
        dodge = self.width*.3
        y = boxData
        x = np.random.uniform(-dodge, dodge, size=len(y)) + center_position
        if not self.vertical:
            x, y = y, x
        ax.scatter(x, y,color=color,zorder=3,**scatter_kws)

    def add_legend(self, ax, fill_box,color_box,legend_kws):
        handles = []
        for label,color in self.colors.items():
            # Box Shape
            if fill_box:
                handles.append(
                    Patch(edgecolor='k', facecolor=color,label=label)
                )
            # Box edge color shape
            elif color_box:
                handles.append(
                    Patch(edgecolor=color, facecolor='white', label=label)
                )
            # Point shape
            else:
                handles.append(
                    Line2D([0], [0], marker='o', label=label,
                           color='w', markerfacecolor=color)
                )
            
        ax.legend(handles=handles, loc=(1.05, 0), **legend_kws)

    def _restype(self, ax, grid):
        ## Add grid
        if grid == True:
            if self.vertical:
                ax.xaxis.grid()
            else:
                ax.yaxis.grid()

    def plot(self, ax, fill_box=False, color_box=True,
                stat_anno=True,
                grid=False,
                split_group_by_line=False,
                box_kws={},
                stat_kws={},
                scatter_kws={}, legend_kws={}):

            if ax == None:
                ax = plt.gca()

            
            for obj, v in box_kws.items():
                self.plot_props[obj] = v

            # Determine whether draw violin or boxplot
            tick_locations = []
            total_n = 0  # counter for number of box

            for i, (group_label, group_value) in enumerate(self.df.groupby(self.group)):
                tick_loci = i
                # There is subgroups, using nested loops to iterate subgroups
                group_locus = OrderedDict()
                if self.subgroup:
                    num_subgroups = 0
                    pre_group_last_position = 0
                    for j, (subgroup_label, subgroup_value) in enumerate(group_value.groupby(self.subgroup)):
                        boxData = subgroup_value[self.value].values
                        facecolor = self.colors[subgroup_label]
                        
                        if len(boxData) == 0: # ignore empty groups
                            continue

                        if self.rm_empty_space:
                            positions = total_n
                        else:
                            sg_i = self.subgroup_order.index(subgroup_label)
                            positions = i+self.subgroup_offsets[sg_i]

                        # For separating groups, record positions
                        if j == 0:
                            current_group_first_position = positions

                        self.draw_box(
                            ax=ax, boxData=boxData, positions=[positions], props=self.plot_props,
                            fill_box=fill_box, color_box=color_box, facecolor=facecolor, scatter_kws=scatter_kws)

                        # Ticks and Groups locus information
                        group_locus[(group_label, subgroup_label)] = positions
                        total_n += 1
                        num_subgroups += 1

                    self.add_stat_annotation(
                        ax, stat_anno=stat_anno, group_locus=group_locus, **stat_kws)

                    # Re-assign tick loci if we want to remove empty space caused by uneven number of subgroups
                    if self.rm_empty_space:
                        tick_loci = (total_n-num_subgroups) + (num_subgroups-1)/2.0

                    # Add reference lines for separating groups if there are subgroups?
                    if split_group_by_line and i>0:
                        x_loc = pre_group_last_position + \
                            (current_group_first_position-pre_group_last_position)/2.0

                        if self.vertical:
                            ax.axvline(x=x_loc, color='k', linewidth=1)
                        else:
                            ax.axhline(y=x_loc, color='k', linewidth=1)

                    # For separating groups, record positions
                    pre_group_last_position = positions 

                # There is no subgroups
                else:
                    positions = i
                    group_locus[group_label] = i
                    boxData = group_value[self.value].values
                    facecolor = self.colors[group_label]
                    self.draw_box(
                        ax=ax, boxData=boxData, positions=[positions], props=self.plot_props,
                        fill_box=fill_box, color_box=color_box, facecolor=facecolor, scatter_kws=scatter_kws)

                tick_locations.append(tick_loci)

            # Add statitical annotation in case there is no subgroup information but stat_anno is not None
            if not self.subgroup:
                group_locus = dict(zip(self.order, np.arange(len(self.order))))
                self.add_stat_annotation(
                    ax, stat_anno=stat_anno, group_locus=group_locus, **stat_kws)

            # Add legends
            self.add_legend(ax, fill_box, color_box, legend_kws)
            num_box = total_n if self.subgroup and self.rm_empty_space else len(self.order)
            self.annotate_axes(
                ax=ax, tick_locations=tick_locations, num_box=num_box)


            ### Overall layout
            # self._restype(ax=ax, grid=grid)
            plt.tight_layout()
            return ax

class BoxBase(_Base):

    def __init__(self, df, x, y, subgroup=None, order='median', subgroup_order='median', rm_empty_space=False,colors='Set1'):
        ### Box Properties Part
        super().__init__(df=df,x=x, y=y, subgroup=subgroup,order=order,
                         subgroup_order=subgroup_order, rm_empty_space=rm_empty_space,
                         colors=colors)
        self.plot_props = dict(
            widths=self.width,
            boxprops=dict(),
            medianprops=dict(linestyle='-', linewidth=2.5),
            showfliers=True,
            flierprops=dict(marker='+'),
            whiskerprops=dict(),
            capprops=dict(),
        )

    def draw_box(self,ax,boxData,positions,props,fill_box,color_box,facecolor,scatter_kws):
        # Draw violin or box plot?
        # Fill box by groups?
        if fill_box:
            props['boxprops']['facecolor'] = facecolor
            props['medianprops']['color'] = 'k'
            scatter_color = None

        elif color_box:
            props['boxprops']['edgecolor'] = facecolor
            props['medianprops']['color'] = facecolor
            props['boxprops']['facecolor'] = 'white'
            props['flierprops']['markeredgecolor'] = facecolor
            props['whiskerprops']['color'] = facecolor
            props['capprops']['color'] = facecolor
            scatter_color = None
            # Add scatter instead if do not fill box by groups
        else:
            props['boxprops']['facecolor'] = 'white'
            props['medianprops']['color'] = 'k'
            scatter_color = facecolor

       
        ax.boxplot(
                boxData, vert=self.vertical,
                positions=positions, patch_artist=True, **props
                )
        
        # Add scatter points for each box?
        if scatter_color:
            self.add_scatter(
                ax, boxData, center_position=positions, color=scatter_color, scatter_kws=scatter_kws)

   
class ViolinBase(_Base):
    def __init__(self, df, x, y, subgroup=None, order='median', subgroup_order='median', rm_empty_space=False, colors='Set1'):
        super().__init__(df=df, x=x, y=y, subgroup=subgroup, order=order,
                         subgroup_order=subgroup_order, rm_empty_space=rm_empty_space,
                         colors=colors)
        self.plot_props = dict(
            widths=self.width,
            showmeans=False, showmedians=False,
            showextrema=False
        )

    def draw_box(self, ax, boxData, positions, props, fill_box, color_box, facecolor, scatter_kws):

        parts = ax.violinplot(
                boxData, vert=self.vertical,
                positions=positions, **props
                )
        for pc in parts['bodies']:
            # Fill box by groups?
            if fill_box:
                pc.set_facecolor(facecolor)
                pc.set_edgecolor('k')
            else:
                pc.set_edgecolor(facecolor)
            pc.set_alpha(1)
        quartile1, medians, quartile3 = np.percentile(
            boxData, [25, 50, 75], axis=0)

       
        if self.vertical:
            ax.scatter(positions, medians, marker='o',
                       color='white', s=20, zorder=3)
            ax.vlines(positions, quartile1, quartile3,
                    color='k', linestyle='-', lw=5)
        else:
            ax.scatter(medians,positions, marker='o',
                       color='white', s=20, zorder=3)
            ax.hlines(positions, quartile1, quartile3,
                      color='k', linestyle='-', lw=5)
            

class BarBase(_Base):
    def __init__(self, df, x, y, subgroup=None, order='median', subgroup_order='median', rm_empty_space=False, colors='Set1'):
        super().__init__(df=df, x=x, y=y, subgroup=subgroup, order=order,
                         subgroup_order=subgroup_order, rm_empty_space=rm_empty_space,
                         colors=colors,stat_anno_median_max=True)

        self.plot_props = dict()

    def draw_box(self, ax, boxData, positions, props, fill_box, color_box, facecolor, scatter_kws):

        if len(boxData) > 1:
            quartile1, medians, quartile3 = np.percentile(boxData, [25, 50, 75], axis=0)
            y = medians
        else:
            y = boxData

        edgecolor = 'white'
        errbar_color = 'k'

        if color_box:
            facecolor, edgecolor = edgecolor,facecolor
            errbar_color = edgecolor

        if self.vertical:
            ax.bar(positions,y,self.width,color=facecolor,edgecolor=edgecolor)
            # Add error bar
            if len(boxData) > 1:
                ax.vlines(positions, quartile1, quartile3,
                          color=errbar_color, linestyle='-', lw=5)
        else:
            ax.barh(positions, y, self.width,
                    color=facecolor, edgecolor=edgecolor)
            # Add error bar
            if len(boxData) > 1:
                ax.hlines(positions, quartile1, quartile3,
                          color=errbar_color, linestyle='-', lw=5)


class LollipopBase(_Base):
    def __init__(self, df, x, y, subgroup=None, order='median', subgroup_order='median', rm_empty_space=False, colors='Set1'):
        super().__init__(df=df, x=x, y=y, subgroup=subgroup, order=order,
                         subgroup_order=subgroup_order, rm_empty_space=rm_empty_space,
                         colors=colors,stat_anno_median_max=True)

        self.plot_props = dict()

    def draw_box(self, ax, boxData, positions, props, fill_box, color_box, facecolor, scatter_kws):

        if len(boxData) > 1:
            quartile1, quartile3 = np.percentile(boxData, [25, 75], axis=0)
            scatter_x = positions * 2
            scatter_y = [quartile1, quartile3]
        else:
            quartile1, quartile3 = 0,boxData
            scatter_x = positions
            scatter_y = [quartile3]

        edgecolor = 'white'
        errbar_color = 'k'

        if self.vertical:
            ax.vlines(positions, quartile1, quartile3,
                        color='k', linestyle='-', lw=1)
        else:
            scatter_x, scatter_y = scatter_y, scatter_x
            ax.hlines(positions, quartile1, quartile3,
                        color='k', linestyle='-', lw=1)

        ax.scatter(scatter_x,scatter_y, color=facecolor, **scatter_kws)

def box(df, x, y, subgroup=None, order='median', subgroup_order='median', rm_empty_space=False, colors='Set1',
            ax=None, fill_box=False, color_box=True, stat_anno=True, split_group_by_line=False,
            grid=False,
            box_kws={},
            stat_kws={},
            scatter_kws={}, legend_kws={'frameon':False}
            ):
    """ Box plot"""
    plotter = BoxBase(df=df, x=x, y=y, subgroup=subgroup, order=order, subgroup_order=subgroup_order,
                         rm_empty_space=rm_empty_space, colors=colors)
    
    ax = plotter.plot(ax=ax,fill_box=fill_box,color_box=color_box,
                    stat_anno=stat_anno,split_group_by_line=split_group_by_line,grid=grid,box_kws=box_kws,stat_kws=stat_kws,
                    scatter_kws=scatter_kws, legend_kws=legend_kws)
    plt.tight_layout()
    return ax


def violin(df, x, y, subgroup=None, order='median', subgroup_order='median', rm_empty_space=False, colors='Set1',
            ax=None, fill_box=True,stat_anno=True, split_group_by_line=False,
            grid=False,
            box_kws={},
            stat_kws={},
            scatter_kws={}, legend_kws={'frameon': False}
            ):
    """ Violin plot"""
    plotter = ViolinBase(df=df, x=x, y=y, subgroup=subgroup, order=order, subgroup_order=subgroup_order,
                         rm_empty_space=rm_empty_space, colors=colors)

    ax = plotter.plot(ax=ax, fill_box=fill_box, color_box=False,
                      stat_anno=stat_anno, split_group_by_line=split_group_by_line, grid=grid, box_kws=box_kws, stat_kws=stat_kws,
                      scatter_kws=scatter_kws, legend_kws=legend_kws)

    return ax


def bar(df, x, y, subgroup=None, order='median', subgroup_order='median', rm_empty_space=False, colors='Set1',
               ax=None, fill_box=True, stat_anno=True, split_group_by_line=False,
               grid=False,
               box_kws={},
               stat_kws={},
               scatter_kws={}, legend_kws={'frameon': False}
               ):
    """ Bar plot"""
    plotter = BarBase(df=df, x=x, y=y, subgroup=subgroup, order=order, subgroup_order=subgroup_order,
                            rm_empty_space=rm_empty_space, colors=colors)
    color_box = not fill_box
    ax = plotter.plot(ax=ax, fill_box=fill_box, color_box=color_box,
                      stat_anno=stat_anno, split_group_by_line=split_group_by_line, grid=grid, box_kws=box_kws, stat_kws=stat_kws,
                      scatter_kws=scatter_kws, legend_kws=legend_kws)

    return ax


def lollipop(df, x, y, subgroup=None, order='median', subgroup_order='median', rm_empty_space=False, colors='Set1',
             ax=None, fill_box=True, stat_anno=True, split_group_by_line=False,
             grid=False,
             box_kws={},
             stat_kws={},
             scatter_kws={'alpha':0.7,'s':100,'marker':'o','edgecolor':'k'}, legend_kws={'frameon': False}
             ):
    """ lollipop plot"""
    plotter = LollipopBase(df=df, x=x, y=y, subgroup=subgroup, order=order, subgroup_order=subgroup_order,
                         rm_empty_space=rm_empty_space, colors=colors)
    color_box = not fill_box
    ax = plotter.plot(ax=ax, fill_box=fill_box, color_box=color_box,
                      stat_anno=stat_anno, split_group_by_line=split_group_by_line, grid=grid, box_kws=box_kws, stat_kws=stat_kws,
                      scatter_kws=scatter_kws, legend_kws=legend_kws)

    return ax


## Documentation
_distribution_docs = dict(
    # Shared function parameters
    input_params=dedent("""\
    x, y, subgroup : names of variables in ``df`` (subgroup is optional)
        Inputs for plotting long-form data. See examples for interpretation.\
        """),
    categorical_data=dedent("""\
    df : pandas.DataFrame
        Dataset for plotting.\
    """),

    order_vars=dedent("""\
    order, subgroup_order : lists of strings or a string ('median','name'), optional
        Order to plot the categorical levels in. \
        Default: groups are ordered by its median.  \
        """),

    color=dedent("""\
    color : matplotlib color, optional
        Defualt: 'Set1' \
        Color for all of the elements Or a palette name.\
    """),

    width=dedent("""\
    width : float, optional
        Width of a full element\
    """),

    
    ax_in=dedent("""\
    ax : matplotlib Axes, optional
        Axes object to draw the plot onto, otherwise uses the current Axes.\
    """),
    ax_out=dedent("""\
    ax : matplotlib Axes
        Returns the Axes object with the plot drawn onto it.\
    """),

    # Shared see also
    boxplot=dedent("""\
    box : A box plot.\
    """),
    violinplot=dedent("""\
    violin : A violin plot.\
    """),
    barplot=dedent("""\
    bar : A bar plot.\
    """),
    lollipopplot=dedent("""\
    lollipop : A lollipop plot.\
    """),

)


box.__doc__ = dedent("""\
    Draw a box plot to show interested feature distributions with respect to groups and subgroups.

    Parameters
    ----------
    {input_params}
    {categorical_data}
    {order_vars}
    {color}
    {width}
    {ax_in}
    kwargs : key, value mappings
        Other keyword arguments are passed through to
        :meth:`matplotlib.axes.Axes.boxplot`.

    Returns
    -------
    {ax_out}

    See Also
    --------
    {violinplot}
    {barplot}
    {lollipopplot}

    Examples
    --------

    Draw a single vertical boxplot grouped by ``temp`` variable:

    .. plot::
        :context: close-figs

        >>> import Bioplots as bpt
        >>> df = bpt.get_rdataset('beaver')
        >>> df['day'] = df['day'].map(str)
        >>> df['activ'] = df['activ'].map(str)
        >>> ax = bpt.box(df=df,x='day', y="temp")

    Draw a vertical boxplot grouped by ``temp`` and ``activ`` variables:

    .. plot::
        :context: close-figs

        >>> ax = bpt.box(df=df,x='day', y="temp",subgroup='activ')

    Remove the empty space on the frist group:

    .. plot::
        :context: close-figs

        >>> ax = bpt.box(df=df,x='day', y="temp",subgroup='activ',
        ...             rm_empty_space=True)

    Using dots to separate groups

    .. plot::
        :context: close-figs

        >>> ax = bpt.box(df=df,x='day', y="temp",subgroup='activ',
        ...             rm_empty_space=True,fill_box=False,color_box=False)
    
    Using box fillings to separate groups

    .. plot::
        :context: close-figs

        >>> ax = bpt.box(df=df,x='day', y="temp",subgroup='activ',
        ...             rm_empty_space=True,fill_box=True,color_box=False)

    Horizontal plot

    .. plot::
        :context: close-figs

        >>> ax = bpt.box(df=df,y='day', x="temp",subgroup='activ',
        ...              rm_empty_space=True,fill_box=True,color_box=False)
         
    """).format(**_distribution_docs)

violin.__doc__ = dedent("""\
    Draw a violin plot to show interested feature distributions with respect to groups and subgroups.

    Parameters
    ----------
    {input_params}
    {categorical_data}
    {order_vars}
    {color}
    {width}
    {ax_in}
    kwargs : key, value mappings
        Other keyword arguments are passed through to
        :meth:`matplotlib.axes.Axes.boxplot`.

    Returns
    -------
    {ax_out}

    See Also
    --------
    {boxplot}
    {barplot}
    {lollipopplot}

    Examples
    --------

    Draw a single vertical violin plot grouped by ``temp`` variable:

    .. plot::
        :context: close-figs

        >>> import Bioplots as bpt
        >>> df = bpt.get_rdataset('beaver')
        >>> df['day'] = df['day'].map(str)
        >>> df['activ'] = df['activ'].map(str)
        >>> ax = bpt.violin(df=df,x='day', y="temp")

    Draw a vertical boxplot grouped by ``temp`` and ``activ`` variables:

    .. plot::
        :context: close-figs

        >>> ax = bpt.violin(df=df,x='day', y="temp",subgroup='activ')

    Remove the empty space on the frist group:

    .. plot::
        :context: close-figs

        >>> ax = bpt.violin(df=df,x='day', y="temp",subgroup='activ',
        ...             rm_empty_space=True)

    Horizontal plot

    .. plot::
        :context: close-figs

        >>> ax = bpt.violin(df=df,y='day', x="temp",subgroup='activ',
        ...              rm_empty_space=True)
         
    """).format(**_distribution_docs)

bar.__doc__ = dedent("""\
    Draw a bar plot to show interested feature distributions with respect to groups and subgroups.

    Parameters
    ----------
    {input_params}
    {categorical_data}
    {order_vars}
    {color}
    {width}
    {ax_in}
    kwargs : key, value mappings
        Other keyword arguments are passed through to
        :meth:`matplotlib.axes.Axes.boxplot`.

    Returns
    -------
    {ax_out}

    See Also
    --------
    {violinplot}
    {boxplot}
    {lollipopplot}

    Examples
    --------

    Draw a single vertical boxplot grouped by ``temp`` variable:

    .. plot::
        :context: close-figs

        >>> import Bioplots as bpt
        >>> df = bpt.get_rdataset('beaver')
        >>> df['day'] = df['day'].map(str)
        >>> df['activ'] = df['activ'].map(str)
        >>> ax = bpt.bar(df=df,x='day', y="temp")

    Draw a vertical bar plot  grouped by ``temp`` and ``activ`` variables:

    .. plot::
        :context: close-figs

        >>> ax = bpt.bar(df=df,x='day', y="temp",subgroup='activ')

    Remove the empty space on the frist group:

    .. plot::
        :context: close-figs

        >>> ax = bpt.bar(df=df,x='day', y="temp",subgroup='activ',
        ...             rm_empty_space=True)
    
    Using box fillings to separate groups

    .. plot::
        :context: close-figs

        >>> ax = bpt.bar(df=df,x='day', y="temp",subgroup='activ',
        ...             rm_empty_space=True,fill_box=True)

    Horizontal plot

    .. plot::
        :context: close-figs

        >>> ax = bpt.bar(df=df,y='day', x="temp",subgroup='activ',
        ...              rm_empty_space=True,fill_box=True)
         
    """).format(**_distribution_docs)

lollipop.__doc__ = dedent("""\
    Draw a lollipop plot to show interested feature distributions with respect to groups and subgroups.

    Parameters
    ----------
    {input_params}
    {categorical_data}
    {order_vars}
    {color}
    {width}
    {ax_in}
    kwargs : key, value mappings
        Other keyword arguments are passed through to
        :meth:`matplotlib.axes.Axes.boxplot`.

    Returns
    -------
    {ax_out}

    See Also
    --------
    {violinplot}
    {barplot}
    {lollipopplot}

    Examples
    --------

    Draw a single vertical boxplot grouped by ``temp`` variable:

    .. plot::
        :context: close-figs

        >>> import Bioplots as bpt
        >>> df = bpt.get_rdataset('beaver')
        >>> df['day'] = df['day'].map(str)
        >>> df['activ'] = df['activ'].map(str)
        >>> ax = bpt.lollipop(df=df,x='day', y="temp")

    Draw a vertical lollipop plot grouped by ``temp`` and ``activ`` variables:

    .. plot::
        :context: close-figs

        >>> ax = bpt.lollipop(df=df,x='day', y="temp",subgroup='activ')

    Remove the empty space on the frist group:

    .. plot::
        :context: close-figs

        >>> ax = bpt.lollipop(df=df,x='day', y="temp",subgroup='activ',
        ...             rm_empty_space=True)

    Horizontal plot

    .. plot::
        :context: close-figs

        >>> ax = bpt.lollipop(df=df,y='day', x="temp",subgroup='activ',
        ...              rm_empty_space=True)
         
    """).format(**_distribution_docs)
