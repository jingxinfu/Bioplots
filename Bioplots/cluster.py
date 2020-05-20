#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################################
# Created Date : Saturday March 28th 2020                                      #
# Author: Jingxin Fu (jingxinfu.tj@gmail.com)                                  #
# ----------                                                                   #
# Last Modified: Saturday March 28th 2020 8:47:31 pm                           #
# Modified By: Jingxin Fu (jingxinfu.tj@gmail.com)                             #
# ----------                                                                   #
# Copyright (c) Jingxin Fu 2020                                                #
################################################################################

from textwrap import dedent
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend import Legend
import adjustText

from .utils import MidpointNormalize

__all__ = ["heatmap", "volcano", 'scatterplot']

class _BaseScatter:
    __slot__ = ['x', 'y','value','size','marker','param']

    def __init__(self,df,x,y,color=None,size=None,marker=None):
        """ Initialize and check variables"""
        self.param = {
            'title': {},
            'default_size':10,
            'default_marker':'o',
            'default_color':'steelblue',
            }
        self._check_assign_variables(df,x,y,color,size,marker)
        
    
    def _check_assign_variables(self, df, x, y,color, size, marker):
        """Check variables are correct and assign variables"""
        if not isinstance(df, pd.DataFrame):
            raise ValueError('Please assign a pandas.DataFrame object to df.')
        ## check variables
        vals = [x, y]

        if color:
            vals.append(color)
            self.param['title']['color'] = color
        if size:
            vals.append(size)
            self.param['title']['size'] = size
        if marker:
            vals.append(marker)
            self.param['title']['marker'] = marker

        for v in vals:
            if not isinstance(v, str):
                raise KeyError(
                    'Please using column name (str variable) as index.')
            if not v in df.columns:
                raise KeyError(
                    'Cannot find column name {} in the given dataframe.'.format(v))

        # Assign values
        self.x,self.y = df[x],df[y]
        self.color = df[color] if color else None
        self.size = df[size] if size else None
        self.marker = df[marker] if marker else None
        
    def orderTicks(self,x_order=None,y_order=None):
        """ Customize axis orders and output labels for x and y axis
        and return ordered tick labels.

        Parameters
        ----------
        x_order : list, Optional
            Order list of x axis
        y_order : list, Optional
            Order list of y axis

        Returns
        -------
        tuple
            Ordered tick labels. (xticklabels,yticklabels)
        """
        xticklabels =  sorted(self.x.unique().tolist()) 
        yticklabels =  sorted(self.y.unique().tolist()) 
        if x_order:
            if set(x_order) == set(yticklabels):
                xticklabels = x_order
            else:
                raise ValueError('Number of elements in x_order is not equal to number of xticks')

        if y_order:
            if set(y_order) == set(yticklabels):
                yticklabels = y_order
            else:
                raise ValueError('Number of elements in y_order is not equal to number of yticks')
        
        # Assign position to x and y axis 
        self.x = self.x.map({k:i for i,k in enumerate(xticklabels)})
        self.y = self.y.map({k:i for i,k in enumerate(yticklabels)})

        return xticklabels, yticklabels

    def colorMap(self, palette='Set1',vmax=None, vmin=None, center=None):
        """ Map values to color

        Parameters
        ----------
        palette : str, list, or dict, optional 
            Paltte to control the color mapping options. Default is 'Set1' 
        vmax : float, optional
            Max value on the continuous color range mapping, by default None
        vmin : float, optional
            min value on the continuous color range mapping, by default None
        center : float, optional
            center value on the continuous color range mapping, by default None
        Raises
        ------
        ValueError
            The number of assigned colors is not equivalent to the number 
            of groups.
        KeyError
            There are groups without specified assigned colors in the platte dict.
        """
        if np.issubdtype(self.color.dtype, np.number) and isinstance(palette, str):
            # For continuous value
            if center is None: # assign center to the mean point by default
                center = self.color.mean()
            # To control the colorRange 
            # in case that we need to draw more than one plots with shared colorbar
            if vmax is None:
                vmax = self.color.max()
            if vmin is None:
                vmin = self.color.min()
            self.param['color_norm'] = MidpointNormalize(vmax=vmax,vmin=vmin,center=center)

        else: 
            groups = sorted(self.color.unique().tolist())
            # Option 1: Input a name of palette
            if isinstance(palette, str):
                # Select correspondant number of colors from the palette
                color_values = plt.get_cmap(palette).colors
                color_maps = {k:v for k,v in zip(groups,color_values)}
            
            # Option 2: Input a list or dict to explicitly assign colors to value 
            elif len(palette) < len(groups):
                raise ValueError('Please assign color to every group. %d color found but there are %d groups.'
                                % (len(palette),len(groups)))

            elif isinstance(palette, list):
                color_values = palette 
                color_maps = {k:v for k,v in zip(groups,color_values)}

            elif isinstance(palette, dict):
                miss_groups = set(groups) - palette.keys()
                if len(miss_groups) > 0:
                    raise KeyError('Please assign colors to %s groups' %(','.join(groups)))

                color_maps = palette

            self.color = self.color.map(color_maps)
            self.param['legend_color'] = color_maps  

    def sizeMap(self,size_scale=400,bin_labels=None, ascending=True):
        """ Map value to different sizes

        Parameters
        ----------
        size_scale : int, optional
            The scale proportion to size position, by default 400
        bin_labels : dict, optional
            A dict contains bins and lable information to cut values in 
            different group, by default None
            e.g bins=[0,.01,.05,1],labels=['**','*','NS']
        ascending : bool, optional
            Whether larger values have larger marker size, by default True
        """
        if bin_labels:
            self.size = pd.cut(self.size,bins=bin_labels['bin'],labels=bin_labels['label'])
        else:
            # default separate into four group
            qt = [0, .25,.75, 1.]
            labels = ['< %.3f' % x for x in self.size.quantile(qt[1:])]
            self.size = pd.qcut(self.size, qt, labels=labels)

        ordered_categories = self.size.cat.categories.tolist()
        if not ascending: # low value has large marker size
            ordered_categories = ordered_categories[::-1]

        num_categories = float(len(ordered_categories))
        size_map = {
            lb: size_scale * ((i+1)/num_categories) 
            for i,lb in enumerate(ordered_categories)
            }
        self.size = self.size.map(size_map)
        # Avoid oversize marker
        legend_size_map = {
            lb: 10 * ((i+1)/num_categories)
            for i, lb in enumerate(ordered_categories)
        }
        self.param['legend_size'] = legend_size_map
        
    def markerMap(self,marker_values=None):
        """ Map values to different markers

        Parameters
        ----------
        marker_values : list or dict, optional
            A list of markers or a dict with group name as key and marker as 
            value, by default None

        Raises
        ------
        ValueError
            The number of customized marker list/dict is not equal to the 
            number of groups.
        """

        groups = sorted(self.marker.unique().tolist())
        if marker_values is None:
            marker_map = {k: v for k, v in zip(groups,Line2D.filled_markers)}
        elif len(marker_values) < len(groups) :
            raise ValueError('Please assign marker to every group. %d marker found but there are %d groups.'
                             % (len(marker_values), len(marker_values)))
        elif isinstance(marker_values, list):
            marker_map = {k:v for k,v in zip(groups,marker_values)}
        elif isinstance(marker_values, dict):
            marker_map = marker_values

        self.marker = self.marker.map(marker_map)
        self.param['legend_marker'] = marker_map 

    def addLegend(self,ax,cur,cbar_ax=None,c=None,m=None,s=None,**legend_kws):
        """Add legend to current axes

        Parameters
        ----------
        ax : matplotlib.axes 
           Axes to be added legend on 
        cur: matplotlib.collections.PathCollection
            Current axes for cbar_ax to map colors
        cbar_ax: matplotlib.axes
            colorbar axes
        c : str, optional
            default color, by default 'k'
        m : str, optional
            default marker stype, by default 'o'
        s : int, optional
            default marker size, by default 10
        """
        if not c:
            c = self.param['default_color']
        if not m:
            m = self.param['default_marker']
        if not s:
            s = self.param['default_size']

        def unpackLegend(legend_type):
            """ Generate legend handles and labels
            Returns
            -------
            tuple
                ([list of labels],[list of Line2D])
            """
            labels = []
            handles = []
            for k, v in self.param['legend_%s' % legend_type].items():
                labels.append(k)
                if legend_type == 'color':
                    handles.append(
                        Line2D([0], [0], marker=m, ms=s, ls='', color=v))
                elif legend_type == 'size':
                    handles.append(
                        Line2D([0], [0], marker=m, ms=v, ls='', color=c))
                elif legend_type == 'marker':
                    handles.append(
                        Line2D([0], [0], marker=v, ms=s, ls='', color=c))
            return labels, handles

        ## Add colorbar
        if cbar_ax:
            x_loc = 1.30
            cbar_ax.figure.colorbar(cur)
        else:
            x_loc = 1.05 
        y_loc = 1
        flag = False # Determine whether there is legend or not
        for legend_type in ['color','size','marker']:
            if ('legend_%s'% legend_type ) in self.param:
                labels, handles = unpackLegend(legend_type)
                title = self.param['title'][legend_type]
                if flag:
                     leg = Legend(ax, handles=handles,labels=labels,
                                  loc=(x_loc,y_loc),title=title, **legend_kws)
                     ax.add_artist(leg)
                else:
                    ax.legend(handles=handles, labels=labels, 
                              loc=(x_loc, y_loc), title=title, **legend_kws)
                flag = True
                y_loc -= .1*len(labels)


scatter_doc = dict(
    # Shared function parameters
    input_params=dedent("""
    df : pd.DataFrame
        Dataset for plotting.
    x, y : str
        Names of variables in ``df``.\
        """),

    axes_order=dedent("""
    x_order,y_order : list
        A list to customize axes order.\
        """),

    color = dedent("""
    color: str
        names of variables in ``df`` where piont color is mapped to.
    vmax : float, optional
        Max value on the continuous color range mapping, by default None
    vmin : float, optional
        min value on the continuous color range mapping, by default None
    center : float, optional
        center value on the continuous color range mapping, by default None.\
        """),
    palette = dedent("""
    palette : str, list, or dict, optional 
        Paltte to control the color mapping options. Default is 'Set1'.\
        """),

    size = dedent("""
    size: str
        names of variables in ``df`` where piont size is mapped to.
    size_scale : int, optional
            The scale proportion to size position, by default 200
    bin_labels : dict, optional
        A dict contains bins and lable information to cut values in 
        different group, by default None
        e.g bins=[0,.01,.05,1],labels=['**','*','NS']
    size_ascending : bool, optional
        Whether larger values have larger marker size, by default True.\
        """),
    marker = dedent("""
    marker : str
        names of variables in ``df`` where piont shape is mapped to.\
        """),

    ax_in=dedent("""
    ax : matplotlib Axes, optional
        Axes object to draw the plot onto, otherwise uses the current Axes.\
        """),
    cbar_ax=dedent("""
    cbar_ax : matplotlib Axes, optional
        Axes object to draw the colorbar onto, otherwise uses the same Axes as ``ax``.\
        """),
    ax_out=dedent("""
    ax : matplotlib Axes
        Returns the Axes object with the plot drawn onto it.\
        """),
)

class Mtrx(_BaseScatter):

    def __init__(self,df, x, y, color=None, size=None):
        super().__init__(df=df, x=x, y=y, color=color, size=size, marker=None)

    def plot(self, ax=None, x_order=None, y_order=None, size_scale=400, bin_labels=None, size_ascending=True, palette='RdBu_r', vmax=None, vmin=None, center=None, marker='s', cbar_ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()

        xticklabels,yticklabels = self.orderTicks(x_order,y_order)
        self.sizeMap(size_scale, bin_labels, size_ascending)
        self.colorMap(palette,vmax,vmin,center)

        ## Exclude pre-occupied kws
        scatter_kws = {k: v for k, v in kwargs.items() if k not in [
            'x','y','marker','s','c'
        ]}

        ## Plotting
        cur = ax.scatter(
            x = self.x,
            y = self.y,
            marker = marker,
            s = self.size,
            c = self.color,
            cmap = palette,
            norm = self.param['color_norm'],
            **scatter_kws
        )
        ax.set_xticks(np.arange(len(xticklabels)))
        ax.set_yticks(np.arange(len(yticklabels)))
        ax.set_xticklabels(xticklabels,rotation=90)
        ax.set_yticklabels(yticklabels)

        ## Add legend
        cbar_ax = cbar_ax if cbar_ax else ax
        self.addLegend(ax,c='k', m=marker,cbar_ax=cbar_ax,cur=cur,frameon=False)

        ## Theme Tuning
        # Add grid to separate cells
        ax.grid(False, 'major')
        ax.grid(True, 'minor')
        # Padding axes
        ax.set_xlim([-0.5, self.x.max() + 0.5])
        ax.set_ylim([-0.5, self.y.max() + 0.5])
        # Move ticklabels to the center
        ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
        ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)
        plt.tight_layout()

        return ax

class Anno(_BaseScatter):

    def __init__(self, df, x, y, color=None, size=None,marker=None):
        super().__init__(df=df, x=x, y=y, color=color, size=size, marker=marker)

    def draw(self, ax, palette, cur_marker=None,subset=None, **scatter_kws):
        """ Draw plot for subsets """
        if subset is None :
            subset = [True] * len(self.x)

        if isinstance(self.size, pd.Series):
            cur_size = self.size[subset]
        else:
            cur_size = self.param['default_size']

        if isinstance(self.color, pd.Series):
            cur_color = self.color[subset]
        else:
            cur_color = self.param['default_color']
    
        if 'color_norm' in self.param:
            norm = self.param['color_norm']
            cmap = palette
        else:
            norm, cmap = None, None

        if cur_marker:
            marker = cur_marker
        else:
            marker = self.param['default_marker']

        ## Plotting
        cur = ax.scatter(
            x=self.x[subset],
            y=self.y[subset],
            marker=cur_marker,
            s=cur_size,
            c=cur_color,
            cmap=cmap,
            norm=norm,
            **scatter_kws
        )
        return cur

    def plot(self, ax=None,size_scale=400, bin_labels=None, size_ascending=True,
            palette='RdBu_r', vmax=None, vmin=None, center=None, cbar_ax=None,
            marker_values=None, **kwargs):

        if ax is None:
            ax = plt.gca()    

        ## Exclude pre-occupied kws
        scatter_kws = {k: v for k, v in kwargs.items() if k not in [
            'x', 'y', 'm', 's', 'c','norm','cmap'
        ]}
        
        ## Map values to corresponding size, color and marker
        if self.color is not None:
            self.colorMap(palette, vmax, vmin, center)
        elif 'c' in kwargs:
            self.param['default_color'] = kwargs['c']

        if self.size is not None:
            self.sizeMap(size_scale, bin_labels, size_ascending)
        elif 's' in kwargs:
            self.param['default_size'] = kwargs['s']

        if self.marker is not None:
            self.markerMap(marker_values)
            for cur_marker in self.param['param']['legend_marker']:
                subset = [x for x in self.marker if x == cur_marker]
                cur = self.draw(ax=ax,cur_marker=cur_marker,palette=palette,subset=subset,**scatter_kws)
        elif 'm' in kwargs:
            self.param['default_marker'] = kwargs['m']

        cur = self.draw(ax=ax,palette=palette, **scatter_kws)
        ## Add legend
        if 'color_norm' in self.param:
            cbar_ax = cbar_ax if cbar_ax else ax
        else:
            cbar_ax = None
        self.addLegend(ax, c='k',cur=cur,cbar_ax=cbar_ax)
        plt.tight_layout()

        return ax
        

def heatmap(df, x, y, color,size,ax=None,x_order=None,y_order=None,
            size_scale=200, bin_labels=None, size_ascending=True,
            palette='RdBu_r',vmax=None,vmin=None,center=None,marker='s',cbar_ax=None,**kws):
    """ 

    Parameters
    ----------
    {input_params}
    {axes_order}
    {color}
    {size}
    {ax_in}
    {cbar_ax}

    marker : str, optional
        The point shape of cells, by default 's'
    palette: str, optional
        Name of palette. by default 'Set1'.Reference: http://matplotlib.org/examples/color/colormaps_reference.html
    
    Returns
    -------
    {ax_out}
    """
    plotter = Mtrx(df, x, y,color,size)
    ax = plotter.plot(
        ax=ax,
        x_order=x_order,y_order=y_order,
        size_scale=size_scale, bin_labels=bin_labels, size_ascending=size_ascending,
        palette=palette,vmax=vmax,vmin=vmin,center=center,
        marker=marker,cbar_ax=cbar_ax,**kws)
    return ax

heatmap.__doc__ = heatmap.__doc__.format(**scatter_doc) + \
    dedent("""\
    
    Examples
    --------

    Visualize correlation heatmap with different marker sizes:

    .. plot::
        :context: close-figs

        >>> import Bioplots as bpt
        >>> df = bpt.get_rdataset('lung')
        >>> df_cor = df.corr().reset_index().melt(id_vars=['index'])
        >>> df_cor['size'] = df_cor['value'].abs()
        >>> ax = bpt.heatmap(df=df_cor,x='index',y='variable',color='value',size='size');

    Customize the axes order by the averge

    .. plot::
        :context: close-figs

        >>> orders = df_cor.groupby('index')['value'].mean().sort_values().index.tolist()
        >>> ax = bpt.heatmap(df=df_cor,x='index',y='variable',color='value',size='size',
        ...                  x_order=orders,y_order=orders);

    Change the size label

    .. plot::
        :context: close-figs

        >>> ax = bpt.heatmap(df=df_cor,x='index',y='variable',color='value',size='size',
        ...                  x_order=orders,y_order=orders,
        ...                  bin_labels=dict(bin=[0,.5,.8,1],label=['<.5','<.8','<1']));

        """)

def volcano(df, lfc, pvalue, color_list=('orange', 'lightgray', 'skyblue'), 
            visible_hits=10, label=None, lfc_cutoff=1, pvalue_cutoff=.05,
            text_adjust=dict(arrowprops=dict(arrowstyle='->', color='k'))
            ):
    """ It plots significance versus fold-change on the y and x axes, respectively. 

    Parameters
    ----------
    df : pd.DataFrame 
    Dataset for plotting.
    
    lfc : str
        Names of variables in ``df``, which contians log fold change values
    pvalue : str
        Names of variables in ``df``, which contians p values
    color_list : tuple, optional
        A list of color to be assigned to significant positive hits, non-significant 
        hits, and significant negtive hits, respectively.by default ('orange', 'lightgray', 'skyblue')
    visible_hits : int or str, optional
        Number of hits to show the label
        or a list of customized label to show, by default 10
    label : str
        Names of variables in ``df``, which contians labels for points. by default None
    lfc_cutoff : float, optional
        Cutoff of log fold changes, by default 1.0
        significant points with ``lfc`` larger than   ``lfc_cutoff`` are the positive hits.
        significant points with ``lfc`` lower than -``lfc_cutoff`` are the negative hits.
    pvalue_cutoff : float, optional
        Cutoff of p values, by default .05
        Points with ``pvalue`` lower than the cutoff are considered significant.
    text_adjust : dict, optional
        Annotation text property, by default dict(arrowprops=dict(arrowstyle='->', color='k'))
        Reference: https://adjusttext.readthedocs.io/en/latest/

    Returns
    -------
    {ax_out}
    """
    df = df.copy().dropna(subset=[lfc,pvalue])
    signal_col = lfc+'_signal'
    # Group samples by LFC and p value
    significant = df[pvalue] < pvalue_cutoff
    neg_sig = (significant) & (df[lfc] < -lfc_cutoff)
    pos_sig = (significant) & (df[lfc] > lfc_cutoff)
    # Log transform p value 
    df[pvalue] = -np.log10(df[pvalue])
    # Assign groups to the signal column
    df[signal_col] = 0
    df.loc[neg_sig, signal_col] = -1
    df.loc[pos_sig, signal_col] = 1
    palette = {k:v for k,v in zip([1,0,-1],color_list)}
    plotter = Anno(df=df,x=lfc,y=pvalue,color=signal_col)
    ax = plotter.plot(palette=palette)

    ## Show Text For Significant Points]
    if label is not None:
        if isinstance(visible_hits,list):
            ## Show customized hit from the list of input hits
            visible_pos = df.loc[df[label].isin(visible_hits),:]
        else:
            sig_df = df.loc[df[signal_col] !=0,:]
            ranks = sig_df[lfc].abs().rank(method='min', ascending=False)
            visible_pos = sig_df.loc[ranks<visible_hits,:]
       
        texts = [
            ax.text(row[lfc],row[pvalue], row[label],c='k')
            for _,row in visible_pos.iterrows()
        ]
        adjustText.adjust_text(texts, **text_adjust)

    ax.set(xlabel='Log Fold Change',ylabel='-log10(P-value)')
    return ax


volcano.__doc__ = volcano.__doc__.format(**scatter_doc) + \
    dedent("""\
    
    Examples
    --------

    Visualize correlation with significance:

    .. plot::
        :context: close-figs

        >>> import Bioplots as bpt
        >>> from scipy.stats import pearsonr
        >>> df = bpt.get_rdataset('lung')
        >>> def pearson_pvalue(x,y):
        ...     return pearsonr(x,y)[1]
        >>> df_cor = df.corr().reset_index().melt(id_vars=['index'])
        >>> df_pvalue = df.corr(method=pearson_pvalue).reset_index().melt(id_vars=['index'])
        >>> cor_pvalue = df_cor.merge(df_pvalue,on=['index','variable'])
        >>> cor_pvalue.rename(columns={'value_x':'cor','value_y':'pvalue'},inplace=True)
        >>> cor_pvalue.head()
        >>> bpt.volcano(df=cor_pvalue,lfc='cor',pvalue='pvalue',lfc_cutoff=0.1,pvalue_cutoff=.05,label='index');

        """)


def scatterplot(df,x,y,color=None,marker=None,size=None,
                size_scale=10, bin_labels=None, size_ascending=True,
                palette='Set1', vmax=None, vmin=None, center=None, cbar_ax=None,
                marker_values=None,
                visible_hits=None,label=None,text_adjust=dict(arrowprops=dict(arrowstyle='->', color='k'))
            ):
    """ It plots scatters with annotation by different color and marker shape (optional). 

    Parameters
    ----------
    {input_params}
    {color}
    {size}
    {marker}
    {palette}
    
    visible_hits : int or str, optional
        Number of hits to show the label
        or a list of customized label to show, by default 10
    label : str
        Names of variables in ``df``, which contians labels for points. by default None
    text_adjust : dict, optional
        Annotation text property, by default dict(arrowprops=dict(arrowstyle='->', color='k'))
        Reference: https://adjusttext.readthedocs.io/en/latest/

    Returns
    -------
    {ax_out}
    """
    use_columns = [x,y]
    if color:
        use_columns.append(color)
    if size:
        use_columns.append(size)
    if marker:
        use_columns.append(marker)

    df = df.copy().dropna(subset=use_columns)
    plotter = Anno(df=df, x=x, y=y, color=color,size=size,marker=marker)
    ax = plotter.plot(size_scale=size_scale, bin_labels=bin_labels, size_ascending=size_ascending,
                      palette=palette, vmax=vmax, vmin=vmin, center=center, cbar_ax=cbar_ax,
                      marker_values=marker_values)

    ## Show customized hit from the list of input hits
    if (label is not None) and isinstance(visible_hits, list):
        visible_pos = df.loc[df[label].isin(visible_hits), :]
        texts = [
            ax.text(row[x], row[y], row[label], c='k')
            for _, row in visible_pos.iterrows()
        ]
        adjustText.adjust_text(texts, **text_adjust)
    return ax


scatterplot.__doc__ = scatterplot.__doc__.format(**scatter_doc) + \
    dedent("""\
    
    Examples
    --------

    Visualize 2D scatter plot with mutiple annotations:

    .. plot::
        :context: close-figs

        >>> import Bioplots as bpt
        >>> df = bpt.get_rdataset('lung')
        >>> bpt.scatterplot(df=df.head(20),x='wt.loss',y='age',color='inst',palette='RdBu_r',
        ...        size='time',label='status',visible_hits=[2],size_scale=400)
       
        """)