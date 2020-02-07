#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

class _BasePlotter(object):
    __slots__ = [
        'df', 'x', 'y', 'order', 'color_palette', 'widths',
        'vals', 'groups', 'grouped_vals', 'color', 'vals_name','sub_group'
    ]

    def __init__(self, df, x, y,sub_group=None,widths=.5,order='median', color_palette='Set1'):

        self.color_palette = color_palette
        self.df = df
        self.order = order
        self.check_variables(x, y)
        self.widths = widths
        self.sub_group = sub_group


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
            self.order = exist_order_list + [x for x in exist_group if not x in self.order]

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

    def autolabel(rects,offset_point=3,str_format='%.1f',horizontal=False):
        """Attach a text label above each bar in *rects*, displaying its height."""""
        for rect in rects:
            height = rect.get_height()
            width = rect.get_width()
            if horizontal:
                xy = (width, rect.get_y() + height/2 )
                xytext = (offset_point,0)
                value = width
                ha='center'
                va='bottom'
            else:
                height = rect.get_height()
                xy = (rect.get_x() + rect.get_width() / 2, height)
                xytext = (0,offset_point)
                value = height
                ha='left'
                va='center'

            ax.annotate(str_format % value,
                    xy=xy,
                    xytext=xytext,
                    textcoords="offset points",
                    ha=ha, va=va)

class BarPlotter(_BasePlotter):
    def __init__(self, df, x, y,sub_group=None,platter='Dark2', ax=None, order='median'):
        super().__init__(df, x, y,sub_group=sub_group,color_palette=platter, order=order)

    def plot(self,width=.35,ax=None):
        if ax is None:
            ax = plt.gca()

        boxDat = [np.asarray(self.grouped_vals.get_group(g)[self.vals_name].dropna())
                  for g in self.order]

        if self.orient == 'v':
            ax.bar()





