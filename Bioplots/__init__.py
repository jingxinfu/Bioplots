#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# License           : MIT
# Author            : Jingxin Fu <jingxinfu.tj@gmail.com>
# Date              : 24/01/2020
# Last Modified Date: 27/01/2020
# Last Modified By  : Jingxin Fu <jingxinfu.tj@gmail.com>
# -*- coding:utf-8 -*-

import os
import pandas as pd
import pkg_resources

__version__ = '0.1.0'

DATA_DIR = pkg_resources.resource_filename('Bioplots', 'data/')
DATASET = pd.read_csv(os.path.join(
    DATA_DIR, 'datasets.csv')).set_index('Item')['CSV'].to_dict()

from .utils import *
from .distribution import *

themes = dict(
        paper={
        "font.weight": "normal",
        "font.size": 17,
        "axes.titleweight": "normal",
        "axes.labelweight": "normal",
        "figure.titleweight": "normal",
        "axes.grid": False,
        "axes.axisbelow": True,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 2
    }

)

COLORS = [
    'black', 'gray', 'silver', 'rosybrown', 'firebrick', 'red', 'darksalmon', 'sienna', 'sandybrown', 'bisque',
    'tan', 'gold', 'darkkhaki', 'olivedrab', 'chartreuse', 'darkgreen', 'seagreen', 'mediumspringgreen', 'lightseagreen', 'paleturquoise',
    'darkcyan', 'deepskyblue', 'slategray', 'royalblue', 'navy', 'mediumpurple', 'darkorchid', 'plum', 'm', 'palevioletred',
    'indianred', 'khaki', 'olive'
]
