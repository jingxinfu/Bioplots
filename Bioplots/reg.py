#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__doc__ = """
    Function to evaluate regression
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

def plotAUC(df,y_pred,y_true,platter='Set1',pos_label='Responder',title='',factor=None,ax=None):
    # Establish variables
    if ax is None:
        ax = plt.gca()
    plot_dt = {}
    if isinstance(platter, str):
        ax.set_prop_cycle(color=plt.get_cmap(platter).colors)
    elif isinstance(platter, list):
        ax.set_prop_cycle(color=platter)
    else:
        raise ValueError('Only string and list is available for platter.')

    # group by factor
    if factor is None:
        grouped = {y_pred:df}.items()
    else:
        grouped = df.groupby(factor)

    # AUC calculation
    for gp_name,sub_dt in grouped:
        if sub_dt.shape[0] < 3:
            continue
        fpr, tpr, _ = metrics.roc_curve(
                y_true=np.array(sub_dt[y_true]),
                y_score=np.array(sub_dt[y_pred]),
                pos_label=pos_label
                )
        z = metrics.auc(fpr,tpr)
        pos_num = sum(sub_dt[y_true] == pos_label)
        neg_num = sub_dt.shape[0] - pos_num
        plot_dt[gp_name]={
                'fpr':fpr,
                'tpr':tpr,
                'z':z,
                'size':'(Pos:%d,Neg:%d)'% (pos_num,neg_num),
                }

    for k,v in plot_dt.items():
        ax.plot(v['fpr'], v['tpr'],label='%s\nSample Size = %s,AUC = %0.2f' % (k,v['size'],v['z']),lw=2)
        ax.legend(loc=(1.01,.5),labelspacing=1)
    ax.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    return ax

