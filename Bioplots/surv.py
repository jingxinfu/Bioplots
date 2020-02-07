# License           : MIT
# Author            : Jingxin Fu <jingxinfu.tj@gmail.com>
# Date              : 25/10/2019
# Last Modified Date: 27/01/2020
# Last Modified By  : Jingxin Fu <jingxinfu.tj@gmail.com>
# -*- coding:utf-8 -*-
import lifelines
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
import numpy as np
import pandas as pd
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import matplotlib as mpl
from Tigger.visual.utils import fancy_scientific
mpl.use('Agg')
mpl.rcParams.update({
    "font.size": 15,
    "legend.frameon": False,
})

##################------------------------Kaplan Meier Plot -----------##################


def kmPlot(dt, duration_col, event_col, name, stat, title, ax=None, group={1: ('Top', 'red', '-'), 0: ('Bottom', 'blue', '--')}):
    kmf = KaplanMeierFitter()
    if ax is None:
        ax = plt.gca()
    ax.set_title(title)
    for gp in sorted(group.keys(), reverse=True):
        sub_dt = dt.loc[dt[name] == gp, :]
        T = sub_dt[duration_col]
        E = sub_dt[event_col]
        label = '%s (n=%d)' % (group[gp][0], sub_dt.shape[0])
        kmf.fit(T, E, label=label)
        kmf.plot(ax=ax, show_censors=True, ci_show=False,
                 c=group[gp][1], linestyle=group[gp][2],)
    # Same stype as Peng draw before
    leg = ax.get_legend()
    for hanldes, text in zip(leg.legendHandles, leg.get_texts()):
        hanldes.set_visible(False)
        text.set_color(hanldes.get_color()[0])
    max_duration = sub_dt[duration_col].max()
    if max_duration < 15:
        duration_unit = 'year'
    elif max_duration < 180:
        duration_unit = 'month'
    else:
        duration_unit = 'day'
    statistic_info = '$ Z =' + \
        fancy_scientific(stat[0]) + ', p=' + \
        fancy_scientific(stat[1]) + ' $'
    ax.set_title('\n'.join([title, statistic_info]))
    ax.set_xlabel('%s (%s)' % (duration_col, duration_unit))
    ax.set_ylabel("Survival Fraction")


def _rm_singluarity(dt, event_col):
    """ Remove columns while triger singularity problem during the coxph regression

    Parameters
    ----------
    dt : pd.DataFrame
        patients survival info and correpsonidng investigate columns
    event_col : str
        Binary column name indicate patients status

    Returns
    -------
    pd.DataFrame
        data frame has already been removed problem columns
    """
    # dt = dt[dt[event_col].str.isnumeric(),:]
    events = dt[event_col].map(int).astype(bool)
    problem_col = (
        (dt.loc[events, ].var(axis=0) <= 0) |
        (dt.loc[~events, ].var(axis=0) <= 0)
    )
    return dt.loc[:, (~problem_col) | (dt.columns == event_col)].dropna(axis=0)


def evaluate(dt, duration_col, event_col, name, title='', quantile=.5, interact_name=None, output=None):
    """ Function to evalute the prognostic effect of the <name> column
    1. Overall effect on patients survival
    2. Interaction effect with <interact_name> term on patients survival

    Parameters
    ----------
    dt : pd.DataFrame
        patients survival information and corresponding analysis term
    duration_col : str
        Survival length column name
    event_col : str
        Binary valule column indicates patients status
    name : str
        variable term want to evaluate
    ax : axis, optional
        figure axis, by default None
    quantile : float, optional
        stratify threashold by quantile, by default .5
        high > quantile
        low < 1 - quantile
    interact_name : str, optional
        confonding variables, by default None
    output : str, optional
       path to save figures, by default []
    """
    results = pd.DataFrame(columns=['z-score', 'Pvalue', 'Term'])
    errs = ''

    try:
        dt = _rm_singluarity(dt, event_col=event_col)
    except ValueError:
        return results, '[ERROR] Remove Signuarity step for %s' % title

    if not duration_col in dt.columns:
        return results, '[ERROR] duration col is consistent %s' % title

    if not name in dt.columns:
        return results, '[ERROR] %s col is consistent in %s' % (name, title)

    default_size = 6
    if (interact_name is None) or (not interact_name in dt.columns):
        basic = dt.copy()
        if not output is None:
             fig, ax1 = plt.subplots(1, 1, figsize=(default_size, default_size))
        do_interact = False
    else:
        basic = dt.drop(interact_name, axis=1)
        interacts = dt.copy()
        interacts['interact'] = interacts[name] * interacts[interact_name]
        if not output is None:
            fig = plt.figure(figsize=(default_size*2, default_size*2))
            gs = fig.add_gridspec(4, 4)
            ax1 = fig.add_subplot(gs[:2, 1:3])
            sub_axs = {
                'High %s' % name: fig.add_subplot(gs[2:, :2]),
                'Low %s' % name: fig.add_subplot(gs[2:, 2:])
            }

        do_interact = True

    var_high = (dt[name] > dt[name].quantile(quantile))
    var_low = (dt[name] <= dt[name].quantile(1-quantile))
    cph = CoxPHFitter()

    output_figs = 0
    # 1. basical evaluation
    stat = [np.nan, np.nan]
    basic_error = True
    try:
        cph.fit(basic, duration_col=duration_col, event_col=event_col)
        z,p = cph.summary.loc[name, 'z'],cph.summary.loc[name, 'p'],
        results = results.append({
            'z-score':z,
            'Pvalue':p,
            'Term': 'Basic'
        }, ignore_index=True)
        stat = [z,p]
        basic_error = False

    except ValueError as err:
        errs += '%s Basic step ValueError ;' % title
    except ZeroDivisionError as err:
        errs += '%s Basic step ZeroDivisionError;' % title
    except lifelines.utils.ConvergenceError as err:
        errs += '%s Basic step ConvergenceError;' % title
    except TypeError as err:
        errs += '%s Basic step  TypeError' % title
    # Binary Kaplan-Meier plot
    if (not output is None) and (not basic_error):
        basic[name] = np.nan
        basic.loc[var_low, name] = 0
        basic.loc[var_high, name] = 1
        group = {
            1: ('%s High' % name, 'red', '-'),
            0: ('%s Low' % name, 'blue', '--'),
        }
        basic_title = 'All' if do_interact else ''
        kmPlot(dt=basic.dropna(axis=0), duration_col=duration_col, event_col=event_col, group=group,
               name=name, stat=stat, title=basic_title, ax=ax1)
        output_figs += 1

    if not do_interact:
        if not output is None:
            if output_figs > 0:
                plt.suptitle(title)
                plt.savefig("%s.png" % output, dpi=300,bbox_inches='tight')
            #fig.close('all')
            plt.close(fig)

        return results, errs

    # 2. interaction evaluation
    # Pure calculation for continuous interaction test score
    try:
        cph.fit(interacts, duration_col=duration_col, event_col=event_col)
        results = results.append({
            'z-score': cph.summary.loc['interact', 'z'],
            'Pvalue': cph.summary.loc['interact', 'p'],
            'Term': 'Interact With CTL'
        }, ignore_index=True)
    except ValueError as err:
        errs += '%s Interact continuous step ValueError;' % title
    except ZeroDivisionError as err:
        errs += '%s Interact continuous step ZeroDivisionError;' % title
    except lifelines.utils.ConvergenceError as err:
        errs += '%s Interact continuous step  ConvergenceError' % title
    except TypeError as err:
        errs += '%s Interact continuous step  TypeError' % title

    # Binary Kaplan-Meier plot
    stra_interact = interacts.drop([name, 'interact'], axis=1)

    for sub_title, sub_gp in {
        'High %s' % name: var_high,
        'Low %s' % name: var_low
    }.items():
        sub_dt = stra_interact.loc[sub_gp, :].copy()
        sub_high = (sub_dt[interact_name] >
                    sub_dt[interact_name].quantile(quantile))
        sub_low = (sub_dt[interact_name] <=
                   sub_dt[interact_name].quantile(1 - quantile))

        sub_dt[interact_name] = np.nan
        sub_dt.loc[sub_low, interact_name] = 0
        sub_dt.loc[sub_high, interact_name] = 1
        sub_dt = sub_dt.dropna(axis=0)
        group = {
            1: ('%s High' % interact_name, 'red', '-'),
            0: ('%s Low' % interact_name, 'blue', '--'),
        }
        stat = [np.nan, np.nan]
        interact_error = True
        try:
            cph.fit(sub_dt, duration_col=duration_col, event_col=event_col)
            z,p =cph.summary.loc[interact_name, 'z'],cph.summary.loc[interact_name, 'p']
            results = results.append({
                'z-score': z,
                'Pvalue': p,
                'Term': 'Subgroup %s' % sub_title
            }, ignore_index=True)

            stat = [z,p]
            interact_error = False

        except ValueError as err:
            errs += '%s Interact binary (%s) step ValueError;' % (title,
                                                                  sub_title)
        except ZeroDivisionError as err:
            errs += '%s Interact binary (%s) step ZeroDivisionError;' % (
                title, sub_title)
        except lifelines.utils.ConvergenceError as err:
            errs += '%s Interact binary (%s) step  ConvergenceError' % (
                title, sub_title)
        except TypeError as err:
            errs += '%s Interact binary (%s ) step  TypeError' % (title,
                                                                  sub_title)

        if (not output is None) and (not interact_error):
            kmPlot(dt=sub_dt, duration_col=duration_col, event_col=event_col, group=group,
                   name=interact_name, stat=stat, title=sub_title, ax=sub_axs[sub_title])
            output_figs += 1

    if not output is None:
        if output_figs > 0:
            plt.suptitle(title)
            plt.subplots_adjust(hspace=1,wspace=0.7)
            plt.savefig("%s.png" % output, dpi=300)
        plt.close(fig)

    if len(errs) > 0:
        errs = '[ERROR] %s' % errs
    return results, errs
