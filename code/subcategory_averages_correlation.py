"""
reads in the anonymized_obfuscated_data.csv,
creates the subcategory matrix (columns average over subcategories.)

Code to:
- make a plot of subcategory ave data's correlation (w/ nan's before imputation)
- imputaton by (1) sampling from the DT-neighbor-informed KDEs and (2) sampling uniform[0,1]
- plot subcategory ave data's correlation after imputing
- makes the the convergence plot for the average correlation matrix (step change per imputation)
- makes the correlation confidence intervals for a random selection of variables.

calls imputation_correlation.py and predict_missing_values.py functions
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import copy, time
from random import random
sys.path.append(str(Path(".", "code")))
from predict_missing_values import get_dt_imputed_values
from imputation_correlation import *


def make_subcat(df):
    """
    Input: df = pandas dataframe, anonymized data
    Output: dfsc = pandas dataframe, subcategory averages from anonymized data
    """
    dfsc = df.copy()
    new_cols = []

    pre = "likert"
    cols = [c for c in data_cols if c.startswith(pre)]
    subcats = set(sorted(map(lambda s: s.split("-")[1], cols)))
    for cat in subcats:
        s = f"{pre}-{cat}-ave"
        new_cols.append(s)
        dfsc[s] = dfsc.apply(lambda row: row[[c for c in cols if c.startswith(f"{pre}-{cat}")]].mean(), axis = 1)

    pre = "sentiment"
    sentcats = { "recurring-tasks" : [1, 10, 15],
          "collab" : [5,6,7],
          "apt": [8,9,11],
          "info-quality": [12, 13, 14],
          "other": [0,2,3,4,16,17,18,19]
          }
    for cat, vals in sentcats.items():
        dfsc[f"{pre}-{cat}-ave"] = dfsc.apply(lambda row: row[[f"{pre}-{c}" for c in vals]].mean() , axis = 1)
        new_cols.append( f"{pre}-{cat}-ave")

    pre = "ticket"
    categories = ["nids", "malware", "iprep"]
    for cat in categories:
        cols = [c for c in dfsc.columns if c.lower().startswith(f"{pre}-{cat}")]
        dfsc[f"{pre}-{cat}-ave"] = dfsc.apply(lambda row: row[cols].mean(), axis = 1)
        new_cols.append(f"{pre}-{cat}-ave")

    pre = "swaps"
    categories = ['malware_analysis', 'malware', 'xsoc', 'iprep', 'nids']
    for cat in categories:
        cols = [c for c in dfsc.columns if c.lower().startswith(f"{pre}-{cat}")]
        dfsc[f"{pre}-{cat}-ave"] = dfsc.apply(lambda row: row[cols].mean(), axis = 1)
        new_cols.append(f"{pre}-{cat}-ave")

    pre = "time"
    categories = ['malware_analysis', 'malware', 'xsoc', 'iprep', 'nids']
    for cat in categories:
        cols = [c for c in dfsc.columns if c.lower().startswith(f"{pre}-{cat}")]
        dfsc[f"{pre}-{cat}-ave"] = dfsc.apply(lambda row: row[cols].mean(), axis = 1)
        new_cols.append(f"{pre}-{cat}-ave")


    dfsc = dfsc[['tool_label', 'familiarity', 'role', 'socs', 'years'] + new_cols]
    return dfsc


def pretty_clustermap(df, outputpath = False, name = 'imputation-before-clustermap-', title = "Clustermap Before Imputation" ):
    """
    df = dataframe to be clustermapped
    outputpath = folder path for saving plot or False = no saving the plot
    name = str for file saving
    title = str for plot title.
    """
    if outputpath: assert name
    cg = sns.clustermap(df, vmin=-1, vmax = 1, cmap=cmap, figsize=(12,12))
    cg.ax_col_dendrogram.set_visible(False)
    plt.setp(cg.ax_heatmap.get_xticklabels(), rotation = 45, fontsize = 14,  horizontalalignment = "right", rotation_mode ="anchor")
    plt.setp(cg.ax_heatmap.get_yticklabels(), fontsize = 14)
    plt.title(title, fontsize= 16)
    if outputpath:
        plt.savefig(Path(outputpath,f'{name}.svg'), format = "svg")
        plt.savefig(Path(outputpath,f'{name}.png'))
    plt.show()
    print("clustermap made!")
    return


if __name__ == '__main__':
    # paths:
    anondata = Path('anonymized_obfuscated_data.csv')
    anondata.exists()
    outputpath = False ## plots not saved

    # outputpath = Path("plots") ## plots saved in ../plots/
    # if not outputpath.exists(): outputpath.mkdir()

    df0 = pd.read_csv(anondata, index_col = 0)

    id_cols = ['familiarity', 'role', 'socs', 'years']
    data_cols = df0.columns.difference(["tool_label"] +id_cols )

    ## set datatypes correctly:
    df0[id_cols + ["tool_label"]] = df0[id_cols + ["tool_label"]].astype("str")

    dfsc = make_subcat(df0)

    df = dfsc[dfsc.columns.difference(["tool_label"] + id_cols)]

    ## plot dfsc correlation before imputation:
    pretty_clustermap(df.corr(), outputpath=outputpath, name = 'imputation-before-clustermap-', title = "Subcategory Averages' Correlations\n No Imputation (with Missing Values)" )
    # pretty_clustermap(df.corr(), outputpath = False)


    # Make subcategory average correlation plots with the decision tree imputed values:
    ## get m values for every missing cell using DT clustering:
    m = 1000
    imputed_values = get_dt_imputed_values(dfsc, m = m)

    ## run imputation with DT neighbors--> KDEs--> and save plots:
    title = "Subcategory Averages' Correlations\nImputed with Decision Tree Neighbors"
    c, mid, interval = mi_corr(dfsc, imputed_values =imputed_values, outputpath = outputpath, name = title, cmap=cmap )

    ##----for imputed values sampling at random, use this: ----##
    # c, mid, interval = mi_corr(dfsc, imputed_values =None, outputpath = outputpath, name = title, cmap=cmap ) #### imputed_values  = None means it will draw samples uniformly at random in [0,1] to fill missing data.
    ##----  //  ----##

    # plot dfsc correlation after DT-neighbors KDE imputation
    pretty_clustermap(c, outputpath=outputpath, name = "imputation-DT-clustermap-", title = title)


    ## now do confidence intervals plots:
    # randomly sample one of each type of variable:
    variables = []
    for s in ['likert', 'sentiment', 'ticket', 'time', 'swap']:
        clusters=[x for x in mid.columns if x.startswith(s)]
        np.random.shuffle(clusters)
        variables += clusters[:2]
        # variables.append(clusters[0])
    variables

    name = False
    ## make conf interval plots:
    conf_interval_plot(mid, interval, variables = variables, outputpath=outputpath, name=name)
