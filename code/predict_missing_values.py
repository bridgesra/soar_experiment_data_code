"""
predict_missing_values.py

- has one function to be called by others:
    get_dt_imputed_values(df, m = 1000):
            For each missing value in df, this trains a decision tree on the subset of df with present values.
            It identifies those training data points that lie in the same leaf node of the unknown data point
            and creates a KDE (distribution) from their values.
            It samples this distribution m times and stores in output dict.
            makes KDE plots.
        Output: dictionary {colunm: {index: [m values]}}


"""

import sys, time
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KernelDensity

pd.options.mode.chained_assignment = None ## diable stupid pandas copy warnings


def _predict_values(obv, m = 1000, verbose = False):
    """
    Input:
        obv = observations, pd.DataFrame, shape = (n, ),
        m = number of samples to return
        verbose = bool, default False. if True, shows a little plot of the KDE.
    fits KDE w/ tophat kernal to obv data, uses bandwidth = (min distance from obv to {0,1}) /2
    Output:
        m samples from kde fit to obv.
    """
    obv[obv<.05] = .05
    obv[obv>.95] = .95
    b = min(.1, (1-obv).min(), obv.min())/2
    data = obv.values[:,np.newaxis]

    kde = KernelDensity(kernel="tophat", bandwidth=b).fit(data)
    if verbose:
        xplot = np.linspace(-.01,1.1,100)[:,np.newaxis]
        yplot = np.exp(kde.score_samples(xplot))
        plt.plot(xplot, yplot)
        plt.show()
    return kde.sample(m)


def get_dt_imputed_values(df, m = 1000, fillna_zero = False):
    """
    Input
        df = dataframe with: columns that are all numeric (STRIP OUT "user", "tool" cols. )
                columns ["tool", "user"] stripped off
                each remaining col normalized to [0,1]
                swap and time cols inverted (so higher is better for all cols. )
                joined with user and config data is ok so long as user wants to use those for prediction.
        m = int;, default 1000, number of simulations/imputations to run
        fillna_zero = bool, default False. If False, when it sets up the DT (to find peers) it will throw out rows with nans (DT requires no nans)
                    If True it will fill nans w/ 0 for the DT step.
    Description
        For each missing value in df, this trains a decision tree on the subset of df with present values.
        It identifies those training data points that lie in the same leaf node of the unknown data point
        and creates a KDE (distribution) from their values.
        It samples this distribution m times and stores in output dict.
    Output: dictionary {colunm: {index: [m values]}}
    """
    print("Starting get_dt_imputed_values()...")
    imputed_values = {}

    starttime = time.time()
    for target in df.columns.difference(["user", "tool"]): # targets
        print(f"\tstarted target {target}")
        imputed_values[target] = {}
        # target = 'likert-collaboration-ave'
        # dft = df[~df[target].isna()] ## can only fit a model based on those rows where we have y's
        indices = df.index[df[target].isna()] ## which indices are missing? we need to sample these.
        for i in indices:
            # i = indices[0]

            #get features = columns where row i has values.
            s = ~df.iloc[i].isna()
            features = list(s.index[s]) # features = all cols where row i has values

            ## set up ml:
            if not fillna_zero: Xy = df[features + [target]].dropna(axis = 0) #keeps only features and target cols, keeps only rows that have no missing values in those cols
            else: #fillna_zero evaluates to True
                Xy = df[features + [target]].fillna(0) #keeps only features and target cols, zeros replace the missing values in those cols
            X = Xy[features]
            y = Xy[target]

            ## get and train models:
            dt = DecisionTreeRegressor(min_samples_leaf=4)
            dt.fit(X, y)

            x = pd.DataFrame(df.iloc[i][features]).transpose() ## our vector of interest, with the missing value
            # dt.apply(x) ## gives the leaf number for our guy of interst
            # dt.apply(X) ## gives the leaf number for every training member
            # peers = Xy[dt.apply(X)==dt.apply(x)].index ## indices of rows that are in the same leaf as the guy we're trying to predict.
            peer_values = Xy[dt.apply(X)==dt.apply(x)][target] ## values of those that cluster with our guy of interest
            imputed_values[target][i] = _predict_values(peer_values, m = m).tolist()
    endtime=time.time()
    print(f"it took {endtime-starttime}s")
    return imputed_values
