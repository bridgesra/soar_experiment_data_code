"""
imputation_correlation.py

This script has two functions to be imported by other scripts:

1. mi_corr() does the multiple imputation for computing correlation
    - plots the averaged correlation acorss imputation of missing values
    - outputs  the interval dataframe for 95% interval: mid +/ interval

2. conf_interval_plot() plots 95% confidence intervals.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import copy, time
from random import random

cmap = sns.diverging_palette(240, 20, as_cmap=True)
type(cmap)

pd.options.mode.chained_assignment = None ## diable stupid pandas copy warnings

def mi_corr(df, m = 1000, verbose = True, imputed_values = None, outputpath = None, name = None, cmap = cmap):
    """
    Input:
        df = dataframe with: columns that are all numeric (STRIP OUT "user", "tool" cols. )
                columns ["tool", "user"] stripped off
                each remaining col normalized to [0,1]
                swap and time cols inverted (so higher is better for all cols. )
        m = int;, default 1000, number of simulations/imputations to run
        verbose = True means it will display plots/False will not display plots
        imputed_values  = None means it will draw samples uniformly at random in [0,1] to fill missing data.
                        = dict of form {col_name: {row_index: [list of imputed values of length m]}}
                        if a dict is passed, it go thru each col and row index given and use those imputed values.
                        (e.g,. run predict_missing_values.get_dt_imputed_values() and pass its output here)
        outputpath = False (no saved images) or path variable to folder where to save images.
        name = False (no saved images) or string.
        cmap = colormap (ideally diverging). defaults to a blue-white-red diverging color palette
    Output:
        c = dataframe; average of sampled correlation matrices
        mid = dataframe; fisher-inverse of average (fisher(correlation matrix) )
        interval = dataframe; for each cell gives the 95% confidence interval length, i.e., [mid - interval, mid + interval] = 95% confidence interval
    """
    if outputpath and not name:
        name = f"NAME-{int(1000*random())}"
        print(f"No name given, will use name = {name} when saving files")

    ds = [] ## to be filled with data including imputed values

    if imputed_values:
        for k in range(m): ## m'th imputation
            rdf = copy.copy(df) ## copy w/ nans to be filled
            for col in df.columns: # per col
                for i in rdf.index[rdf[col].isna()]: # if that row has a nan in that col
                    rdf.at[i,col] = imputed_values[col][i][k][0] # give it it's m-th imputed value
            if (col == 'familiarity') and ('familiarity' in rdf.columns): #
                rdf["familiarity"] = rdf["familiarity"].apply(lambda x: np.round(x))
            ds.append( df.fillna(rdf) )## fills holes with generated random numbers
    else:
        for k in range(m):
            rdf = pd.DataFrame(np.random.random(df.shape), columns = df.columns) ## generate random numbers for every spot
            if 'familiarity' in rdf.columns:
                rdf["familiarity"] = rdf["familiarity"].apply(lambda x: np.round(x))
            ds.append( df.fillna(rdf) )## fills holes with generated randome numbers

    cs = list(map(lambda a: a.corr(), ds))## our correlation matrices!
    delta = [] #populated with |ave(c1,,..,. ck) - ave(c1, .., c{k+1}) | /n^2
    c = cs[0] ## average of first i data points
    for i in range(1,len(cs)):
        c2 = cs[i]
        cnew = (c*i + c2)/(i+1) ## average matrix w/ new entry added in
        delta.append((c - cnew).abs().sum().sum()/(c.shape[0]*c.shape[1]))
        c = cnew
    ### c now equals average of all datapoints.

    # plot convergence of correlation matrix (ave l1 difference of each step):
    sns.set(rc = {'figure.figsize':(12,12)})
    sns.lineplot(range(len(delta)), list(map(np.log,delta)) )
    # plt.rcParams['text.usetex'] = True
    plt.title(r"$\log( \| \bar{C_i}-\bar{C_{i-1}}\|_{1}/n^2$"+f"\n{name}")
    if (outputpath and name):
        plt.savefig(Path(outputpath,f'imputation-corr-stepsizes-{name}.svg'), format = "svg")
        plt.savefig(Path(outputpath,f'imputation-corr-stepsizes-{name}.png'))
    if verbose: plt.show()

    fisher= lambda r: (1/2)*np.log((1+r)/(1-r)) ## fisher transformation:
    qs = list(map(fisher , cs)) ## apply fisher to ck, so fix i,j. for each M in qs, M[i,j] is a sample of a normal
    Qbar = sum(qs)/len(qs) ### Qbar(i,j) is mean
    U = 1/(c.shape[0]-3) ## var(Qi|Yi), but constant in i. Same for each pair of variables too
    B = sum(list(map(lambda Q: (Q-Qbar)**2, qs)))/(m-1) ## matrix of between imputation variances
    ## T-Test, need variance (T) as below and degrees of freedom (v) as below, but
    ## v>1000 so same as normal. we'll use that.
    # T = (1+(1/m))*Bs + U ## variance for t-test
    # v = (m-1)*(1+ (U/ ( (1+(1/m))*Bs )))**2 ## degrees of freedom for ttest

    ##Normal 95% confidence interval:
    ##A = 1.96  from Z table, P( |Z| > A) =  2x0.025 = 5%
    # Qbar - 1.96*sqrt(U+B) ## top of interval in z
    # Qbar + 1.96*sqrt(U+B) ## bottom of interval in z
    ## invert back to correlation
    invfisher = lambda z:  (np.exp(2*z)-1)/(np.exp(2*z)+1)
    # test: invfisher(fisher(.3))
    left_conf = invfisher(Qbar - 1.96*np.sqrt(U+B))
    mid = invfisher(Qbar)
    # right_conf = invfisher(Qbar + 1.96*(U+B))
    interval = mid - left_conf
    return c, mid, interval



    ## plot c, the correlation after imputation plot
    if verbose or outputpath:
        cg = sns.clustermap(c, vmin=-1, vmax = 1, cmap=cmap, figsize = (10,10))
        cg.ax_col_dendrogram.set_visible(False)
        plt.setp(cg.ax_heatmap.get_xticklabels(), rotation = 45, fontsize = 11,  horizontalalignment = "right", rotation_mode ="anchor")
        plt.setp(cg.ax_heatmap.get_yticklabels(), fontsize = 11)
        plt.title(f"Clustermap After Imputation\n{name}", fontsize= 16)
        if (outputpath and name):
            plt.savefig(Path(outputpath,f'imputation-after-clustermap-{name}.svg'), format= "svg")
            plt.savefig(Path(outputpath,f'imputation-after-clustermap-{name}.png'))
        if verbose: plt.show()
        print("clustermap made!")


def conf_interval_plot(mid, interval,  variables = None, outputpath = None, name = None, cmap = cmap):
    '''
    inputs: mid, interval: dataframes from mi_corr output
            variables: list of names of columns in mid nad interval.
                if variables are passed, plots only the pairs of these variables
            outputpath: path variable. if passed, it will save plots in this folder.
            name = string, name to be included in the saved plots
    output: none, makes plots, saves them if outputpath is given.
    '''
    if outputpath and not name:
        name = f"NAME-{int(1000*random())}"
        print(f"No name given, will use name = {name} when saving files")

    if not variables:
        variables = mid.columns
    if len(variables)>25:
        print("whoah! pretty big plot coming.... \nthis will take a while")

    records = [ {"correlation": f"{v1},\n{v2}" , "mid": mid[v1][v2], "interval": interval[v1][v2]} for i,v1 in enumerate(variables[:-1]) for v2 in variables[i+1:] ]
    df = pd.DataFrame.from_records(records).sort_values(by = 'mid')

    # Set sns plot style back to 'poster'. This will make bars wide on plot
    # sns.set_context("paper")

    # Define figure, axes, and plot
    fig, ax = plt.subplots(figsize=(10, 10))
    # Error bars for 95% confidence interval: Can increase capsize to add whiskers
    df.plot(x='correlation', y='mid', kind='bar',
                 ax=ax, color='none', fontsize=16,
                 ecolor='steelblue',
                 rot=45,
                 capsize=0,
                 yerr='interval', legend=False)
    # Turns off grid on the left Axis.
    plt.xticks(ha='right')
    ax.grid(False)
    # Set title & labels
    plt.title('Multiple Imputation Correlation 95\% Confidence Intervals',fontsize=16)
    # plt.xticks(rotation=90)
    ax.set_ylabel(' ',fontsize=22)
    ax.set_xlabel(' ',fontsize=22)

    # Coefficients
    ax.scatter(x=pd.np.arange(df.shape[0]),
               marker='o', s=80,
               y=df['mid'], color='steelblue')

    # Line to define zero on the y-axis
    ax.axhline(y=0, linestyle='--', color='red', linewidth=1)
    plt.tight_layout()
    if outputpath:
        plt.savefig(Path(outputpath, f"confidence_interval_{name}.svg"), format = 'svg')
        plt.savefig(Path(outputpath, f"confidence_interval_{name}.png"))
    plt.show()
    return
