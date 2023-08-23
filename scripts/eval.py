import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import silhouette_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import NuSVC
from sklearn.svm import SVC

import matplotlib.pyplot as plt
import seaborn as sns

import warnings

from itertools import combinations

from scipy.spatial import distance_matrix
from scipy.stats import mannwhitneyu

from matplotlib.patches import Patch


def metrics_from_classifier(skmodel, Xtrain, ytrain, Xtest, ytest):
    """Helper function to use default Sci-kit learn classifiers and return metrics"""
    skmodel.fit(Xtrain, ytrain)
    ytest_pred = skmodel.predict(Xtest)

    acc = accuracy_score(ytest, ytest_pred)
    prec = precision_score(ytest, ytest_pred, average="weighted")
    rec = recall_score(ytest, ytest_pred, average="weighted")

    return acc, prec, rec

def get_all_classifier_metrics(Xtrain, ytrain, Xtest, ytest):

    accuracy = {}
    precision = {}
    recall = {}

    accuracy['Logistic Regression'], precision['Logistic Regression'], recall['Logistic Regression'] = metrics_from_classifier(LogisticRegression(),
                                                                                                                                  Xtrain, ytrain, Xtest, ytest)
    print('LR')
    accuracy['Random Forest'], precision['Random Forest'], recall['Random Forest'] = metrics_from_classifier(RandomForestClassifier(),
                                                                                                                Xtrain, ytrain, Xtest, ytest)
    print('RF')
    # accuracy['Gradient Boosting'], precision['Gradient Boosting'], recall['Gradient Boosting'] = metrics_from_classifier(GradientBoostingClassifier(),
    #                                                                                                                         Xtrain, ytrain, Xtest, ytest)
    # print('XGB')
    accuracy['Naive Bayes'], precision['Naive Bayes'], recall['Naive Bayes'] = metrics_from_classifier(GaussianNB(),
                                                                                                          Xtrain, ytrain, Xtest, ytest)
    print('NB')

    try:
        accuracy['RBF SVM'], precision['RBF SVM'], recall['RBF SVM'] = metrics_from_classifier(NuSVC(),
                                                                                               Xtrain, ytrain, Xtest, ytest)
    except:
        print('Switching to traditional SVC')
        accuracy['RBF SVM'], precision['RBF SVM'], recall['RBF SVM'] = metrics_from_classifier(SVC(),
                                                                                               Xtrain, ytrain, Xtest,
                                                                                               ytest)

    print('SVM')
    return accuracy, precision, recall


def discrete_correlation_distribution(Xcorrelation, ytest):
    """Input: correlation df (Xcorrelation) and Series with ample condition information (ytest).
    Output: df with three columns: Cond1, Cond2, and Values indicating the
    two conditions the correlation is coming from and the value of the correlation """
    yvalues = ytest.unique()

    # make a copy of correlation dataframe so can edit indices without affecting original
    corrdataframe = Xcorrelation.copy()
    corrdataframe.index = ytest
    corrdataframe.columns = ytest

    # prime dataframe
    full_cdf = pd.DataFrame([], columns=('Cond1', 'Cond2', 'Values'))

    for y in yvalues:

        # For condition = condition, want only upper triangle of correlation dataframe witout diagonal
        df = corrdataframe.loc[y,y]
        values = df.values[np.triu_indices(df.shape[0], k=1)]

        # Make dataframe for condition = condition values
        cdf = pd.DataFrame({'Cond1':y,
                        'Cond2':y,
                        'Values':values})
        # Append to primed dataframe
        full_cdf = pd.concat([full_cdf, cdf], ignore_index=True, sort=False)

        # Identify condition != condition
        otheryvalues = yvalues[yvalues != y]

        # Iterate through other conditions
        for o in otheryvalues:

            # Identify subset of correlation dataframe for condition, other condition
            df = corrdataframe.loc[y,o]
            values = df.to_numpy().flatten()

            cdf = pd.DataFrame({'Cond1':y,
                            'Cond2':o,
                            'Values':values})
            # Append to primed dataframe
            full_cdf = pd.concat([full_cdf, cdf], ignore_index=True, sort=False)


    return full_cdf


def plot_similiarities(df, path, labels=['Cond1', 'Cond2', 'Values'], metric="Correlation"):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pal = sns.color_palette('colorblind')
        g = sns.FacetGrid(df, row=labels[0], hue=labels[1], aspect=4, height=1.5, palette=pal,
                          sharey=False, sharex=False)

        # Draw the densities in a few steps
        g.map(sns.kdeplot, "Values",
              bw_adjust=.5, clip_on=False,
              fill=True, alpha=1, linewidth=1.5)
        g.add_legend(title='')
        # Add pretty white line to outline densities
        g.map(sns.kdeplot, "Values", clip_on=False, color="w", lw=0.5, bw_adjust=.5)

        # passing color=None to refline() uses the hue mapping
        g.refline(y=0, linewidth=0.5, linestyle="-", color=None, clip_on=False)

        # Define and use a simple function to label the plot in axes coordinates
        def label(x, color, label):
            ax = plt.gca()
            ax.text(0, .2, x.unique()[0], fontweight="bold", color='k',
                    ha="left", va="center", transform=ax.transAxes)

        g.map(label, labels[0])

        # Remove axes details that don't play well with overlap
        g.set_titles("")
        g.set(yticks=[], ylabel="")
        g.set(xlabel=metric)
        g.despine(bottom=True, left=True)

        plt.savefig(path, transparent=True)

def calculate_ccc(Xtest):
    df = Xtest.transpose()

    numerator = df.cov() * 2

    denominator = pd.DataFrame(np.zeros(numerator.shape), index=numerator.index, columns=numerator.columns)
    var = df.var()
    mu = df.mean()

    denominator = denominator.add(mu, axis='rows')
    denominator = (denominator.sub(mu, axis='columns')) ** 2

    denominator = denominator.add(var)
    denominator = denominator.add(var, axis='rows')

    ccc = numerator.div(denominator)
    return ccc

def calculate_euc(Xtest):
    return pd.DataFrame(distance_matrix(Xtest, Xtest), index=Xtest.index, columns=Xtest.index)

def calculate_manhattan(Xtest):
    return pd.DataFrame(distance_matrix(Xtest, Xtest, p=1), index=Xtest.index, columns=Xtest.index)


def concordance_correlation_coefficient(y_true, y_pred):
    """Concordance correlation coefficient. I was too lazy to type it out, so I found this at:
    https://rowannicholls.github.io/python/statistics/agreement/correlation_coefficients.html"""
    # Remove NaNs
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred
    })
    df = df.dropna()
    y_true = df['y_true']
    y_pred = df['y_pred']
    # Pearson product-moment correlation coefficients
    cor = np.corrcoef(y_true, y_pred)[0][1]
    # Mean
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    # Variance
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    # Standard deviation
    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)
    # Calculate CCC
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    return numerator / denominator


def calculate_silhouette_from_distance(Xcorrelation, ytest, distance=True):
    """ Calculate silhouette score from previously calculated distance metrics for each pair of condition combinations. For
    correlations and cos similiarity, values will be transformed by subtracting matrix from 1 (see cosine distance approach),
    for correlations, this will transform the interval to [2,0] where 0 is an ideal score"""
    yvalues = ytest.unique()
    comb = pd.DataFrame(combinations(yvalues, 2))

    if distance == False:
        corrdataframe = 1 - Xcorrelation
    elif distance == True:
        # Make a copy so do not alter original
        corrdataframe = Xcorrelation.copy()

    corrdataframe.index = ytest
    corrdataframe.columns = ytest

    silhouette = []
    if comb.shape[0] > 1:
        silhouette_names = comb.apply(lambda x: (x[0] + '-' + x[1]), axis=1)
    else:
        silhouette_names = (str(comb.iloc[0, 0]) + '-' + str(comb.iloc[0, 1]))

    for i in range(comb.shape[0]):
        # Get subset dataframe with just two conditions
        df = corrdataframe.loc[comb.iloc[i], comb.iloc[i]]

        # Take silhouette score for two-condition set
        ss = silhouette_score(df, df.index, metric="precomputed")
        silhouette.append(ss)

    if comb.shape[0] > 1:
        returned = dict(zip(silhouette_names, silhouette))
    else:
        returned = {silhouette_names: silhouette[0]}

    return returned

def calculate_mannwhitneyu_from_distribution(df, ytest, distance=False):
    """Calculate the Mann Whitney p value based on the comparison of condition1 vs all other conditions with values
    coming from the similarity distributions"""
    yvalues = ytest.unique()
    df.index = df.Cond1

    mw_p = []

    for y in yvalues:
        # Select subset of distribution df belonging to one condition
        conddf = df.loc[y, :]

        # Prepare for slicing
        conddf.index = conddf.iloc[:, 1]
        othervalues = yvalues[yvalues != y]

        # Mann whitney u test
        cond1 = np.array(conddf.loc[y, 'Values'], dtype=float)
        cond2 = np.array(conddf.loc[othervalues, 'Values'], dtype=float)

        if not distance:
            _, pvalue = mannwhitneyu(cond1, cond2, alternative='greater')                   
        else:
            _, pvalue = mannwhitneyu(cond1, cond2, alternative='less')

        # Cap -log pvalue at 100
        if pvalue < 1e-100:
            pvalue = 1e-100
        mw_p.append(-np.log10(pvalue))

    return dict(zip(yvalues, mw_p))


def plot_preprocessing_metric_heatmap(df, scaling_map, preprocessing_map, vmin=0, vmax=1, font=1):
    """Convenience function to plot heatmap from metric dataframe for preprocessing data with color bar indicating scaling"""
    sns.set(font_scale=font)

    palette1 = dict(zip(["Feature Scaling", "Global Scaling"], sns.color_palette("Set2")[0:2]))
    row_colors = df.index.map(scaling_map).map(palette1)
    df.index = df.index.map(preprocessing_map)

    handles = [Patch(facecolor=palette1[name]) for name in palette1]

    cg = sns.clustermap(df, col_cluster=False,
                        row_colors=row_colors,
                        cmap="mako", vmin=vmin, vmax=vmax,
                        annot=True)
    plt.legend(handles, palette1, title="Scaling",
               bbox_to_anchor=(0.43, 1), bbox_transform=plt.gcf().transFigure, loc='upper right')
    plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation=30)
