import argparse
import time
from pathlib import Path
import os
import json

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

import pandas as pd
import numpy as np

import eval as ev
from sklearn.metrics.pairwise import cosine_distances

import warnings


parser = argparse.ArgumentParser(description='This script runs an evaluation using three possible methods: classification,'
                                             'a silhouette metric, and a Mann Whitney U Test metric. To run these tests,'
                                             'a csv of the data representation with rows as samples and columns as features'
                                             'is required. Additionally, a csv of metadata is required with sample in the rows'
                                             'and at least one column defining discrete (categorical) distinction of interest.'
                                             'If there are many unique values in the categorical metadata of interest, '
                                             'evaluation may not work well. Additionally, it is assumed that there is '
                                             'only one unique metadata value for each sample name.')

parser.add_argument('-t', '--test', type=str,
                    help='Path to dataset test representation csv. Samples as rows and features as columns.', required=True)

parser.add_argument('-m', '--meta', type=str,
                    help='Path to metadata csv. Named samples as first row and characteristic metadata as columns. '
                         'If running evaluation with classifier machine learning models, train dataset metadata '
                         'must also be included.', required=True)

parser.add_argument('-d', '--train', type=str,
                    help='Path to dataset representation csv. Samples as rows and features as columns. Only required if '
                         'running evaluation by building classifiers.', default=[])

parser.add_argument('-c', '--characteristic', type=str,
                    help='Name of metadata column with discrete (categorical) metadata of interest. Defaults to the second '
                    'column of the metadata csv.', default=[])

parser.add_argument('-o', '--output_directory', type=str,
                    help='Directory to which output will be written', default='output')

parser.add_argument('-e', '--eval', type=str, nargs="+",
                    help='List of one or more discrete evaluation methods to run. Defaults to all methods. '
                         'Options are: classification, silhouette, utest',
                    default=['classification', 'silhouette', 'utest'])

parser.add_argument('-p', '--plots', type=str,
                    help='Plot barplots of outcomes for each method. Options are True  or False for no plots.',
                    default='True')

args = parser.parse_args()


if args.output_directory == 'output':
    args.output_directory = (Path.cwd() / 'output').as_posix()
    
## Make run folder
run_id = time.strftime("eval_%Y_%m_%d-%H_%M_%S")
qc_path =  Path(args.output_directory) / 'Quality Assessment' / 'Evaluation' / run_id
if not os.path.exists(qc_path): qc_path.mkdir(parents=True)
print("\nEvaluation Run: ", run_id)

## Write run details to json file
argparse_dict = vars(args)
with open((qc_path / 'evaluation_arguments.json'), "w") as handle:
    json.dump(argparse_dict, handle)

## Import data
if args.train:
    train = pd.read_csv(Path(args.train), index_col=0)
test = pd.read_csv(Path(args.test), index_col=0)
meta = pd.read_csv(Path(args.meta), index_col=0)

## Check to see overap of indexes
overlap = pd.Series(list(set(test.index) & set(meta.index)))

if overlap.shape[0] < test.shape[0]:
    warnings.warn("\nOnly {} samples of {} total test dataset samples were located in the metadata."
                  .format(overlap.shape[0], test.shape[0]))
    test = test.loc[overlap]
    meta = meta.loc[overlap]

if args.train:
    trainoverlap = pd.Series(list(set(train.index) & set(meta.index)))
    if trainoverlap.shape[0] < train.shape[0]:
        warnings.warn("\nOnly {} samples of {} total train dataset samples were located in the metadata."
                      .format(trainoverlap.shape[0], train.shape[0]))




## Get vector of characteristic
if args.characteristic:
    try:
        ytest = meta[args.characteristic][test.index]
        if args.train:
            ytrain = meta[args.characteristic][train.index]
    except:
        raise Exception("Unable to map input characteristic to metadata column.")
else:
    ytest = meta.iloc[:, 0][test.index]
    if args.train:
        ytrain = meta.iloc[:, 0][train.index]

## Evaluate number of categories in y
print("\nThere were {} categories identified in the test data.".format(ytest.unique()))
print(ytest.value_counts())

## Configure plots
if args.plots == 'True':
    plt.rcParams["figure.dpi"] = 300

## Run evaluations
for e in args.eval:
    if e == 'classification':

        if args.train:
            # Train and test classifiers
            accuracy, precision, recall = ev.get_all_classifier_metrics(train, ytrain, test, ytest)

            class_df = pd.DataFrame(zip(accuracy.values(), precision.values()),
                                    index=accuracy.keys(),
                                    columns=['Accuracy', 'Precision'])
            class_df.to_csv(qc_path / 'classification_results.csv')

            # Plot figure
            if args.plots == 'True':

                class_df = pd.melt(class_df.reset_index(), id_vars='index')
                class_df.columns = ['Model', 'Metric', 'Value']

                plt.clf()

                ax = sns.barplot(x='Model', y='Value', hue='Metric', data=class_df,
                                 palette='colorblind')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
                ax.set_ylim(0,1)

                fig = ax.get_figure()
                fig.savefig(qc_path/'classification_barplot.png',
                            bbox_inches='tight')
        else:
            warnings.warn("No training data was uploaded so classification evaluation was not run.")

            
                
    if e == 'silhouette':
        
        # Calculate similarities
        pearson = pd.DataFrame(test, index=test.index).transpose().corr()
        spearman = pd.DataFrame(test, index=test.index).transpose().corr(method="spearman")
        cos = pd.DataFrame(cosine_distances(test), index=test.index, columns=test.index)
        ccc = ev.calculate_ccc(test)
        euc = ev.calculate_euc(test)
        man = ev.calculate_manhattan(test)

        # Calculate silhouette metrics
        ps = ev.calculate_silhouette_from_distance(pearson, ytest, distance=False)
        ss = ev.calculate_silhouette_from_distance(spearman, ytest, distance=False)
        cs = ev.calculate_silhouette_from_distance(cos, ytest, distance=True)
        ccs = ev.calculate_silhouette_from_distance(ccc.round(6), ytest, distance=False)
        es = ev.calculate_silhouette_from_distance(euc, ytest, distance=True)
        ms = ev.calculate_silhouette_from_distance(man, ytest, distance=True)

        # Build dataframe
        sil_df = pd.DataFrame(zip(ps.values(), ss.values(),
                                  ccs.values(), es.values(),
                                  ms.values(), cs.values()),
                                index=ps.keys(),
                                columns=['Pearson', 'Spearman',
                                         'CCC', 'Euclidean',
                                         'Manhattan', 'Cosine'])
        sil_df.to_csv(qc_path / 'silhouette_results.csv')

        # Plot figure
        if args.plots == 'True':
            sil_df2 = pd.melt(sil_df.reset_index(), id_vars='index')
            sil_df2.columns = ['Comparison', 'Similarity', 'Value']

            plt.clf()

            if len(sil_df2['Comparison'].unique()) < 4:

                ax = sns.barplot(x='Comparison', y='Value', hue='Similarity', data=sil_df2,
                                 palette='colorblind')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
                ax.set_ylim(-0.1, 1)

                fig = ax.get_figure()
                fig.savefig(qc_path/'silhouette_barplot.png',
                            bbox_inches='tight')
            else:
                cg = sns.clustermap(sil_df,
                                    cmap="mako", vmin=-0.1, vmax=1)
                plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation=30)
                plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=30)

                cg.savefig(qc_path/'silhouette_heatmap.png',
                            bbox_inches='tight')
    if e == 'utest':

        if ('silhouette' not in args.eval):
            # Calculate similarities only if not already run
            pearson = pd.DataFrame(test, index=test.index).transpose().corr()
            spearman = pd.DataFrame(test, index=test.index).transpose().corr(method="spearman")
            cos = pd.DataFrame(cosine_distances(test), index=test.index, columns=test.index)
            ccc = ev.calculate_ccc(test)
            euc = ev.calculate_euc(test)
            man = ev.calculate_manhattan(test)

        # Build data table for similarity distributions
        pearsondf = ev.discrete_correlation_distribution(pearson, ytest)
        spearmandf = ev.discrete_correlation_distribution(spearman, ytest)
        cosdf = ev.discrete_correlation_distribution(cos, ytest)
        cccdf = ev.discrete_correlation_distribution(ccc, ytest)
        eucdf = ev.discrete_correlation_distribution(euc, ytest)
        mandf = ev.discrete_correlation_distribution(man, ytest)

        # Calculate t-test metrics
        pt = ev.calculate_mannwhitneyu_from_distribution(pearsondf, ytest)
        st = ev.calculate_mannwhitneyu_from_distribution(spearmandf, ytest)
        ct = ev.calculate_mannwhitneyu_from_distribution(cosdf, ytest, distance=True)
        cct = ev.calculate_mannwhitneyu_from_distribution(cccdf, ytest)
        et = ev.calculate_mannwhitneyu_from_distribution(eucdf, ytest, distance=True)
        mt = ev.calculate_mannwhitneyu_from_distribution(mandf, ytest, distance=True)

        # Build dataframe
        utest_df = pd.DataFrame(zip(pt.values(), st.values(),
                                  cct.values(), et.values(),
                                  mt.values(), ct.values()),
                                index=pt.keys(),
                                columns=['Pearson', 'Spearman',
                                         'CCC', 'Euclidean',
                                         'Manhattan', 'Cosine'])
        utest_df.to_csv(qc_path / 'utest_results.csv')

        # Plot figure
        if args.plots == 'True':
            utest_df2 = pd.melt(utest_df.reset_index(), id_vars='index')
            utest_df2.columns = ['Comparison', 'Similarity', 'Value']

            plt.clf()

            if len(utest_df2['Comparison'].unique()) < 4:

                ax = sns.barplot(x='Comparison', y='Value', hue='Similarity', data=utest_df2,
                                 palette='colorblind')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
                ax.set_ylim(0, 1)

                fig = ax.get_figure()
                fig.savefig(qc_path/'utest_barplot.png',
                            bbox_inches='tight')
            else:
                cg = sns.clustermap(utest_df,
                                    cmap="mako", vmin=0, vmax=1)
                plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation=30)
                plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=30)

                cg.savefig(qc_path/'utest_heatmap.png',
                            bbox_inches='tight')

