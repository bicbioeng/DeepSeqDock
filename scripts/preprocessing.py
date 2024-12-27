# Supporting functions for 0_0_2_Preprocessing tasks

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import pickle

from zlib import crc32
import numpy as np

import pandas as pd
from scipy.interpolate import make_interp_spline

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
import warnings

# r_deseq2 = importr('DESeq2')
# r_base = importr('base')

###Use of these functions presupposes that the datasets have been brought to the same format as the
### train dataset (pd dataframe where rows are samples, genes are columns) with the same number and identity as the
### train dataset. Failure to account for this may result in failure of functions...

import time


def TicTocGenerator():
    # Generator that returns time differences
    ti = 0  # initial time
    tf = time.time()  # final time
    while True:
        ti = tf
        tf = time.time()
        yield round(tf - ti, 3)  # returns the time difference


# This will be the main function through which we define both tic() and toc()
def toc(tictocgen, logged=True, tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(tictocgen)
    if tempBool:
        print("Elapsed time: %f seconds.\n" % tempTimeInterval)
        if logged:
            return tempTimeInterval


def tic(tictocgen):
    # Records a time in TicToc, marks the beginning of a time interval
    toc(tictocgen, tempBool=False)


def split_train_test_valid_by_id(ids, valid_ratio=0.1, test_ratio=0.1):
    """Reproducibly splits name id list by end of CRC-32 checksum

    This approach extends the function in Hands-On Machine Learning with SciKit-Learn, Keras & TensorFlow by Aurelion
    Geron(Second Edition) pp52 to include a validation split, and only is intended to split an id list (not a dataset).

    :param ids: The sample index from the dataset to be split
    :type ids: :class:'pandas.Series'
    :param valid_ratio: The ratio of samples to be assigned to the validation split [0:1], defaults to 0.1
    :type valid_ratio: float
    :param test_ratio: The ratio of samples to be assigned to the test split [0:1], defaults to 0.1
    :type test_ratio: float

    :return: Tuple of ids in train split, ids in validation split, ids in test split
    :rtype: tuple
    """
    idn = range(ids.size)
    in_test_set = [(crc32(np.int64(i)) & 0xffffffff < test_ratio * 2 ** 32) for i in idn]
    in_valid_set = [(test_ratio * 2 ** 32 <= crc32(np.int64(i)) & 0xffffffff <= (valid_ratio + test_ratio) * 2 ** 32)
                    for i in idn]
    in_either = [i + j for i, j in zip(in_test_set, in_valid_set)]
    in_train_set = [False if i != 0 else True for i in in_either]
    return ids[in_train_set], ids[in_valid_set], ids[in_test_set]


class NiceifyRawCountData(BaseEstimator, TransformerMixin):
    """Scikit-learn style class to 1) remove genes with 0 expression in all training data, 2) add pseudocount of 1
    3) cap vales at cap sample-wise value"""
    def __init__(self, cap=0.99):
        self.cap = cap

    def fit(self, X, y=None):
        self.zerogenes = (X.sum(axis=0) == 0)
        return self

    def transform(self, X, y=None):
        Xz = X.loc[:, ~self.zerogenes.values] + 1
        scap = Xz.quantile(self.cap, axis=1).apply(np.ceil)

        return Xz.clip(0, scap, axis=0)


# def make_r_df(data):
#     """Use rpy2 to make a DataFrame into a r dataframe object"""
#     with localconverter(ro.default_converter + pandas2ri.converter):
#         r_data = ro.conversion.py2rpy(data)
#         return r_data

def full_scaling_workflow(train, run_id, dataset_path, prefix, valid=None, test=None):
    """Convenience function to run standard scaling, and global standard scaling"""
    scaler = StandardScaler()
    train_ss = pd.DataFrame(scaler.fit_transform(train),
                                   index=train.index,
                                   columns=train.columns)
    glb_scaler = GlobalStandardScaler()
    train_gs = glb_scaler.fit_transform(train)
    min_scaler = MinMaxScaler()
    train_sm = pd.DataFrame(min_scaler.fit_transform(train),
                                   index=train.index,
                                   columns=train.columns)
    glb_min = GlobalMinMaxScaler()
    train_gm = glb_min.fit_transform(train)

    train_ss.to_csv(dataset_path / (run_id + '-train_' + prefix + '_ss.csv'))
    train_gs.to_csv(dataset_path / (run_id + '-train_' + prefix + '_gs.csv'))
    train_sm.to_csv(dataset_path / (run_id + '-train_' + prefix + '_sm.csv'))
    train_gm.to_csv(dataset_path / (run_id + '-train_' + prefix + '_gm.csv'))

    returndict = dict(train_ss=train_ss, train_gs=train_gs, train_sm=train_sm, train_gm=train_gm)

    if isinstance(valid, pd.DataFrame):
        valid_ss = pd.DataFrame(scaler.transform(valid),
                                       index=valid.index,
                                       columns=valid.columns)
        valid_gs = glb_scaler.transform(valid)
        valid_sm = pd.DataFrame(min_scaler.transform(valid),
                                       index=valid.index,
                                       columns=valid.columns)
        valid_gm = glb_min.transform(valid)

        valid_ss.to_csv(dataset_path / (run_id + '-valid_' + prefix +'_ss.csv'))
        valid_gs.to_csv(dataset_path / (run_id + '-valid_' + prefix + '_gs.csv'))
        valid_sm.to_csv(dataset_path / (run_id + '-valid_' + prefix + '_sm.csv'))
        valid_gm.to_csv(dataset_path / (run_id + '-valid_' + prefix + '_gm.csv'))

        returndict.update({'valid_ss':valid_ss,
        'valid_gs':valid_gs,
        'valid_sm':valid_sm,
        'valid_gm':valid_gm})

    if isinstance(test, pd.DataFrame):
        test_ss = pd.DataFrame(scaler.transform(test),
                                      index=test.index,
                                      columns=test.columns)
        test_gs = glb_scaler.transform(test)
        test_sm = pd.DataFrame(min_scaler.transform(test),
                                      index=test.index,
                                      columns=test.columns)
        test_gm = glb_min.transform(test)

        test_ss.to_csv(dataset_path / (run_id + '-test_' + prefix + '_ss.csv'))
        test_gs.to_csv(dataset_path / (run_id + '-test_' + prefix + '_gs.csv'))
        test_sm.to_csv(dataset_path / (run_id + '-test_' + prefix + '_sm.csv'))
        test_gm.to_csv(dataset_path / (run_id + '-test_' + prefix + '_gm.csv'))

        returndict.update({'test_ss':test_ss,
                           'test_gs':test_gs,
                           'test_sm':test_sm,
                           'test_gm':test_gm})

        with open((dataset_path / (run_id + '-' + prefix +'_dict.pickle')), 'wb') as handle:
            pickle.dump({'Std_s': scaler,
                         'Std_g': glb_scaler,
                         'Min_s': min_scaler,
                         'Min_g': glb_min}, handle, protocol=4)

        return returndict


def plot_distributions(datas, n=5, random_state=None):
    """Convenience function to plot distributions and mean/var relationship of a subset of features/samples"""
    fig, axes = plt.subplots(2, 2)


    samples = datas.sample(n=n, axis=0, random_state=random_state)
    features = datas.sample(n=n, axis=1, random_state=random_state)

    sns.kdeplot(data=samples.transpose().astype(float), ax=axes[0, 0])
    sns.kdeplot(data=features.astype(float), ax=axes[1, 0])

    samples.transpose().boxplot(ax=axes[0, 1])
    features.boxplot(ax=axes[1, 1])

def save_distributions(datas, outdir, prefix, n=5, random_state=None, genes=False):
    """Convenience function to save plotted distributions to outputdir"""

    plt.close()
    samples = datas.sample(n=n, axis=0, random_state=random_state)
    features = datas.sample(n=n, axis=1, random_state=random_state)

    sns.kdeplot(data=samples.transpose().astype(float))
    plt.savefig(outdir / (prefix + '_sampledis.png'), bbox_inches='tight')
    plt.close()

    if not genes:
        sns.kdeplot(data=features.astype(float))
        plt.savefig(outdir / (prefix + '_genedis.png'), bbox_inches='tight')
        plt.close()
    else:
        genesamples = datas.loc[:, genes]
        sns.kdeplot(data=genesamples.astype(float))
        plt.savefig(outdir / (prefix + '_genedis.png'), bbox_inches='tight')
        plt.close()




def plot_meanvar(datas):
    """Convenience function to plot mean/variance relationship of a dataset"""
    t_mean = datas.mean(axis=1)
    t_var = datas.var(axis=1)
    t_meanvar = pd.DataFrame({'mean': t_mean, 'var': t_var}).sort_values(by=['mean']).drop_duplicates(subset=['mean'])

    meanvar_spline = make_interp_spline(t_meanvar['mean'], t_meanvar['var'], k=1)

    x_ = np.linspace(t_mean.min(), t_mean.max(), 30)
    y_ = meanvar_spline(x_)

    plt.scatter(t_mean, t_var, alpha=0.25)
    plt.plot(x_, y_, "k-")
    plt.title('Mean-Variance Relationship')


class GlobalStandardScaler(BaseEstimator, TransformerMixin):
    """Standard scaling based on global means and standard deviations"""
    def __init__(self, mean_=None, std_=None):
        self.mean_ = mean_
        self.std_ = std_

    def fit(self, X, y=None):
        self.mean_ = X.values.mean()
        self.std_ = X.values.std(ddof=1)
        return self

    def transform(self, X, y=None):
        return (X - self.mean_) / self.std_

class GlobalMinMaxScaler(BaseEstimator, TransformerMixin):
    """Min max scaling based on global min and max values"""
    def __init__(self, min_=None, max_=None):
        self.min_ = min_
        self.max_ = max_

    def fit(self, X, y=None):
        self.min_ = X.values.min()
        self.max_ = X.values.max()
        return self

    def transform(self, X, y=None):
        return (X - self.min_) / (self.max_ - self.min_)


def gene_length_scaling(datas, genelength):
    """Convenience function to scale features according to gene length"""

    # remove genes not in gene_length list
    allgenelength = datas.columns.isin(genelength.index)
    datas = datas.loc[:, allgenelength]
    genelength = genelength.loc[datas.columns]
    print("{} genes match gene length files.".format(genelength.size))

    datas_scaled = datas.divide(genelength, axis='columns')
    return datas_scaled


class LSPreprocessing(BaseEstimator, TransformerMixin):
    """Sklearn-style class to perform library scale preprocessing a la Scanpy"""

    def fit(self, X, y=None):
        self.median_ = X.sum(axis=1).median()
        return (self)

    def transform(self, X, y=None):
        Xt = X.mul(self.median_ / X.sum(axis=1), axis="rows")
        return Xt.apply(np.log)


class TPMPreprocessing(BaseEstimator, TransformerMixin):
    """Class to calculate TPM from pseudocounts, gene length"""
    def __init__(self, genelength=None):
        self.genelength = genelength

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        rpk = gene_length_scaling(X, self.genelength)
        pkm = (rpk / 1E6).sum(axis=1)
        rpkm = rpk.div(pkm, axis=0)
        return rpkm.apply(np.log)

class QuantilePreprocessing(BaseEstimator, TransformerMixin):
    """Class to quantile preprocess dataset"""
    def __init__(self, means_=None):
        self.means_ = means_

    def fit(self, X, y=None):
        Xs = pd.DataFrame(np.sort(X.values, axis=1)).apply(np.log)
        means_ = Xs.mean(axis=0)
        means_.index = np.arange(1, len(means_) + 1)
        self.means_ = means_
        return self

    def transform(self, X, y=None):
        sorted = X.rank(method="min", axis=1).stack().astype(int).map(self.means_).unstack()
        return sorted

class RLEPreprocessing(BaseEstimator, TransformerMixin):
    """Class for relative log expression preprocessing, also called median of ratios method"""
    def __init__(self, pseudoref=None):
        self.pseudoref = pseudoref

    def fit(self, X, y=None):
        self.pseudoref = stats.gmean(X, axis=0)
        return self

    def transform(self, X, y=None):
        ls = X.div(self.pseudoref, axis=1).median(axis=1)
        return X.div(ls, axis=0).apply(np.log)

def rle_preprocessing(datas, geo_mean=None, floor=0.05, ceiling=0.95):
    """Modified version of rle scaling in DESeq2 - includes addition of 0.1 pseudocount since intended for large
    datasets - gene geometric means were predominated by 0s since at least one sample had a count of 0 for a given
    gene. Quantile capping of floor and ceiling also performed to mitigate effect of outliers."""

    datas, floors, ceilings = quantile_floorcap(datas + 0.1, floor=floor, ceiling=ceiling)

    if geo_mean is None:
        geo_mean = stats.gmean(datas, axis=0)
    # remove genes with geo_mean==0
    datas_z = datas.iloc[:, geo_mean > 0]
    geo_mean_z = geo_mean[geo_mean > 0]

    # calculate ratio of each sample to the reference
    datas_ratio = datas_z.div(geo_mean_z, axis="columns")

    # calculate median of sample-wise ratios
    datas_median = datas_ratio.median(axis=1)

    # scale samples
    datas_scaled = datas_z.div(datas_median, axis="index")

    # apply ln
    datas_ln = datas_scaled.apply(np.log)

    return datas_ln, geo_mean, floors, ceilings


def quantile_floorcap(datas, floor=0.05, ceiling=0.95):
    """Cap dataset at floor and ceiling quantiles to mitigate outlier effect"""
    if type(floor) is float and type(ceiling) is float:
        floors = datas.quantile(q=floor, axis="rows")
        ceilings = datas.quantile(q=ceiling, axis="rows")

    elif type(floor) is pd.Series and type(ceiling) is pd.Series:
        floors = floor
        ceilings = ceiling

    else:
        warnings.warn('Mismatch between floor and ceiling types or type is not a float/pandas Series.')

    for i, f in enumerate(floors):
        datas.iloc[datas.iloc[:, i] < f, i] = f

    for i, c in enumerate(ceilings):
        datas.iloc[datas.iloc[:, i] > c, i] = c

    return datas, floors, ceilings

# def make_r_DE_obj(data):
#     data = data.transpose()
#     r_data = make_r_df(data)
#     r_col_data = ro.DataFrame({'sample': r_base.colnames(r_data)})
#     r_DEobject = r_deseq2.DESeqDataSetFromMatrix(countData=r_data,
#                                                  colData=r_col_data,
#                                                  design=ro.Formula('~1'))
#     r_DEobject = r_deseq2.DESeq(r_DEobject)
#     return r_DEobject

# def execute(cmd):
#     popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
#     for stdout_line in iter(popen.stdout.readline, ""):
#         yield stdout_line
#     popen.stdout.close()
#     return_code = popen.wait()
#     if return_code:
#         raise subprocess.CalledProcessError(return_code, cmd)
