import argparse
import time
from pathlib import Path
import os
import subprocess
import json
# import pickle

from random import randint

import pandas as pd
import numpy as np

import preprocessing as pp

import warnings

parser = argparse.ArgumentParser(
    description='This script preprocesses and scales the training and test/validation datasets '
                'by standard and non-standard RNA-sequencing preprocessing methods. '
                ' Frozen preprocessing is applied by fitting variables to the training dataset'
                ' and applying the frozen algorithm and parameters to test/validation datasets.'
                ' Two scaling methods suitable for downstream AE building can also be applied  -'
                ' a global min-max scaling and a feature min-max scaling. Input datasets'
                ' should be formatted as a csv with samples as rows and features as columns.'
                ' Data is output as csvs indicating preprocessing and scaling method.')
parser.add_argument('-d', '--train_data', type=str,
                    help='Path to training data csv. Samples as rows and features as columns. Must contain same'
                         ' feature list as training data.', required=True)
parser.add_argument('-t', '--test_data', type=str, nargs="*",
                    help='List of paths to validation or test data csv(s) separated by spaces. '
                         'Samples as rows and features as columns.', default=[])
parser.add_argument('-o', '--output_directory', type=str,
                    help='Directory to which output will be written', default='output')
parser.add_argument('-g', '--gene_length', type=str,
                    help='Path to gene length csv. Necessary only if wish to run all, TPM, GeTMM, or GeVST preprocessing.'
                         ' Default is effective gene lengths from an outside sample mapped by Kallisto.',
                    default='supporting/genelength.csv')
parser.add_argument('-p', '--preprocessing_method', type=str, nargs="+",
                    help='List of one or more preprocessing methods to run. Defaults to QT method. '
                         'For all methods, values above the 99th percentile (sample-wise) are capped, a pseudo count of 1'
                         'is added, and gene features with 0 expression across the train dataset are removed.'
                         ' Options are: all, none, LS, TPM, QT, RLE, VST, GeVST, TMM, GeTMM',
                    default=['QT'])
parser.add_argument('-s', '--scaling_method', type=str, nargs="+",
                    help='List of one or more scaling methods to run. Defaults to feature scaling methods.'
                         ' Options are: all, none, Global, Feature',
                    default=['Feature'])
parser.add_argument('--plots', type=str, nargs="+",
                    help='Plot random gene-wise and feature-wise distributions. Options are True for random list of 5 genes,'
                         ' False for no plots, or a list of genes of interest separated by spaces. Will fail if genes'
                         ' are not in dataset features or gene length file (if GeTMM or GeVST is run).',
                    default=['True'])
parser.add_argument('--datetime', type=str, help="Used internally to identify runs")

args = parser.parse_args()


def sklearn_to_json(scaler, file):
    """Function to serialize modified sklearn class objects to a json file"""

    model = {}
    model['init_params'] = scaler.get_params()

    ## np.ndarrays and pd objects cannot by default be serialized - convert to lists
    model['init_params_types'] = it = {}
    model['init_params_index'] = ii = {}
    for name, p in model['init_params'].items():
        it[name] = type(p).__name__
        if type(p) == np.ndarray:
            model['init_params'][name] = p.tolist()
        elif isinstance(p, np.floating):
            model['init_params'][name] = float(p)
        elif type(p) == pd.core.series.Series:
            model['init_params'][name] = p.to_list()
            ii[name] = p.index.to_list()

    model['model_params'] = mp = {}
    model['model_params_types'] = mt = {}
    model['model_params_index'] = mi = {}
    for p in vars(scaler).keys():

        attrb = getattr(scaler, p)
        mt[p] = type(attrb).__name__

        if type(attrb) == np.ndarray:
            mp[p] = attrb.tolist()
        elif isinstance(attrb, np.floating):
            mp[p] = float(attrb)
        elif type(attrb) == pd.core.series.Series:
            mp[p] = attrb.to_list()
            mi[p] = attrb.index.to_list()
        else:
            mp[p] = attrb

    with open(file, "w") as handle:
        json.dump(model, handle)


## Set up vectors for 'all' preprocessing, scaling
if 'all' in args.preprocessing_method:
    args.preprocessing_method = ['none', 'LS', 'TPM', 'QT', 'RLE', 'VST', 'GeVST', 'TMM', 'GeTMM']

if 'all' in args.scaling_method:
    args.scaling_method = ['none', 'Global', 'Feature']

## Make run folder

if args.datetime:
    run_id = ("preprocess_" + args.datetime)
else:
    run_id = time.strftime("preprocess_%Y_%m_%d-%H_%M_%S")
    
if args.output_directory == 'output':
    args.output_directory = (Path.cwd() / 'output').as_posix()
    
if args.gene_length == 'supporting/genelength.csv':
    args.gene_length = (Path(__file__).parents[1] / 'supporting' / 'genelength.csv').as_posix()

dataset_path = Path(args.output_directory) / 'Data Representations' / 'Normalized' / run_id
if not os.path.exists(dataset_path): dataset_path.mkdir(parents=True)

model_path = Path(args.output_directory) / 'Raw Python Package' / 'Normalized' / run_id
if not os.path.exists(model_path): model_path.mkdir(parents=True)

qc_path = Path(args.output_directory) / 'Quality Assessment' / 'Normalized' / run_id
if not os.path.exists(qc_path): qc_path.mkdir(parents=True)

print("Preprocessing Run: ", run_id)

## Write run details to json file
argparse_dict = vars(args)
with open((model_path / (run_id + '-normalization_arguments.json')), "w") as handle:
    json.dump(argparse_dict, handle)

## Load data
train = pd.read_csv(Path(args.train_data), index_col=0).astype('float32')

test_data = {}
if args.test_data:
    for t in args.test_data:
        test_data[t] = pd.read_csv(Path(t), index_col=0).astype('float32')

gene_length = pd.read_csv((Path(__file__).parents[1] / 'supporting' / 'genelength.csv'), index_col=0).iloc[:, 0]

seed = randint(1, 100)

## Check data for numeric, NAs
if not train[~train.applymap(lambda x: isinstance(x, (int, float))).all(1)].empty:
    raise Exception("Imported train data contain non-numeric elements.")

if train.isna().to_numpy().sum() > 0:
    raise Exception("Imported train data contains NAs or empty values.")

if args.test_data:
    for t in args.test_data:
        if not test_data[t][~test_data[t].applymap(lambda x: isinstance(x, (int, float))).all(1)].empty:
            raise Exception("Imported dataset " + t + " contains non-numeric elements.")

        if test_data[t].isna().to_numpy().sum() > 0:
            raise Exception("Imported dataset " + t + " contains NAs or empty values.")

## Niceify data - remove outliers, add pseudocount, remove 0 genes
niceify = pp.NiceifyRawCountData(cap=0.99)

trainz = niceify.fit_transform(train)

print("Dimensions of full dataset: ", train.shape)
print("Dimensions of dataset no zeros: ", trainz.shape)

trainz_path = (dataset_path / (run_id + '_train-none_none.csv')).as_posix()
trainz.to_csv(trainz_path)

# Save feature-wise medians for imputing
with open(model_path / (run_id + '-raw_medians.json'), "w") as handle:
    json.dump(trainz.median().astype(int).to_json(), handle)

del train

if args.test_data:
    testz = {}
    testz_path = {}
    for t in args.test_data:
        testz[t] = niceify.transform(test_data[t])

        testz_path[t] = (dataset_path / (run_id + '_' + Path(t).stem + '-none_none.csv')).as_posix()
        testz[t].to_csv(testz_path[t])
del test_data

### Plot niceified data
if not args.plots[0] == "False":
    if args.plots[0] == "True":
        pp.save_distributions(trainz,
                              outdir=qc_path,
                              prefix='train-none_none',
                              random_state=seed)
    else:
        pp.save_distributions(trainz,
                              outdir=qc_path,
                              prefix='train-none_none',
                              genes=args.plots,
                              random_state=seed)
print('None preprocessing none scaling complete.')

### Scale niceified data
if 'Global' in args.scaling_method:
    global_scaler = pp.GlobalMinMaxScaler()

    trainz_gm = global_scaler.fit_transform(trainz)
    trainz_gm.to_csv(dataset_path / (run_id + '_train-none_global.csv'))

    ## Save model
    sklearn_to_json(global_scaler, (model_path / (run_id + '-none_global_model.json')))
    # with open((model_path / (run_id + '-none_global_model.p')), "wb") as handle:
    #     pickle.dump(global_scaler, handle, protocol=4)

    if args.test_data:
        for t in args.test_data:
            testz_gm = None
            testz_gm = global_scaler.transform(testz[t])
            testz_gm.to_csv(dataset_path / (run_id + '_' + Path(t).stem + '-none_global.csv'))
        del testz_gm

    if not args.plots[0] == 'False':
        if args.plots[0] == 'True':
            pp.save_distributions(trainz_gm,
                                  outdir=qc_path,
                                  prefix='train-none_global',
                                  random_state=seed)
        else:
            pp.save_distributions(trainz_gm,
                                  outdir=qc_path,
                                  prefix='train-none_global',
                                  genes=args.plots,
                                  random_state=seed)
    del trainz_gm, global_scaler

    print('None preprocessing global scaling complete.')

if 'Feature' in args.scaling_method:
    feature_scaler = pp.MinMaxScaler()

    trainz_sm = pd.DataFrame(feature_scaler.fit_transform(trainz),
                             index=trainz.index,
                             columns=trainz.columns)

    trainz_sm.to_csv(dataset_path / (run_id + '_train-none_feature.csv'))

    ## Save model
    sklearn_to_json(feature_scaler, (model_path / (run_id + '-none_feature_model.json')))
    # with open((model_path / (run_id + '-none_feature_model.p')), "wb") as handle:
    #     pickle.dump(feature_scaler, handle, protocol=4)

    if args.test_data:
        for t in args.test_data:
            testz_sm = None
            testz_sm = pd.DataFrame(feature_scaler.transform(testz[t]),
                                    index=testz[t].index,
                                    columns=testz[t].columns)
            testz_sm.to_csv(dataset_path / (run_id + '_' + Path(t).stem + '-none_feature.csv'))
        del testz_sm

    if not args.plots[0] == 'False':
        if args.plots[0] == 'True':
            pp.save_distributions(trainz_sm,
                                  outdir=qc_path,
                                  prefix='train-none_feature',
                                  random_state=seed)
        else:
            pp.save_distributions(trainz_sm,
                                  outdir=qc_path,
                                  prefix='train-none_feature',
                                  genes=args.plots,
                                  random_state=seed)
    del trainz_sm, feature_scaler

    print('None preprocessing feature scaling complete.')

## Run all preprocessing methods
prep_method = [value for value in args.preprocessing_method if value != "none"]

for p in prep_method:
    
    if p == 'LS':
        prep = pp.LSPreprocessing()

        train_prep = prep.fit_transform(trainz)
        train_prep.to_csv(dataset_path / (run_id + '_train-' + p + '_none.csv'))

        ## Save model
        sklearn_to_json(prep, (model_path / (run_id + '-LS_none_model.json')))
        # with open((model_path / (run_id + '-LS_none_model.p')), "wb") as handle:
        #     pickle.dump(prep, handle, protocol=4)

        if args.test_data:
            test_prep = {}
            for t in args.test_data:
                test_prep[t] = prep.transform(testz[t])
                test_prep[t].to_csv(dataset_path / (run_id + '_' + Path(t).stem + '-' + p + '_none.csv'))
    
    elif p == 'TPM':
        prep = pp.TPMPreprocessing(genelength=gene_length)

        train_prep = prep.fit_transform(trainz)
        train_prep.to_csv(dataset_path / (run_id + '_train-' + p + '_none.csv'))

        ## Save model
        sklearn_to_json(prep, (model_path / (run_id + '-TPM_none_model.json')))
        # with open((model_path / (run_id + '-TPM_none_model.p')), "wb") as handle:
        #     pickle.dump(prep, handle, protocol=4)

        if args.test_data:
            test_prep = {}
            for t in args.test_data:
                test_prep[t] = prep.transform(testz[t])
                test_prep[t].to_csv(dataset_path / (run_id + '_' + Path(t).stem + '-' + p + '_none.csv'))
                
    elif p == 'QT':
        prep = pp.QuantilePreprocessing()

        train_prep = prep.fit_transform(trainz)
        train_prep.to_csv(dataset_path / (run_id + '_train-' + p + '_none.csv'))

        ## Save model
        sklearn_to_json(prep, (model_path / (run_id + '-QT_none_model.json')))
        # with open((model_path / (run_id + '-QT_none_model.p')), "wb") as handle:
        #     pickle.dump(prep, handle, protocol=4)

        if args.test_data:
            test_prep = {}
            for t in args.test_data:
                test_prep[t] = prep.transform(testz[t])
                test_prep[t].to_csv(dataset_path / (run_id + '_' + Path(t).stem + '-' + p + '_none.csv'))
                
                
    elif p == 'RLE':
        prep = pp.RLEPreprocessing()

        train_prep = prep.fit_transform(trainz)
        train_prep.to_csv(dataset_path / (run_id + '_train-' + p + '_none.csv'))

        ## Save model
        sklearn_to_json(prep, (model_path / (run_id + '-RLE_none_model.json')))
        # with open((model_path / (run_id + '-RLE_none_model.p')), "wb") as handle:
        #     pickle.dump(prep, handle, protocol=4)

        if args.test_data:
            test_prep = {}
            for t in args.test_data:
                test_prep[t] = prep.transform(testz[t])
                test_prep[t].to_csv(dataset_path / (run_id + '_' + Path(t).stem + '-' + p + '_none.csv'))
                
    elif p == 'VST':
        
        if args.test_data:
            poi = trainz_path + '" -t "' + '" "'.join(
                testz_path.values())
        else:
            poi = trainz_path

        os.system((Path(__file__).parent.resolve().as_posix() + 
                   '/VST_preprocessing.R -d "' + poi +
                   '" -o "' + dataset_path.as_posix() + '" -r ' + run_id +
                   ' -p "' + model_path.as_posix() + '"' +
                   ' --datetime ' + run_id))

        train_prep = pd.read_csv(dataset_path / (run_id + '_train-' + p + '_none.csv'), index_col=0)

        if args.test_data:
            test_prep = {}
            for t in args.test_data:
                test_prep[t] = pd.read_csv(dataset_path / (run_id + '_' + Path(t).stem + '-' + p + '_none.csv'),
                                           index_col=0)
    
    elif p == 'GeVST':
        
        if args.test_data:
            poi = trainz_path + '" -t "' + '" "'.join(
                testz_path.values())
        else:
            poi = trainz_path

        os.system((Path(__file__).parent.resolve().as_posix()
                   + '/GeVST_preprocessing.R -d "' + poi
                   + '" -o "' + dataset_path.as_posix() + '" -r ' + run_id +
                   ' -p "' + model_path.as_posix()
                   + '" -g ' + args.gene_length +
                  ' --datetime ' + run_id))

        train_prep = pd.read_csv(dataset_path / (run_id + '_train-' + p + '_none.csv'), index_col=0)

        if args.test_data:
            test_prep = {}
            for t in args.test_data:
                test_prep[t] = pd.read_csv(dataset_path / (run_id + '_' + Path(t).stem + '-' + p + '_none.csv'),
                                           index_col=0)
                
    elif p == 'TMM':
        
        if args.test_data:
            poi = trainz_path + '" -t "' + '" "'.join(
                testz_path.values())
        else:
            poi = trainz_path

        os.system((Path(__file__).parent.resolve().as_posix() 
                   + '/TMM_preprocessing.R -d "' + poi
                   + '" -o "' + dataset_path.as_posix() + '" -r ' + run_id +
                   ' -p "' + model_path.as_posix() + '"' +
                  ' --datetime ' + run_id))

        train_prep = pd.read_csv(dataset_path / (run_id + '_train-' + p + '_none.csv'), index_col=0)

        if args.test_data:
            test_prep = {}
            for t in args.test_data:
                test_prep[t] = pd.read_csv(dataset_path / (run_id + '_' + Path(t).stem + '-' + p + '_none.csv'),
                                           index_col=0)
                
    elif p == 'GeTMM':
        
        if args.test_data:
            poi = trainz_path + '" -t "' + '" "'.join(
                testz_path.values())
        else:
            poi = trainz_path

        os.system((Path(__file__).parent.resolve().as_posix() 
                   + '/GeTMM_preprocessing.R -d "' + poi
                   + '" -o "' + dataset_path.as_posix() + '" -r ' + run_id +
                   ' -p "' + model_path.as_posix()
                   + '" -g ' + args.gene_length +
                  ' --datetime ' + run_id))

        train_prep = pd.read_csv(dataset_path / (run_id + '_train-' + p + '_none.csv'), index_col=0)

        if args.test_data:
            test_prep = {}
            for t in args.test_data:
                test_prep[t] = pd.read_csv(dataset_path / (run_id + '_' + Path(t).stem + '-' + p + '_none.csv'),
                                           index_col=0)
                
    elif p == 'none':
        
        train_prep = pd.read_csv(dataset_path / (run_id + '_train-none_none.csv'), index_col=0)

    ### Plot prep data
    if not args.plots[0] == 'False':
        if args.plots[0] == 'True':
            try:
                pp.save_distributions(train_prep,
                                      outdir=qc_path,
                                      prefix=('train-' + p + '_none'),
                                      random_state=seed)
            except:
                warnings.warn("Unable to plot all distributions for " + p)
        else:
            try:
                pp.save_distributions(train_prep,
                                      outdir=qc_path,
                                      prefix=('train-' + p + '_none'),
                                      genes=args.plots,
                                      random_state=seed)
            except:
                warnings.warn("Unable to plot all distributions for " + p)
    print(p + ' preprocessing none scaling complete.')

    ### Scale prep data
    infinite_train = train_prep.isin([np.inf, -np.inf]).values.sum()

    if not infinite_train:
        if 'Global' in args.scaling_method:
            global_scaler = pp.GlobalMinMaxScaler()

            train_prep_gm = global_scaler.fit_transform(train_prep)
            train_prep_gm.to_csv(dataset_path / (run_id + '_train-' + p + '_global.csv'))

            ## Save model
            sklearn_to_json(global_scaler, (model_path / (run_id + '-' + p + '_global_model.json')))
            # with open((model_path / (run_id + '-' + p + '_global_model.p')), "wb") as handle:
            #     pickle.dump(global_scaler, handle, protocol=4)

            if args.test_data:
                for t in args.test_data:
                    test_prep_gm = None
                    test_prep_gm = global_scaler.transform(test_prep[t])
                    test_prep_gm.to_csv(dataset_path / (run_id + '_' + Path(t).stem + '-' + p + '_global.csv'))
                del test_prep_gm

            if not args.plots[0] == 'False':
                if args.plots[0] == 'True':
                    try:
                        pp.save_distributions(train_prep_gm,
                                              outdir=qc_path,
                                              prefix=('train-' + p + '_global'),
                                              random_state=seed)
                    except:
                        warnings.warn("Unable to plot all distributions for " + p + " global scaled data.")
                else:
                    try:
                        pp.save_distributions(train_prep_gm,
                                              outdir=qc_path,
                                              prefix=('train-' + p + '_global'),
                                              genes=args.plots,
                                              random_state=seed)
                    except:
                        warnings.warn("Unable to plot all distributions for " + p + " global scaled data.")
            del train_prep_gm, global_scaler

            print(p + ' preprocessing global scaling complete.')

        if 'Feature' in args.scaling_method:
            feature_scaler = pp.MinMaxScaler()

            train_prep_sm = pd.DataFrame(feature_scaler.fit_transform(train_prep),
                                         index=train_prep.index,
                                         columns=train_prep.columns)

            train_prep_sm.to_csv(dataset_path / (run_id + '_train-' + p + '_feature.csv'))

            ## Save model
            sklearn_to_json(feature_scaler, (model_path / (run_id + '-' + p + '_feature_model.json')))
            # with open((model_path / (run_id + '-' + p + '_feature_model.p')), "wb") as handle:
            #     pickle.dump(feature_scaler, handle, protocol=4)

            if args.test_data:
                for t in args.test_data:
                    test_prep_sm = None
                    test_prep_sm = pd.DataFrame(feature_scaler.transform(test_prep[t]),
                                                index=test_prep[t].index,
                                                columns=test_prep[t].columns)
                    test_prep_sm.to_csv(dataset_path / (run_id + '_' + Path(t).stem + '-' + p + '_feature.csv'))
                del test_prep_sm, test_prep

            if not args.plots[0] == 'False':
                if args.plots[0] == 'True':
                    try:
                        pp.save_distributions(train_prep_sm,
                                              outdir=qc_path,
                                              prefix=('train-' + p + '_feature'),
                                              random_state=seed)
                    except:
                        warnings.warn("Unable to plot all distributions for " + p + " feature scaled data.")
                else:
                    try:
                        pp.save_distributions(train_prep_sm,
                                              outdir=qc_path,
                                              prefix='train-' + p + '_feature',
                                              genes=args.plots,
                                              random_state=seed)
                    except:
                        warnings.warn("Unable to plot all distributions for " + p + " feature scaled data.")
            del train_prep_sm, feature_scaler
            print(p + ' preprocessing feature scaling complete.')

    else:
        warnings.warn("Infinite values in train data. Unable to fit min-max scaling for " + p)
