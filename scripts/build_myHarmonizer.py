import os
import json
import argparse
import glob
import time
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='This script builds a harmonizer object')
parser.add_argument('-a', '--autoencoder', type=str, required=True,
                    help='Path to directory containing autoencoder and autoencoder run metadata.')
parser.add_argument('-p', '--preprocessing', type=str, required=True,
                    help='Path to normalization model json or txt. Default formatted to '
                         'preprocess_YYYY_mm_dd-HH_MM_SS_prepmethod_none_model.json. '
                         'e.g. preprocess_2022_01_01-01_01_00_00-QT_none_model.json for most preprocesing models, '
                         'but prepmethod_parameters.txt e.g. GeTMM_parameters.txt for VST, GeVST, TMM, GeTMM models')
parser.add_argument('-s', '--scaling', type=str, required=True,
                    help='Path to scaling model json. Default formatted to '
                         'preprocess_YYY_mm_dd_HH_MM_SS_prepmethod_scalingmethod_model.json. '
                         'e.g. preprocess_2022_01_01-01_00_00-QT_feature_model.json')
parser.add_argument('-d', '--data', type=str, nargs="*",
                    help='Path to dataset(s) after preprocessing and enocoding.')
parser.add_argument('-m', '--meta', type=str,
                    help='Path to sample metadata csv. Named samples as first row and discrete characteristic metadata '
                         'as columns.')
parser.add_argument('-o', '--output_directory', type=str,
                    help='Directory to which output will be written', default='output')
parser.add_argument('--datetime', type=str, help="Used internally to identify runs")

args = parser.parse_args()

# get runid
if args.datetime:
    run_id = ("myHarmonizer-" + args.datetime)
else:
    run_id = time.strftime("myHarmonizer_%Y_%m_%d-%H_%M_%S")

# Initialize object
myHarmonizerdict = {'myHarmonizer_version': '0.0.1',
                    'metadata': {},
                    'modelmeta': {},
                    'models': {},
                    'data': {}}

# import metadata

if args.meta:
    myHarmonizerdict['metadata'] = pd.read_csv(Path(args.meta), index_col=0).to_json()

# import normalization and scaling data and metadata
print(Path(args.preprocessing).parent)
moi = glob.glob((Path(args.preprocessing).parent / ('*-raw_medians.json')).as_posix())[0]
with open(moi, "r") as handle:
    myHarmonizerdict['models']['raw_medians'] = json.load(handle)

if Path(args.preprocessing).suffix == '.txt':
    with open(Path(args.preprocessing).as_posix()) as handle:
        myHarmonizerdict['models']['preprocessing'] = handle.read()

elif Path(args.preprocessing).suffix == '.json':
    with open(Path(args.preprocessing).as_posix(), "r") as handle:
        myHarmonizerdict['models']['preprocessing'] = json.load(handle)

elif Path(args.preprocessing).suffix == '.none':
    myHarmonizerdict['models']['preprocessing'] = None

with open(Path(args.scaling).as_posix(), "r") as handle:
    myHarmonizerdict['models']['scaling'] = json.load(handle)

try:
    foi = glob.glob((Path(args.preprocessing).parent / ('*-normalization_arguments.json')).as_posix())[0]
    with open(foi, "r") as handle:
        myHarmonizerdict['modelmeta']['normalize_scale'] = json.load(handle)
except:
    myHarmonizerdict['modelmeta']['normalize_scale'] = {}
    warnings.warn('No metadata found in folder with normalization model. No normalization or scaling metadata will '
                  'be included in myHarmonizer object.')



# import autoencoder data and metadata
for a in ['encoder_config',
          'encoder_input_args',
          'encoder_metrics']:
    foi = glob.glob((Path(args.autoencoder) / ('*-' + a + '.json')).as_posix())[0]
    with open(foi, "r") as handle:
        myHarmonizerdict['modelmeta'][a] = json.load(handle)


for a in ['encoder_model_architecture',
          'encoder_model_weights']:

    foi = glob.glob((Path(args.autoencoder) / ('*-' + a + '.json')).as_posix())[0]
    with open(foi, "r") as handle:
        myHarmonizerdict['models'][a] = json.load(handle)


# import datasets
if args.data:
    data = []
    for d in args.data:
        data.append(pd.read_csv(d, index_col=0).astype('float64').round(4))
    myHarmonizerdict['data'] = pd.concat(data).to_json()

# Save myHarmonizer as json
with open(Path(args.output_directory) / (run_id + '.json'), 'w') as handle:
    json.dump(myHarmonizerdict, handle)



