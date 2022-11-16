import os
import argparse
import time
from pathlib import Path
import json
import re
import glob

parser = argparse.ArgumentParser(
    description='This script builds a harmonizer object from an autoencoder model. This script assumes that all '
                'files within the Raw Python Package retain their default names and locations within the output directory.')

parser.add_argument('-a', '--autoencoder', type=str, required=True,
                    help='Name of autoencoder run. e.g. autoencoder_2022_08_27-13_14_39')
parser.add_argument('-m', '--meta', type=str,
                    help='Path to metadata csv. Named samples as first row and characteristic metadata as columns.')
parser.add_argument('-o', '--output_directory', type=str,
                    help='Directory to which output will be written', default='output')

args = parser.parse_args()

if args.output_directory == 'output':
    args.output_directory = (Path.cwd() / 'output').as_posix()

# Get datetime for runs
dt = time.strftime("%Y_%m_%d-%H_%M_%S")
print("myHarmonizer run: myHarmonizer-" + dt)


# Determine normalization, scaling method
with open((Path(args.output_directory) / "Raw Python Package" / "Autoencoder" / args.autoencoder /
          (args.autoencoder + '-encoder_input_args.json')), "r") as handle:
    encoder_input_args = json.load(handle)

try:
    split1 = encoder_input_args['train_data'].split('-')
    split2 = split1[-1].split('_')

    norm = split2[0]
    scale = Path(split2[1]).stem

    prep_run_id = re.search('preprocess_\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2}', encoder_input_args['train_data']).group(0)

except:
    raise NameError('Normalization and scaling methods could not be parsed from autoencoder metadata.')

# Identify normalization file location
if norm in ['LS', 'TPM', 'QT', 'RLE']:
    preprocessing_path =  (Path(args.output_directory) / 'Raw Python Package' / 'Normalized' / prep_run_id /
            (prep_run_id + '-' + norm + '_none_model.json')).as_posix()

elif norm in ['VST', 'GeVST', 'TMM', 'GeTMM']:
    preprocessing_path = (Path(args.output_directory) / 'Raw Python Package' / 'Normalized' / prep_run_id /
                          (norm + "_parameters.txt")).as_posix()

elif norm == 'none':
    preprocessing_path = (Path(args.output_directory) / 'Raw Python Package' / 'Normalized' / prep_run_id /
                          ("none.none")).as_posix()

else:
    raise KeyError(norm + ' is not one of none, LS, TPM, QT, RLE, VST, GeVST, TMM, GeTMM')

# Identify scaling file location

if scale in ['none', 'feature', 'global']:
    scaling_path = (Path(args.output_directory) / 'Raw Python Package' / 'Normalized' / prep_run_id /
            (prep_run_id + '-' + norm + '_' + scale + '_model.json')).as_posix()
else:
    raise KeyError(scale + ' is not one of none, feature, or global')

# Identify dataset locations
doi = glob.glob((Path(args.output_directory) / 'Data Representations' / 'Autoencoder' /
                 args.autoencoder / (args.autoencoder + "*.csv")).as_posix())
if args.meta:
    os.system(('python ' + Path(__file__).parent.resolve().as_posix() + '/build_myHarmonizer.py -a "' +
               (Path(args.output_directory) / 'Raw Python Package' / 'Autoencoder' / args.autoencoder).as_posix() + '" -p "' +
               preprocessing_path + '" -s "' +
               scaling_path +
               '" -m "' + args.meta + '" --datetime ' +
                dt + ' -d "' +
               '" "'.join(doi) + '" -o "' + args.output_directory + '"'))
else:
    os.system(('python ' + Path(__file__).parent.resolve().as_posix() + '/build_myHarmonizer.py -a "' +
               (Path(args.output_directory) / 'Raw Python Package' / 'Autoencoder' / args.autoencoder).as_posix() + '" -p "' +
               preprocessing_path + '" -s "' +
               scaling_path + '" --datetime ' +
                dt + ' -d "' +
               '" "'.join(doi) + '" -o "' + args.output_directory + '"'))

print("\n \033[1;32m myHarmonizer run: myHarmonizer-" + dt)
print("")






