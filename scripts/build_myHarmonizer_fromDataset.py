import os
import argparse
import time
from pathlib import Path

parser = argparse.ArgumentParser(
    description='This script builds a harmonizer object from train, and validation datasets. '
                'It integrates the normalize_scale, autoencoder_optimization, and build_myHarmonizer '
                'script as a convenience function for the user.')
parser.add_argument('-d', '--train_data', type=str, required=True,
                    help='Path to training data csv. Samples as rows and features as columns. Required.')
parser.add_argument('-v', '--valid_data', type=str, required=True,
                    help='Path to validation data csv. Samples as rows and features as columns. Required.')
parser.add_argument('-t', '--test_data', type=str, nargs="*",
                    help='Path to validation data csv. Samples as rows and features as columns. Required.')
parser.add_argument('-m', '--meta', type=str,
                    help='Path to metadata csv. Named samples as first row and characteristic metadata as columns.')
parser.add_argument('-o', '--output_directory', type=str,
                    help='Directory to which output will be written', default='output')
parser.add_argument('-g', '--gene_length', type=str,
                    help='Path to gene length csv. Necessary only if wish to run all, GeTMM, or GeVST preprocessing.'
                         ' Default is effective gene lengths from an outside sample mapped by Kallisto.',
                    default='supporting/genelength.csv')
parser.add_argument('-p', '--preprocessing_method', type=str,
                    help='One preprocessing method to run. Defaults to QT. '
                         'For all methods, values above the 99th percentile (sample-wise) are capped, a pseudo count of 1'
                         'is added, and gene features with 0 expression across the train dataset are removed.'
                         ' Options are: none, LS, TPM, QT, RLE, VST, GeVST, TMM, GeTMM',
                    default='QT')
parser.add_argument('-s', '--scaling_method', type=str,
                    help='One scaling methods to run. Defaults to feature scaling method.'
                         ' Options are: Global, Feature',
                    default='Feature')
parser.add_argument('--plots', type=str, nargs="+",
                    help='Plot random gene-wise and feature-wise distributions. Options are True for random list of 5 genes,'
                         ' False for no plots, or a list of genes of interest separated by spaces. Will fail if genes'
                         ' are not in dataset features or gene length file (if GeTMM or GeVST is run).',
                    default=['False'])
parser.add_argument('--min_budget', type=float, help='Min number of epochs for training', default=100)
parser.add_argument('--max_budget', type=float, help='Max number of epochs for training', default=2000)
parser.add_argument('--n_iterations', type=int, help='Number of iterations performed by the optimizer', default=20)
parser.add_argument('--scheduler', type=str, help='Learning rate scheduler. One of 1cycle, 1cycle2, exponential, power',
                    default="power")

args = parser.parse_args()

print(
    "\033[1;32m Depending on the autoencoder optimization settings, this code may take a long time to run. \033[1;00m")

# Get datetime for runs
dt = time.strftime("%Y_%m_%d-%H_%M_%S")
print("myHarmonizer run: myHarmonizer-" + dt)

if args.test_data:
    dataset = ('" -t "' +
           args.valid_data + '" "' + '" "'.join(args.test_data))
else:
    dataset = ('" -t "' + args.valid_data)

# Run normalize_scale3

if args.output_directory == "output":
    args.output_directory = (Path.cwd() / 'output').as_posix()
    
if args.gene_length == "supporting/genelength.csv":
    args.gene_length == (Path(__file__).parents[1] / 'supporting' / 'genelength.csv').as_posix()

os.system(('python ' + Path(__file__).parent.resolve().as_posix() + '/normalize_scale.py -d "' +
           args.train_data + dataset + '" -o "' +
           args.output_directory + '" -g ' +
           args.gene_length + ' -p ' +
           args.preprocessing_method + ' -s ' +
           args.scaling_method + ' --plots ' +
           " ".join(args.plots) + ' --datetime ' +
           dt))

# Run autoencoder_optimization

normalized_train_data = (Path(args.output_directory) / "Data Representations" / 'Normalized' / ('preprocess_' + dt) /
                         ('preprocess_' + dt + '_train-' + args.preprocessing_method + "_" + args.scaling_method.lower()
                          + ".csv")).as_posix()
normalized_valid_data = (Path(args.output_directory) / "Data Representations" / 'Normalized' / ('preprocess_' + dt) /
                         ('preprocess_' + dt + '_valid-' + args.preprocessing_method + "_" + args.scaling_method .lower()
                          + ".csv")).as_posix()

if args.test_data:
    normalized_test_data = {}
    for t in args.test_data:
        normalized_test_data[t] = (
                Path(args.output_directory) / "Data Representations" / 'Normalized' / ('preprocess_' + dt) /
                ('preprocess_' + dt + '_' + Path(t).stem + '-' +
                 args.preprocessing_method + "_" + args.scaling_method.lower() + ".csv")).as_posix()

    dataset1 = ('" -v "' + normalized_valid_data +
               '" -t "' + '" "'.join(normalized_test_data.values()))
else:
    dataset1 = ('" -v "' + normalized_valid_data)

os.system(('python ' + Path(__file__).parent.resolve().as_posix() + '/autoencoder_optimization.py -d "' +
           normalized_train_data + dataset1 + '" -o "' +
           args.output_directory + '" --min_budget ' +
           str(args.min_budget) + ' --max_budget ' +
           str(args.max_budget) + ' --n_iterations ' +
           str(args.n_iterations) + ' --scheduler ' +
           args.scheduler + ' --datetime ' +
           dt))

# Run build_myHarmonizer

if args.preprocessing_method in ['LS', 'TPM', 'QT', 'RLE']:
    preprocessing_path = (Path(args.output_directory) / 'Raw Python Package' / 'Normalized' / ('preprocess_' + dt) /
                          ('preprocess_' + dt + '-' + args.preprocessing_method + '_none_model.json')).as_posix()

elif args.preprocessing_method in ['VST', 'GeVST', 'TMM', 'GeTMM']:
    preprocessing_path = (Path(args.output_directory) / 'Raw Python Package' / 'Normalized' / ('preprocess_' + dt) /
                          (args.preprocessing_method + "_parameters.txt")).as_posix()

elif args.preprocessing_method == 'none':
    preprocessing_path = (Path(args.output_directory) / 'Raw Python Package' / 'Normalized' / ('preprocess_' + dt) /
                          ("none.none")).as_posix()

else:
    raise KeyError(args.preprocessing_method + ' is not one of none, LS, TPM, QT, RLE, VST, GeVST, TMM, GeTMM')


latent_train_data = (Path(args.output_directory) / "Data Representations" / 'Autoencoder' / ('autoencoder_' + dt) /
                         ('autoencoder_' + dt + '-train.csv')).as_posix()
latent_valid_data = (Path(args.output_directory) / "Data Representations" / 'Autoencoder' / ('autoencoder_' + dt) /
                         ('autoencoder_' + dt + '-valid.csv')).as_posix()

if args.test_data:
    latent_test_data = {}
    for t in args.test_data:
        latent_test_data[t] = (
                    Path(args.output_directory) / "Data Representations" / 'Autoencoder' / ('autoencoder_' + dt) /
                    ('autoencoder_' + dt + '-' + Path(t).stem + '.csv')).as_posix()

    dataset2 = ('" "' + latent_valid_data +
               '" "' + '" "'.join(latent_test_data.values()))
else:
    dataset2 = ('" "' + latent_valid_data)



if args.meta:
    os.system(('python ' + Path(__file__).parent.resolve().as_posix() + '/build_myHarmonizer.py -a "' +
               (Path(args.output_directory) / 'Raw Python Package' / 'Autoencoder' / (
                           'autoencoder_' + dt)).as_posix() + '" -p "' +
               preprocessing_path + '" -s "' +
               (Path(args.output_directory) / 'Raw Python Package' / 'Normalized' / ('preprocess_' + dt) /
               ('preprocess_' + dt + '-' + args.preprocessing_method + '_' + args.scaling_method.lower() + '_model.json')).as_posix() +
               '" -m "' + args.meta + '" --datetime ' +
               dt + ' -d "' + latent_train_data + dataset2 + '"'
               ))
else:
    os.system(('python ' + Path(__file__).parent.resolve().as_posix() + '/build_myHarmonizer.py -a "' +
               (Path(args.output_directory) / 'Raw Python Package' / 'Autoencoder' / (
                           'autoencoder_' + dt)).as_posix() + '" -p "' +
               preprocessing_path + '" -s "' +
               (Path(args.output_directory) / 'Raw Python Package' / 'Normalized' / ('preprocess_' + dt) /
                ('preprocess_' + dt + '-' + args.preprocessing_method + '_' + args.scaling_method.lower() + '_model.json')).as_posix() +
               '" --datetime ' +
               dt + ' -d "' + latent_train_data + dataset2 + '"'))

print("\n \033[1;32m myHarmonizer run: myHarmonizer-" + dt)
print("")
