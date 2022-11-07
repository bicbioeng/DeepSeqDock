# DeepSeqDock: Framework for building optimized autoencoders for RNA sequencing data harmonization.

## Overview

The advent of high-throughput RNA-sequencing has led to the development of vast databases of sample transcriptomic data. A number of existing projects have sought to bring this data to a more unified representation to facilitate harmonization of multi-institutional RNA-sequencing datasets by unified quality control and mapping pipelines. The output of these efforts is useful for such downstream applications as similarity assessment between samples and machine learning. 

The DeepSeqDock framework takes this approach one step further by providing a unified framework for developing, running, and assessing the meaningfulness of data representations with regards to discrete biological metadata. This harmonization framework incorporates multiple frozen methods for normalization and scaling. Additionally, there is a module for optimizing and building a type of unsupervised deep learning model known as the autoencoder, which has proven useful for denoising tasks in a number of domains.

DeepSeqDock also provides a module for evaluation of the different data representations with regards to categorical sample metadata, so that the efficacy of the preprocessing and encoding steps can be quantitatively assessed. 

The output from the DeepSeqDock framework include 1) a myHarmonizer json object that can be fed into the myHarmonizer python package or web application to allow for the preprocessing and encoding of additional datasets, 2) quality control and metadata from each module in the framework, and 3) a unified representation of the dataset after each of normalization, scaling, and encoding that can be used for additional, downstream applications. 

This framework also provides a local implementation of the pipeline used to preprocess the [ArchS4](https://maayanlab.cloud/archs4/) dataset, to facilitate integration of user knowledge bases with data from the one of the massive mining projects.


<img src="https://raw.githubusercontent.com/bicbioeng/DeepSeqDock/main/images/Fig1.png?raw=true" alt="Fig1" width="800"/>

As with all deep learning approaches, the more clean data available, the stronger the model will be. The framework was tested on knowledge bases of hundreds to over a thousand datasets. This approach has not been validated for small knowledge bases (less than 100 samples). 

## Installation

### Docker

The most straightforward implemenatation of DeepSeqDock is the Dockerized version
```
docker run -d --rm --name deepseqdock us-central1-docker.pkg.dev/nosi-usd-biofilm/nosi-usd-biofilm-arti/deepseqdock
```
Parameters:
 - -d: detach mode
 - --rm: automatic clean container when container stop
 - --name : specify container name

### Conda environment

While the Docker version of the code is the easiest implementation, this version may not utilize GPU resources, which are particularly valuable for the autoencoder optimization step. The framework has also been made available on GitHub with a standard conda environment .yml file, which can be used to create a conda environment with the packages necessary to run the DeepSeqDock framework. 

For this approach, first make sure that a conda package manager is [installed](https://docs.conda.io/en/latest/miniconda.html) and the DeepSeqDock GitHub repository has been [cloned](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) or [forked](https://docs.github.com/en/get-started/quickstart/fork-a-repo) locally if you would like to make changes. This approach has only been tested in Ubuntu environments, and may require changes to be functional in Mac or Windows OS. 

To install the DeepSeqDock conda environment from the [terminal](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file):

```shell
conda env create -f DeepSeqDock.yml
```
and then activate the environment before running the modules.

```shell
conda activate DeepSeqDock
```
For Ubuntu version 20.10 and above, it may be necessary to install libffi7_3. The dependency chain with R does not allow for updated versions of python to be used at the time of this writing.

## Quick start

This function will run normalization, scaling, and autoencoder optimization steps and assemble a myHarmonizer object from the result. After uniform aligment (e.g. ARCHS4 pipeline), count data matrices should be arranged as csv files with samples as rows and gene features as columns. To fully test the harmonization workflow, validation and test datasets should be split from the training dataset before normalization and scaling. When dealing with large multi-institutional datasets, it is recommended that validation (test) datasets contain samples from unique origins, when appropriate. A sample metadata file with samples as rows and categorical features as columns may also be supplied. Default normalization is QT and scaling is feature min-max scaling.

If the user does not supply their own gene length file and intends to use one of the normalization methods that depends on gene length, only the HGNC official gene symbol may be used to denote features. Other feature IDs will not map well to the gene length file. 

Help documentation is available at:

```shell
python DeepSeqDock/scripts/build_myHarmonizer_fromDataset.py --help
```

To run the workflow with a toy dataset (and toy autoencoder optimization settings):

```shell
python DeepSeqDock/scripts/build_myHarmonizer_fromDataset.py -d DeepSeqDock/supporting/train.csv -v DeepSeqDock/supporting/valid.csv -m DeepSeqDock/supporting/trainmeta.csv --min_budget 10 --max_budget 100 --n_iterations 2
```


## Local ARCHS4

To run the local ARCHS4 pipeline, FASTQ files for each sample should be placed inside of unique folders. It is assumed that paired-end reads will have two FASTQ files and single-end reads will have a single FASTQ file. All sample folders should be kept inside one folder (e.g. sampledir) that will be the input for the script. If dumping FASTQ files from the SRA, it is recommended that the flags --concatenate-reads --include-technical are included since the original ARCHS4 pipeline was run with the fastq-dump utility instead of fasterq-dump (see bottom of [fasterq-dump documentation](https://github.com/ncbi/sra-tools/wiki/HowTo:-fasterq-dump).

Help documentation is available as:

```shell
DeepSeqDock/scripts/local_archs4.sh --help
```

Sample FASTQ files have been provided in the 'supporting' folder for testing. For the first run, setting the -i argument to true is necessary to download the Kallisto index file locally. After the first run, this human_index.idx file should already be available. 

```shell
DeepSeqDock/scripts/local_archs4.sh -d DeepSeqDock/supporting/fastq -i true
```

## Normalization and scaling

The purpose of this module is to normalize and scale input data, and provide frozen parameters for these preprocessing methods so that they can be reproduced afterwards. After uniform aligment (e.g. ARCHS4 pipeline), count data matrices should be arranged as csv files with samples as rows and gene features as columns. To fully test the harmonization workflow, validation (and test) dataset(s) should be split from the training dataset before normalization and scaling. When dealing with large multi-institutional datasets, it is recommended that validation (test) datasets contain samples from unique origins, when appropriate. 

For the normalization_scaling script, detailed descriptions of arguments are available as:

```shell
python DeepSeqDock/scripts/normalize_scale.py --help
```

For normalization, it is possible to choose one or more of ten options (separated by spaces): all, none, LS, TPM, QT, RLE, VST, GeVST, TMM, GeTMM. Where LS is library scale, TPM is transcript per kilobase million, QT is quantile, RLE is relative log expression, TMM is trimmed mean of M values, GeTMM is gene length corrected trimmed mean of M values, VST is variance stabilizing transformation, and GeVST is gene length corrected variance stabilized transformation. The default value is QT. 

If chosing one of TPM, GeVST, or GeTMM, a csv file with the gene transcript length may be provided for the -g, --gene_length argument, otherwise, the default file will be used and features that are not mapped will be excluded. In this case, only the HGNC official gene symbol may be used to denote features. Other feature IDs will not map well to the gene length file. 

For scaling, it is possible to chose one of four options: all, none, Global, Feature. All scaling options except none are min-max scaling. The global or feature refers to whether the minimum and maximum values are defined by gene features (Feature) or across the dataset (Global). The default value is Feature.

Sample train, validation, and test csv files have been made available in the 'supporting' folder. Validation and test datasets are treated the same for this script and can both be entered under the -t, -test_data argument. Datetime need not be included for a typical run and will be set to the run datetime.

```shell
python DeepSeqDock/scripts/normalize_scale.py -d DeepSeqDock/supporting/train.csv -t DeepSeqDock/supporting/test.csv DeepSeqDock/supporting/valid.csv --datetime 1900_01_01-00_00_00
```

## Optimize and build autoencoder

<span style="color:red">This module may run a long time (hours) with default parameters!</span>

The purpose of this module is to optimize a domain-specific autoencoder to bring data to a reduced, meaningful representaiton.  Because optimization is based on the loss of the validation datasets, both a train and validation dataset csv must be provided to build and optimize an autoencoder. Detailed descriptions of arguments are available as:

```shell
python DeepSeqDock/scripts/autoencoder_optimization.py --help
```

Of these parameters, the most important hyperparameters are the --min_budget, --max_budget, and --n_iterations. In this context, the min and max budget refer to the number of epochs for each iteration of hyperband training. The defaults for these are the values that were found best for the RNA-seq datasets tested. In general, the min budget should be as small as possible while still retaining some indication of the loss on higher budget runs. The maximum budget was set as the number of epochs necessary to consistently have a plateau in the validation training curve. 

In general, the higher the n_iterations the better, because each iteration gives the Bayesian model more data. Because the model building and optimization have stochastic elements, each run will result in a different model with different optimized parameters.

The sample data in the supporting folder can again be used to test this function. When running actual data, it is not recommended to cut the min_budget, max_budget, and n_iterations this low. Datetime need not be included for a typical run and will be set to the run datetime.

```shell
python DeepSeqDock/scripts/autoencoder_optimization.py -d "output/Data Representations/Normalized/preprocess_1900_01_01-00_00_00/preprocess_1900_01_01-00_00_00_train-QT_feature.csv" -v "output/Data Representations/Normalized/preprocess_1900_01_01-00_00_00/preprocess_1900_01_01-00_00_00_valid-QT_feature.csv" -t "output/Data Representations/Normalized/preprocess_1900_01_01-00_00_00/preprocess_1900_01_01-00_00_00_test-QT_feature.csv" --min_budget 10 --max_budget 100 --n_iterations 2 --datetime 2000_01_01-00_00_00
```

## Categorical Evaluation

This module is intended to use sample metadata to evaluate the meaningfulness of the data representation with regards to continuous similarity metrics and downstream classification machine learning models. To run this module, test (validation) csv(s) should be provided as well as a csv with samples as rows and columns as categorial (nominal) sample characteristics. Examples of this type of sample metadata may be disease state, tissue of origin, etc. Detailed descriptions of arguments are available as:

```shell
python DeepSeqDock/scripts/categorical_evaluation.py -d DeepSeqDock/supporting/train.csv -t DeepSeqDock/supporting/test.csv -m DeepSeqDock/supporting/trainmeta.csv
```

## Build myHarmonizer Object

This final module utilizes the outputs from normalization, scaling, and autoencoder transformations to build a myHarmonizer object, that can be input into the myHarmonizer python package or web application to transform new datasets that fall within the domain of the knowledge base datasets into the same condensed data representation as the output from the autoencoder. Assuming that the output directory tree has been kept intact during the running of the normalization and autoencoder modules, a convenience script has been written to automatically pull all of the necessary data and models based on the name of the autoencoder of interest (can be found at the end of the autoencoder optimization script). If output directory is not located in the current working directory, then the output folder should also be supplied.

```shell
python DeepSeqDock/scripts/build_myHarmonizer_fromEncoder.py -a autoencoder_2000_01_01-00_00_00
```

If the directory tree has been altered or the names changed, then it is also possible to supply the paths to the necessary models individually. Please see documentation for more details.

```shell
python DeepSeqDock/scripts/build_myHarmonizer.py --help
``
