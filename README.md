# DeepSeqDock: Framework for building optimized autoencoders for RNA sequencing data harmonization.

---

## **Table of Contents**
1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick start](#quick-start)
4. [Input and Output Formats](#input-and-output-formats)
5. [Example Workflow](#example-workflow)
6. [Citations and Licensing](#citations-and-licensing)

---

## Overview

The advent of high-throughput RNA-sequencing has led to the development of vast databases of sample transcriptomic data. A number of existing projects have sought to bring this data to a more unified representation to facilitate harmonization of multi-institutional RNA-sequencing datasets by unified quality control and mapping pipelines. The output of these efforts is useful for such downstream applications as machine learning and similarity assessment between samples. 

The DeepSeqDock framework takes this approach one step further by providing a unified framework for developing, running, and assessing the meaningfulness of data representations with regards to discrete biological metadata. This harmonization framework incorporates multiple frozen methods for normalization and scaling. Additionally, there is a module for optimizing and building a type of unsupervised deep learning model known as the autoencoder, which has proven useful for denoising tasks in a number of domains. DeepSeqDock also provides a module for evaluation of the different data representations with regards to categorical sample metadata, so that the efficacy of the preprocessing and encoding steps can be quantitatively assessed. 

As a convenience, this framework also provides a local implementation of the pipeline used to preprocess the [ArchS4](https://maayanlab.cloud/archs4/) dataset, to facilitate integration of user datasets with sequence read archive (SRA) data from the ARCHS4 dataset. Individual datasets can also be aligned using the [Elysium](https://maayanlab.cloud/cloudalignment/elysium.html) tool from the Ma'ayan lab for smaller datasets for integration with ARCHS4 data.

The input for this framework is uniformly aligned bulk RNA-sequencing count data matrices. These matrices may be subsets of massive online datasets such as the ARCHS4 dataset, GREIN, or TCGA.

The output from the DeepSeqDock framework include 1) a myHarmonizer json object that can be fed into the myHarmonizer python package or web application to allow for the preprocessing and encoding of additional datasets, 2) quality control and metadata from each module in the framework, and 3) a unified representation of the dataset after each of normalization, scaling, and encoding that can be used for additional, downstream applications. 

In short, the DeepSeqDock framework:

1) Uniformly aligns RNA-seq data according to the ARCHS4 pipeline
2) Normalizes and scales data
3) Optimizes and builds an autoencoder to transform normalized data
4) Evaluates dataset representations (e.g. before and after autoencoder) relative to categorical sample metadata (e.g. biological condition)
5) Builds a pipeline for transforming new data into the condensed representation of the training data / knowledge base to evaluate similarity between new data and knowledge base data (myHarmonizer object)

<img src="https://raw.githubusercontent.com/bicbioeng/DeepSeqDock/main/images/Fig1_web.png?raw=true" alt="Fig1" width="800"/>

As with all deep learning approaches, the more clean data available, the stronger the model will be. The framework was tested on knowledge bases of hundreds to over a thousand datasets. This approach has not been validated for small knowledge bases (less than 100 samples). 

## Installation

### Docker

The most straightforward implemenatation of DeepSeqDock is the Dockerized version. First, make sure docker is installed. Next, download the DeepSeqDock image from: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10119352.svg)](https://doi.org/10.5281/zenodo.10119352)

Load the docker image:

```
## WSL or Linux
docker load < deepseqdock.tar.gz

## Windows
docker load -i .\deepseqdock.tar.gz
```

then run the container interactively:

```
## WSL or Linux
docker run -it --rm -v "$HOME"/deepseqdockoutput:/app/output deepseqdock

## Windows
docker run -it –rm -v “c:\user\username\deepseqdockoutput:/app/output” deepseqdock
```

Parameters:
 - -it: interactive mode
 - --rm: automatic clean container when container stop
 - --v: bind container volume to host folder. Host folder will by default be in deepseqdockoutput folder of WSL2 or Linux home folder.

If these defaults are kept, the output directory in the Docker container will be mirrored by the deepseqdockoutput folder in the home directory of the user. User data can be placed in this directory to be accessed by the running Docker container. 

### Conda environment for Linux/WSL

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

## Input and Output Formats

### Supported Input Formats
- FASTA
- FASTQ
- CSV

### Output Formats
- Processed data: CSV, JSON

### Train/Validation/Test Data Schema
The train, validation, and test datasets should adhere to the following schema:

| Column Name     | Data Type   | Description                                 |
|------------------|-------------|---------------------------------------------|
| Empty     | String      | Unique identifier for each sample.          |
| `gene_id_1`     | Int       | Value for the gene 1 expression count.                |
| `gene_id_2`     | Int       | Value for the gene 2 expression count.               |
| ...             | ...         | Additional gene expression count as columns.             |

Note:
 - Each row corresponds to a single sample.
 - The feature columns (feature_1, feature_2, etc.) correspond to Ensembl gene IDs, which use the format ENSG********** (e.g., ENSG00000123456). Each column holds the integer value (e.g., expression count) for that specific gene.

### Metadata Data Schema
The metadata file should contain the following structure, the data type of each characteristic is catergory:

| Column Name     | Data Type   | Description                                 |
|------------------|-------------|---------------------------------------------|
| Empty     | String      | Unique identifier for each sample.          |
| `characteristic_1` | String   | Metadata characteristic (e.g., condition).  |
| `characteristic_2` | String   | Additional metadata (e.g., source, group).  |
| ...             | ...         | Additional features as columns.             |

The first column names the samples, and subsequent columns provide associated metadata.

## Quick start

This function will run normalization, scaling, and autoencoder optimization steps and assemble a myHarmonizer object from the result. After uniform aligment (e.g. ARCHS4 pipeline), count data matrices should be arranged as csv files with samples as rows and gene features as columns. To fully test the harmonization workflow, validation and test datasets should be split from the training dataset before normalization and scaling. When dealing with large multi-institutional datasets, it is recommended that validation (test) datasets contain samples from unique origins, when appropriate. A sample metadata file with samples as rows and categorical features as columns may also be supplied. Default normalization is QT and scaling is feature min-max scaling.

If the user does not supply their own gene length file and intends to use one of the normalization methods that depends on gene length, only the HGNC official gene symbol may be used to denote features. Other feature IDs will not map well to the gene length file. 

Help documentation is available at:

```shell
python scripts/build_myHarmonizer_fromDataset.py --help
```

To run the workflow with a toy dataset (and toy autoencoder optimization settings):

```shell
python scripts/build_myHarmonizer_fromDataset.py -d supporting/train.csv -v supporting/valid.csv -m supporting/meta.csv --min_budget 10 --max_budget 100 --n_iterations 2
```
### **Arguments**
| Argument        | Description                                    | Default         |
|------------------|------------------------------------------------|-----------------|
| `--train_data`/`-d`       | Path to training data csv. Required.           | None (required) |
| `--valid_data`/`-v`      | Path to validation data csv. Required.         | None (required)  |
| `--test_data`/`-t`     | Path to validation data csv. Required.      | None (required)       |
| `--meta`/`-m`      | Path to metadata csv. Required if running feature selection.      | None            |
| `--output_directory`/`o`       | Directory to which output will be written           | `output` |
| `--min_budget`      | Min number of epochs for training         | `100`  |
| `--max_budget`     | Max number of epochs for training      | `2000`       |
| `--n_iterations`      | Number of iterations performed by the optimizer      | `20`            |

## Example Workflow
### 1) Local ARCHS4

To run the local ARCHS4 pipeline, FASTQ files for each sample should be placed inside of unique folders. It is assumed that paired-end reads will have two FASTQ files and single-end reads will have a single FASTQ file. All sample folders should be kept inside one folder (e.g. sampledir) that will be the input for the script. If dumping FASTQ files from the SRA, it is recommended that the flags --concatenate-reads --include-technical are included since the original ARCHS4 pipeline was run with the fastq-dump utility instead of fasterq-dump (see bottom of [fasterq-dump documentation](https://github.com/ncbi/sra-tools/wiki/HowTo:-fasterq-dump).

Help documentation is available as:

```shell
scripts/local_archs4.sh --help
```

Sample FASTQ files have been provided in the 'supporting' folder for testing. For the first run, setting the -i argument to true is necessary to download the Kallisto index file locally. After the first run, this human_index.idx file should already be available. 

```shell
scripts/local_archs4.sh -d supporting/fastq -i true
```

### 2) Normalization and scaling

The purpose of this module is to normalize and scale input data, and provide frozen parameters for these preprocessing methods so that they can be reproduced afterwards. After uniform aligment (e.g. ARCHS4 pipeline), count data matrices should be arranged as csv files with samples as rows and gene features as columns. To fully test the harmonization workflow, validation (and test) dataset(s) should be split from the training dataset before normalization and scaling. When dealing with large multi-institutional datasets, it is recommended that validation (test) datasets contain samples from unique origins, when appropriate. 

For the normalization_scaling script, detailed descriptions of arguments are available as:

```shell
python scripts/normalize_scale.py --help
```

For normalization, it is possible to choose one or more of ten options (separated by spaces): all, none, LS, TPM, QT, RLE, VST, GeVST, TMM, GeTMM. Where LS is library scale, TPM is transcript per kilobase million, QT is quantile, RLE is relative log expression, TMM is trimmed mean of M values, GeTMM is gene length corrected trimmed mean of M values, VST is variance stabilizing transformation, and GeVST is gene length corrected variance stabilized transformation. The default value is QT. 

If chosing one of TPM, GeVST, or GeTMM, a csv file with the gene transcript length may be provided for the -g, --gene_length argument, otherwise, the default file will be used and features that are not mapped will be excluded. In this case, only the HGNC official gene symbol may be used to denote features. Other feature IDs will not map well to the gene length file. 

For scaling, it is possible to chose one of four options: all, none, Global, Feature. All scaling options except none are min-max scaling. The global or feature refers to whether the minimum and maximum values are defined by gene features (Feature) or across the dataset (Global). The default value is Feature.

Sample train, validation, and test csv files have been made available in the 'supporting' folder. Validation and test datasets are treated the same for this script and can both be entered under the -t, -test_data argument. Datetime need not be included for a typical run and will be set to the run datetime.

```shell
python scripts/normalize_scale.py -d supporting/train.csv -t supporting/test.csv supporting/valid.csv --datetime 1900_01_01-00_00_00
```

### 3) Optimize and build autoencoder

<span style="color:red">This module may run a long time (hours) with default parameters!</span>

The purpose of this module is to optimize a domain-specific autoencoder to bring data to a reduced, meaningful representaiton.  Because optimization is based on the loss of the validation datasets, both a train and validation dataset csv must be provided to build and optimize an autoencoder. Detailed descriptions of arguments are available as:

```shell
python scripts/autoencoder_optimization.py --help
```

Of these parameters, the most important hyperparameters are the --min_budget, --max_budget, and --n_iterations. In this context, the min and max budget refer to the number of epochs for each iteration of hyperband training. The defaults for these are the values that were found best for the RNA-seq datasets tested. In general, the min budget should be as small as possible while still retaining some indication of the loss on higher budget runs. The maximum budget was set as the number of epochs necessary to consistently have a plateau in the validation training curve. 

In general, the higher the n_iterations the better, because each iteration gives the Bayesian model more data. Because the model building and optimization have stochastic elements, each run will result in a different model with different optimized parameters.

The sample data in the supporting folder can again be used to test this function. When running actual data, it is not recommended to cut the min_budget, max_budget, and n_iterations this low. Datetime need not be included for a typical run and will be set to the run datetime.

```shell
python scripts/encoder_optimization.py -d \
"output/Data Representations/Normalized/preprocess_1900_01_01-00_00_00/preprocess_1900_01_01-00_00_00_train-QT_feature.csv" -v \
"output/Data Representations/Normalized/preprocess_1900_01_01-00_00_00/preprocess_1900_01_01-00_00_00_valid-QT_feature.csv" -t \ 
"output/Data Representations/Normalized/preprocess_1900_01_01-00_00_00/preprocess_1900_01_01-00_00_00_test-QT_feature.csv" -m \
supporting/meta.csv \
--min_budget 10 --max_budget 100 --n_iterations 2 --datetime 2000_01_01-00_00_00
```

### 4) Categorical Evaluation

This module is intended to use sample metadata to evaluate the meaningfulness of the data representation with regards to continuous similarity metrics and downstream classification machine learning models. To run this module, test (validation) csv(s) should be provided as well as a csv with samples as rows and columns as categorial (nominal) sample characteristics. Examples of this type of sample metadata may be disease state, tissue of origin, etc. Detailed descriptions of arguments are available as:

```shell
python scripts/categorical_evaluation.py -d supporting/train.csv -t supporting/test.csv -m supporting/meta.csv
```

### 5) Build myHarmonizer Object

This final module utilizes the outputs from normalization, scaling, and autoencoder transformations to build a myHarmonizer object, that can be input into the myHarmonizer python package or web application to transform new datasets that fall within the domain of the knowledge base datasets into the same condensed data representation as the output from the autoencoder. Assuming that the output directory tree has been kept intact during the running of the normalization and autoencoder modules, a convenience script has been written to automatically pull all of the necessary data and models based on the name of the autoencoder of interest (can be found at the end of the autoencoder optimization script). If output directory is not located in the current working directory, then the output folder should also be supplied.

```shell
python scripts/build_myHarmonizer_fromEncoder.py -e autoencoder_2000_01_01-00_00_00
```

If the directory tree has been altered or the names changed, then it is also possible to supply the paths to the necessary models individually. Please see documentation for more details.

```shell
python scripts/build_myHarmonizer.py --help
```
## Citations and Licensing

DeepSeqDock: a framework for representation learning and similarity evaluation of omics datasets. \
Copyright (C) 2024 bicbioeng

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.

This work includes functions modified from the calcNormFactors function in edgeR and the estimateSizeFactors function from DESeq2. Modified functions are marked and can be found in the GeTMM_preprocessing.R and GeVST_preprocessing.R files. Original code can be found under calcNormFactors_edgeR.R ([https://code.bioconductor.org/browse/edgeR/blob/RELEASE_3_16/R/calcNormFactors.R](https://code.bioconductor.org/browse/edgeR/blob/RELEASE_3_16/R/calcNormFactors.R)) or estimateSizeFactorsForMatrix_DESeq2.R ([https://code.bioconductor.org/browse/DESeq2/blob/RELEASE_3_12/R/core.R]( https://code.bioconductor.org/browse/DESeq2/blob/RELEASE_3_12/R/core.R)). DESeq2 is distributed under the [LGPL license (>=3)](https://www.gnu.org/licenses/lgpl-3.0.en.html) and edgeR is distributed under the [LGPL license (>=2)](https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html). A copy of LGPL 3.0 is also available in this repository. 

DESeq2 Citation:

Love, M.I., Huber, W., Anders, S. (2014) Moderated estimation of fold change and dispersion for RNA-seq data with DESeq2. Genome Biology, 15:550. [https://doi.org/10.1186/s13059-014-0550-8](https://doi.org/10.1186/s13059-014-0550-8)  

edgeR Citations:

  1) Robinson MD, McCarthy DJ and Smyth GK (2010). edgeR: a Bioconductor package for differential
  expression analysis of digital gene expression data. Bioinformatics 26, 139-140

  2) McCarthy DJ, Chen Y and Smyth GK (2012). Differential expression analysis of multifactor RNA-Seq
  experiments with respect to biological variation. Nucleic Acids Research 40, 4288-4297

  3) Chen Y, Lun ATL, Smyth GK (2016). From reads to genes to pathways: differential expression
  analysis of RNA-Seq experiments using Rsubread and the edgeR quasi-likelihood pipeline.
  F1000Research 5, 1438

  4) Chen Y, Chen L, Lun ATL, Baldoni PL, Smyth GK (2024). edgeR 4.0: powerful differential analysis
  of sequencing data with expanded functionality and improved support for small counts and larger
  datasets. bioRxiv doi: 10.1101/2024.01.21.576131
