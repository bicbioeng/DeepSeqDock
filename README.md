# DeepSeqDock: Framework for building optimized autoencoders for RNA sequencing data harmonization.

## Overview

The advent of high-throughput RNA-sequencing has led to the development of vast databases of sample transcriptomic data. A number of existing projects have sought to bring this data to a more unified representation to facilitate harmonization of multi-institutional RNA-sequencing datasets by unified quality control and mapping pipelines. The output of these efforts is useful for such downstream applications as similarity assessment between samples and machine learning. 

The DeepSeqDock framework takes this approach one step further by providing a unified framework for developing, running, and assessing the meaningfulness of data representations with regards to discrete biological metadata. This harmonization framework incorporates multiple frozen methods for normalization and scaling. Additionally, there is a module for optimizing and building a type of unsupervised deep learning model known as the autoencoder, which has proven useful for denoising tasks in a number of domains.

DeepSeqDock also provides a module for evaluation of the different data representations with regards to discrete sample metadata, so that the efficacy of the preprocessing and encoding steps can be quantitatively assessed. 

The output from the DeepSeqDock framework include 1) a myHarmonizer json object that can be fed into the myHarmonizer python package or web application to allow for the preprocessing and encoding of additional datasets, 2) quality control and metadata from each module in the framework, and 3) a unified representation of the dataset after each of normalization, scaling, and encoding that can be used for additional, downstream applications. 

This framework also provides a local implementation of the pipeline used to preprocess the [ArchS4](https://maayanlab.cloud/archs4/) dataset, to facilitate integration of user knowledge bases with data from the one of the massive mining projects.


<img src="https://raw.githubusercontent.com/bicbioeng/DeepSeqDock/main/images/Fig1.png?raw=true" alt="Fig1" width="800"/>

## Installation

The most straightforward implemenatation of DeepSeqDock is the Dockerized version, which is available at:

<span style="color:red">Docker container download and setup information here</span>

While the Docker version of the code is the easiest implementation, this version may not utilize GPU resources, which are particularly valuable for the autoencoder optimization step. The framework has also been made available on GitHub with a standard conda environment .yml file, which can be used to create a conda environment with the packages necessary to run the DeepSeqDock framework. 

For this approach, first make sure that a conda package manager is [installed](https://docs.conda.io/en/latest/miniconda.html) and the DeepSeqDock GitHub repository has been [cloned](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) or [forked](https://docs.github.com/en/get-started/quickstart/fork-a-repo) locally if you would like to make changes. This approach has only been tested in Linux environments, and may require changes to be functional in Mac or Windows OS. 

To install the conda environment from the [terminal](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file):

```shell
conda env create -f environment.yml
```
and then activate the environment before running the modules.

```shell
conda activate DeepSeqDock
```

<span style="color:red">Mention the .R scripts header here</span>
