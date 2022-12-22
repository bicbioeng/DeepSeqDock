FROM continuumio/miniconda3

WORKDIR /app

COPY ./deepseqdock .

RUN conda env create -f DeepSeqDock.yml
RUN apt-get update
RUN apt-get -y install fastqc

# Make RUN commands use the new environment:
RUN echo "conda activate DeepSeqDock" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]
