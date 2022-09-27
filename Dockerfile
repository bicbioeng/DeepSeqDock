FROM ubuntu:20.04

WORKDIR /app

USER root

RUN apt-get update

RUN apt-get install -y wget
# update indices
RUN apt update -qq
# install two helper packages we need
RUN apt install -y --no-install-recommends software-properties-common dirmngr
# add the signing key (by Michael Rutter) for these repos
# To verify key, run gpg --show-keys /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc 
# Fingerprint: E298A3A825C0D65DFD57CBB651716619E084DAB9
RUN wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | sudo tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc
# add the R 4.0 repo from CRAN -- adjust 'focal' to 'groovy' or 'bionic' as needed
RUN add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"

RUN apt-get -y install fastqc

COPY DeepSeqDock.yml /app/DeepSeqDock.yml

RUN conda env create -f DeepSeqDock.yml python=3.10

RUN python -m ipykernel install --user --name DeepSeqDock

SHELL ["/bin/bash","-c"]

RUN Rscript -e "install.packages('BiocManager', repos='http://cran.us.r-project.org')"
RUN Rscript -e "BiocManager::install('edgeR')"
RUN Rscript -e "BiocManager::install('DESeq2')"
RUN Rscript -e "install.packages('argparse',repos='http://cran.us.r-project.org')"
RUN Rscript -e "install.packages('plyr',repos='http://cran.us.r-project.org', dependencies = TRUE)"

COPY . .

ENTRYPOINT ["conda", "activate", "DeepSeqDock"]

