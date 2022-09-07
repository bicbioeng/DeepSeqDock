FROM jupyter/minimal-notebook:ubuntu-20.04

WORKDIR /app

USER root

RUN apt-get update && apt-get install -y --no-install-recommends build-essential r-base r-cran-randomforest

RUN apt-get install -y wget

RUN apt-get install fastqc

COPY GBMMC.yml /app/GBMMC.yml

RUN conda env create -f GBMMC.yml python=3.10

SHELL ["conda","run","-n","GBMMC","/bin/bash","-c"]

RUN python -m ipykernel install --name kernel_one

SHELL ["/bin/bash","-c"]

RUN conda init

RUN echo 'conda activate GBMMC' >> ~/.bashrc

#RUN apt-get install -y gdebi-core

#RUN wget https://download1.rstudio.org/desktop/bionic/amd64/rstudio-2022.07.1-554-amd64.deb

#RUN gdebi rstudio-2022.07.1-554-amd64.deb

#RUN Rscript -e "install.packages('plyr', repos='http://cran.us.r-project.org', dependencies = TRUE)"
#RUN Rscript -e "install.packages('BiocManager')"
#RUN Rscript -e "BiocManager::install('edgeR')"
#RUN Rscript -e "BiocManager::install('DESeq2')"
#RUN Rscript -e "install.packages('argparse')"

COPY . .
