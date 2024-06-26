#!/usr/bin/env Rscript

suppressPackageStartupMessages(library("argparse", warn.conflicts = FALSE, quietly = TRUE))
suppressPackageStartupMessages(library("DESeq2"))


# create parser object
parser <- ArgumentParser(description="Preprocess data using a frozen TMM algorithm.")

# specify arguments
parser$add_argument('-d', '--train_data', type='character',
                    help='Path to training data csv. Samples as rows and features as columns.')

parser$add_argument('-t', '--test_data', type='character', nargs='*',
                    help='List of paths to validation or test data csv(s).')

parser$add_argument('-o', '--output_directory', type='character', 
                    help='Directory to which output will be written')

parser$add_argument('-r', '--preprocessing_run', type='character',
                    help='Preprocessing run number.')

parser$add_argument('-p', '--params', type='character',
                    help='Path to training parameters for frozen normalization.')

parser$add_argument('-f', '--fit', type='character', default='True',
                    help="True or Filename. True indicates train data will be used to fit. Filename is the name of the text file with fitted frozen parameters.")

parser$add_argument('--datetime', type='character', default='False',
                    help="Internal identifier for preprocessing run.")

args <- parser$parse_args()

if(args$datetime == 'False'){
    run_id = paste0('preprocess_', format(Sys.time(), "%Y_%m_%d-%H_%M_%S"))
} else {
    run_id = args$datetime}

## Modified functions from DESeq2 #####################################################################################################
# Original functions available under estimateSizeFactorsForMatrix_DESeq2.R or at https://code.bioconductor.org/browse/DESeq2/blob/RELEASE_3_12/R/core.R
# Modified from DESeq2 Release_3_12 (19 Feb 2021)
# Citation: Love, M.I., Huber, W., Anders, S. (2014) Moderated estimation of fold change and dispersion for RNA-seq data with DESeq2. Genome Biology, 15:550. \url{https://doi.org/10.1186/s13059-014-0550-8}  
# Last modified on 24 November 2023.
estimateSizeFactorsForMatrix.modified <- function (counts, locfunc = stats::median, geoMeans, controlGenes, 
                                                   type = c("ratio", "poscounts")) 
{
  type <- match.arg(type, c("ratio", "poscounts"))
  if (missing(geoMeans)) {
    incomingGeoMeans <- FALSE
    if (type == "ratio") {
      loggeomeans <- rowMeans(log(counts))
    }
    else if (type == "poscounts") {
      lc <- log(counts)
      lc[!is.finite(lc)] <- 0
      loggeomeans <- rowMeans(lc)
      allZero <- rowSums(counts) == 0
      loggeomeans[allZero] <- -Inf
    }
  }
  else {
    incomingGeoMeans <- TRUE
    if (length(geoMeans) != nrow(counts)) {
      stop("geoMeans should be as long as the number of rows of counts")
    }
    loggeomeans <- log(geoMeans)
  }
  if (all(is.infinite(loggeomeans))) {
    stop("every gene contains at least one zero, cannot compute log geometric means")
  }
  sf <- if (missing(controlGenes)) {
    apply(counts, 2, function(cnts) {
      exp(locfunc((log(cnts) - loggeomeans)[is.finite(loggeomeans) & 
                                              cnts > 0]))
    })
  }
  else {
    if (!(is.numeric(controlGenes) | is.logical(controlGenes))) {
      stop("controlGenes should be either a numeric or logical vector")
    }
    loggeomeansSub <- loggeomeans[controlGenes]
    apply(counts[controlGenes, , drop = FALSE], 2, function(cnts) {
      exp(locfunc((log(cnts) - loggeomeansSub)[is.finite(loggeomeansSub) & 
                                                 cnts > 0]))
    })
  }
  ## Modified from DESeq2 Release_3_12 (19 Feb 2021).
  ## Last modified on 24 November 2023. Modification was to comment out two lines below so that size
  ## factors are not stabilized in the frozen representation (as this unfreezes the transformations).
  # if (incomingGeoMeans) {
  #   sf <- sf/exp(mean(log(sf)))
  # }
  sf
}
## End Modified functions from DESeq2 #################################################################################################

# 10,000 genes was chosen as a multiplier because only 35/35238 genes were higher than this
# (effective length in kb?). This value was chosen as a compromise so that the effect 
# of the ceiling function would not drastically limit the information in the dataset, 
# and also with the realization that the closer the data is to infinity, the more the
# vst functions as a log2 transformation. This seemed like a reasonable compromise 
# between the two extremes.

vst_preprocessing <- function(file, geneCorr = c('none', 'rpk'), dispersionList=NULL, output=getwd(), geneLengthFile=NULL){
  
  geneCorr <- match.arg(geneCorr)
  
  if(is.null(dispersionList)){
    switch(geneCorr,
           'none' = {data <- read.csv(file, row.names=1)
           prefix = 'VST'
           },
           'rpk' = {
             data_r <- read.csv(file, row.names=1)
             
             #import genelength
             genelength = read.csv(geneLengthFile)
             
             #preprocess with gene length
             data = data_r[,colnames(data_r) %in% genelength$gene]
             genelength_g = genelength$med[match(colnames(data), genelength$gene)]
             
             if(ncol(data) == length(genelength_g)){
               data = ceiling(sweep(data, 2, as.array(genelength_g), `/`)*10000)
             } else {stop("Mismatch between number of genes in sample and number of genes in genelength variable.")} 
             
             prefix = 'GeVST'
           })
    
    #get geometric means of genes (to freeze)
    gm <- exp(colMeans(log(data)))

    colData = data.frame('sample'=rownames(data))

    dds = DESeqDataSetFromMatrix(countData = t(data),
                                 colData = colData,
                                 design = ~1)
    dds <- estimateSizeFactors(dds)
    dds <- estimateDispersions(dds)
    
    vst = varianceStabilizingTransformation(dds, blind=FALSE) #False in order to use dispersion estimates from DESeq
    dispersion = dispersionFunction(dds)
    
    vst_m = t(assay(vst))
    
    write.csv(vst_m, file.path(output, paste0(run_id, "_train-", prefix, "_none.csv")))
    
    return(list('dispersion'=dispersion,
                'gm'=gm,
                'feature_names_in'=colnames(vst_m)))
    
    
  } else {
    switch(geneCorr,
           'none' = {
             data <- read.csv(file, row.names=1)
             prefix = 'VST'
           },
           'rpk' = {
             data_r <- read.csv(file, row.names=1)
             
             #import genelength
             genelength = read.csv(geneLengthFile)
             
             #preprocess with gene length
             data = data_r[,colnames(data_r) %in% genelength$gene]
             genelength_g = genelength$med[match(colnames(data), genelength$gene)]
             
             if(ncol(data) == length(genelength_g)){
               data = ceiling(sweep(data, 2, as.array(genelength_g), `/`)*10000)
             } else {stop("Mismatch between number of genes in sample and number of genes in genelength variable.")} 
             
             prefix = 'GeVST'
           })
    
    # #track and remove genes with zero expression
    # zeroLoc = (apply(data, 2, sum) == 0)
    # dataz = data[,!zeroLoc]
    
    colData = data.frame('sample'=rownames(data))
    dds = DESeqDataSetFromMatrix(countData = t(data),
                                 colData = colData,
                                 design = ~1)
    
    sizeFactors(dds) <- estimateSizeFactorsForMatrix.modified(counts(dds), geoMeans = dispersionList$gm)
    dispersionFunction(dds) <- dispersionList$dispersion
    
    ####WARNING!!! Size factors will no longer multiply to 1 for validation, test datasets
    
    vst = varianceStabilizingTransformation(dds, blind=FALSE) #False in order to use dispersion estimates from DESeq
    
    vst_mz = t(assay(vst))

    vstname = basename(file)

    vstname = sub("-[[:alpha:]]{2,4}_(none|global|feature)\\.csv", "", vstname)
    vstname = sub("preprocess_\\d{4}_\\d{2}_\\d{2}-\\d{2}_\\d{2}_\\d{2}_", "", vstname)

    # If not already preprocessed
    vstname = sub("\\.csv", "", vstname)

    write.csv(vst_mz, file.path(output, paste0(run_id, '_', vstname, "-", prefix, "_none.csv")))
  }
}

# Fit and transform datasets using training data, test data
if(args$fit == 'True'){
    dispersionList = vst_preprocessing(file=args$train_data, geneCorr = 'none', output = args$output_directory)
    suppress = sapply(args$test_data, function(t){
         vst_preprocessing(file=t, geneCorr = 'none', dispersionList = dispersionList, output = args$output_directory)})

	# Here args$params references a directory for output
	saveRDS(dispersionList, file = file.path(args$params, 'VST_parameters.txt'), ascii = TRUE, compress = FALSE)
}

# Only transform test data

if(!args$fit == 'True'){
    # Here args$params references a .rds file
    dispersionList <- readRDS(file = file.path(args$params, args$fit))

    suppress = sapply(args$test_data, function(t){
         vst_preprocessing(file=t, geneCorr = 'none', dispersionList = dispersionList, output = args$output_directory)})
}


