####################################################
#Consolidate all samples from kallisto folder
###################################################

# Import command line arguments
args <- commandArgs(TRUE)

# Get arguments
directory <- list.files(args[1], full.names=TRUE)
output <- args[2]

# Read in and aggregate samples
df.list <- lapply(directory, function(d){
  if(file.exists(file.path(d, 'gene_abundance.csv'))){
    read.csv(file.path(d, 'gene_abundance.csv'), row.names=1)
  }
})

df <- data.frame(df.list)

# Ceiling of count matrix
int.df <- ceiling(df)

# Names of samples
colnames(int.df) <- basename(directory)

# Write csv
if(!dir.exists(file.path(output, "Data Representations"))){
  dir.create(file.path(output, "Data Representations"))
}

if(!dir.exists(file.path(output, "Data Representations", "Archs4"))){
  dir.create(file.path(output, "Data Representations", "Archs4"))
}

write.csv(int.df, file.path(output, "Data Representations", "Archs4",
                            paste0("archs4_", format(Sys.time(), format="%Y_%m_%d-%H_%M_%S"), ".csv")))