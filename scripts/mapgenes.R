####################################################
#Sum the ENST to Genes
###################################################

library('plyr')

#Import command line arguments
args <- commandArgs(TRUE)

#Load in mapping and abundance file
res = load(args[2])
f = file.path(args[1], 'abundance.tsv')
abu = read.table(f, sep="\t", stringsAsFactors=F)

#Map ENST to gene names, keep only estimated counts values 
ugene = cb[,2]
m3 = match(abu[,1], cb[,1])
cco = cbind(abu,ugene[m3])[-1,]
co = cco[,c(6,4)]
co[,1] = as.character(co[,1])
df = data.frame(co[,1], as.numeric(co[,2]))
colnames(df) = c("gene", "value")

#Sum rows with same gene name
dd = ddply(df,.(gene),summarize,sum=sum(value),number=length(gene))
ge = dd[,2]
names(ge) = dd[,1]

write.csv(ge, file.path(args[1], 'gene_abundance.csv'))