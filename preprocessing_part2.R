# Clear the environment
rm(list = ls())

# Set options to suppress warnings
options("warn"=0)  # print a maximum of 10 warnings on the screen

# Load required libraries
library(limma)  
library(edgeR)

##############################################  Parameters to set

# Set the working directory
work_path <- "/Users/zainabnazari/Desktop/aaaa1ltimate_files_mhpc_thesis"

# Define the input file names
name_file_input <- "ir3_rna_step1_preprocessing.csv"
name_batch_factor_file_input <- "factor_ir3_rna_step1_preprocessing_06feb2024.txt"

##############################################

# Set the working directory
setwd(work_path)

# Read the data from the input file into a dataframe
mydata <- read.table(file = name_file_input, sep = ",", quote = "\"", row.names = 1, header = TRUE, fill = TRUE)

# Read batch factor information from the input file into a dataframe
myfactors <- read.table(file = name_batch_factor_file_input, sep = "\t", quote = "\"", header = TRUE, fill = TRUE)

# Create factors for different variables
fac <- factor(myfactors$Diagnosis, levels = c("PD", "CTR"))  # Discretize the Diagnosis groups
sex_fac <- factor(myfactors$Sex, levels = c("M", "F"))
Clinical_center_fac <- factor(myfactors$Clinical_center)
RIN_covariate <- as.vector(myfactors$RIN)

# Create a copy of the original data (no filtering applied yet)
mydata_filtered <- mydata

# Create a DGEList object for differential gene expression analysis
dge <- DGEList(counts = mydata_filtered, group = fac)

# Create the design matrix for the analysis with sex correction
design <- model.matrix(~0 + fac)  # generate a design matrix

# Normalize factors using the TMM method
dge <- calcNormFactors(dge, method = "TMM")

# Compute log2 counts per million (logCPM)
logCPM <- cpm(dge, log = TRUE, prior.count = 2)

# Create a copy of logCPM (no filtering applied yet)
logCPM_filtered <- logCPM

# Remove batch effects from logCPM using clinical center, sex, and RIN as covariates
logCPM_filtered_batch_effect_removed <- removeBatchEffect(logCPM_filtered, batch = Clinical_center_fac, batch2 = sex_fac, 
                                                          covariates = RIN_covariate, design = design)

# Write the resulting matrix to a file
write.table(logCPM_filtered_batch_effect_removed, "mydata_TMM_Norm_Log2_CPM_filtered.txt", sep = "\t", row.names = TRUE, col.names = TRUE)

