
rm(list = ls())
options("warn"=0)  #print max 10 warnings on screen
library(limma)      # for package 'limma'


##############################################  Parameters to set


				
work_path <- "/Users/zainabnazari/Desktop/aaaa1ltimate_files_mhpc_thesis"

name_file_input="ir3_rna_step1_preprocessing.csv"

name_batch_factor_file_input="factor_ir3_rna_step1_preprocessing_06feb2024.txt"


##############################################
   
setwd(work_path)



mydata<-read.table(file=name_file_input, sep = ",", quote = "\"",row.names=1,header=TRUE, fill=TRUE)  # re-read data into a dataframe with just numbers as real data

myfactors<-read.table(file=name_batch_factor_file_input, sep = "\t", quote = "\"",header=TRUE, fill=TRUE)  # re-read data into a dataframe 


fac = factor(myfactors$Diagnosis,levels=c("PD","CTR"))                  #  factor() discretizza i gruppi, creando dei fattori
sex_fac = factor(myfactors$Sex,levels=c("M","F"))
Clinical_center_fac = factor(myfactors$Clinical_center)
RIN_covariate=as.vector(myfactors$RIN)




mydata_filtered=mydata


dge<- DGEList(counts = mydata_filtered, group = fac)  # compute Differential IDs  group2-group1

		# disegnare/creare la matrice factor, with sex correction
design <- model.matrix(~0 + fac )       #  genera una matrice 'design' ,                               le righe corrispondono ai parametri da stimare, le colonne alla condizione sperimentale


dge <- calcNormFactors(dge,method="TMM")


logCPM <- cpm(dge, log=TRUE, prior.count=2)  #The prior count is used here to avoid log(0). The logCPM values can then be used in any standard limma pipeline, using the trend=TRUE argument when running eBayes or treat. 
logCPM_filtered=logCPM


logCPM_filtered_batch_effect_removed = removeBatchEffect(logCPM_filtered, batch=Clinical_center_fac, batch2=sex_fac, 
														covariates=RIN_covariate, design=design)

write.table(logCPM_filtered_batch_effect_removed, "mydata_TMM_Norm_Log2_CPM_filtered.txt", sep="\t",row.names=TRUE, col.names=TRUE)

