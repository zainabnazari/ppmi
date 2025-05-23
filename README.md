# <span style="color:#8B4513;"> **Project Description**
</span>

This project involves applying Machine Learning techniques to analyze data provided by the Parkinson Progression Markers Initiative (PPMI) example (*omics data, motor scores, and brain imaging, etc). The primary objectives are to create a predictive model for enhanced diagnosis as well as to gain a deeper understanding of the heterogeneous nature of individuals affected by Parkinson's diseasein progressive stages of their disease trajectory with variety of machine learning models, including supervised, unsupervised, and neural network models.

[<span style="color:#8B4513;"> **Zainab Nazari**</span>](mailto:z.nazari@ebri.com)
 
 <span style="color:#8B4513;">EBRI – European Brain Research Institute Rita Levi-Montalcini | MHPC - Master in High Performance Computing</span>
 

# Table of Contents

## - Excluding Pateints
In the file "excluded_patients.ipynb" we extract patients that are either using dopaminergic medication or having mutation genes.

## - RNA Sequencing /

We consider latest available data for RNA Sequencing Feature Counts/TPM (IR3/B38/Phases 1-2) version 2021-04-02
We make a table of Ensembl Gene IDs versus Patient numbers in the file rna_seq_table.ipynb with 58780 genes and 1530 individuals. 

### Preprocessing Part I

- Keep only individuals with diagnosis of Health Control or Parkinson's Disease.
- Remove patients that have these gene mutations : SNCA, GBA, LRRK2, and taking dopaminergic drugs.
- Remove the duplicated gene IDs which are those that carry ensembl genes with suffix _PAR_Y and their X transcripts.
-  Only keep genes that are either in the 19393 protein coding gene list or in 5874 long intergenic non-coding RNAs (lincRNAs) list that we obtained from the official HGNC repository (date: 31-Jan-2024).
- Filter out genes with low expression levels, retaining only those genes that exhibit more than five counts in a minimum of 10% of the individuals. 


### Preprocessing Part II

- Create factors for diagnosis, sex, clinical center, and RIN from batch factor information.
- Perform differential gene expression analysis using the limma package.
- Normalize factors, compute log2 counts per million, and create a design matrix with sex correction.
- Filter and normalize gene expression data.
- Remove batch effects using clinical center, sex, and RIN as covariates.

The preprocessing file can be found in preprocessing_part2.R, I am grateful to Ivan Arisi for sharing valuable information with me regarding this aspect.

### ML AdaBoost

The code performs machine learning analysis using AdaBoost algorithm on RNA-Seq dataset and evaluates the performance of multiple models across multiple trials.

#### ML with best 148 genes and using XGBoost and CatBoost:
clearly Catboost outperfom in the computaion of AUC with cross validation.

## RNA-Seq data from PDBP with diagnosis 
We add the table where we extract the RNA-Seq of PDBP cohort from AMP-PD cohort.
We only keep those individuals with parents having no PD so to keep more data for analysis. 

## - Proteomics/


In the file "proteomic-table.ipynb" you can find code on how to make the table which contains the proteomic csf genes with patients and their diagnosis

In the file "proteomic-ML.ipynb" you can find the code with predictive model using xgboost for dianosis of PD vs Control.

## - Motor Score 

In the file "motor_score.ipynb" you can find a ML test for UPDRS total score.

## - UPSIT

University of Pennsylvania Smell Identification Test, in the file for ppmi cohort in ppmi_UPSIT.ipynb and pdbp cohort in pdbp_UPSIT.ipynb

## - Plots

Distribution of Participants Diagnosis, Ages and  Across Different Visits in the file: plots.ipynb

## - External_Data/

Some external data that is needed for this study.

**Installation**

- In the file conda_list.txt you acn find all the packeges installed using conda.

**Contact**

If you have any questions/suggestion or want to contribute feel to contact me: z.nazari@ebri.it

**Acknowledgement**

I am grateful to Ivan Arisi for sharing valuable information with me regarding this project and particularly for the prepresossesing STEP II as well as ML learning algorithm with AdaBoost.

Last update : 2024-05-22
