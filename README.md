**Project Description** 

This project is to perform Machine Learning on RNA-Seq data obtained from PPMI (Parkinson Progression Markers Initiative). The aim is to develop a predictive model to find the most important features contributing to Parkinson's disease.

**Installation**

To run the file you need to download the file: main_ppmi_notebook.ipynb and the 'Data' folder a directory before this file. In the Data folder, you can find all the necessary data file to be able to read inside the main jupyter notebook file.
- In the file conda_list.txt you acn find all the packeges installed using conda.

**Acknowledgment**

Thanks to Serafina Di Gioia for many helpful contribution.

**Contact**

If you have any questions feel to contact me: nazari.zainab@gmail.com

Note: You may ignore the file Downloading_data.ipynb which is to download automatically updated PPMI data, this file is under construction.

To install R from source code in the cluster : use the script given in the following link
https://gist.github.com/jtilly/d421b5d363cc2db860cc431a1128abc4




# <span style="color:#8B4513;"> Machine Learning and RNA-Seq Data of Parkinson Disease
</span>



[<span style="color:#8B4513;">Author: **Zainab Nazari**</span>](mailto:z.nazari@ebri.com)
 
 <span style="color:#8B4513;">EBRI – European Brain Research Institute Rita Levi-Montalcini | MHPC - Master in High Performance Computing</span>
 


## Introduction
By employing machine learning in PPMI clinical data set, we can develop predictive models that aid in the early diagnosis of the disease. These models can potentially identify specific genetic markers or gene signatures that correlate with disease progression or response to treatment.

## Table of Contents
- [Matrix of Gene IDs and Counts for Pateints](#matrixcreation)
- [Data Preprocessing STEP I](#preprocessing)
- [Data Preprocessing STEP II](#preprocessing2)
- [Model Training](#training)
- [Results and Evaluation](#results)

## Matrix of Gene IDs and Counts for Patients
- Loading the data from IR3/counts folder and extracting the associated last column (counts) of each patient file for their BL visit.


## Data Preprocessing STEP I
- We remove patients that have these mutations of genes: SNCA (ENRLSNCA), GBA (ENRLGBA), LRRK2 (ENRLLRRK2).
-  We only keep genes with the intersection of counts and quants with proteing coding and RNAincs.
- We remove the duplicated gene IDs in which they are also lowly expressed.
- We keep only patients with diagnosis of Health control or Parkinson disease.
- We check if there are some patients were they were taking dopamine drug, so we exclude them. Dopaminergic medication can impact the interpretation of experimental data or measurements and can alter gene expression patterns, so we need to remove them to have less biased data.

## Data Preprocessing STEP II
1. We remove lowely expressed genes, by keeping only genes that had more than five counts in at least 10% of the individuals, which left us with 21,273 genes

2. Similar DESeq2 but with numpy:  we estimated size factors, normalized the library size bias using these factors, performed independent filtering to remove lowly expressed genes using the mean of normalized counts as a filter statistic. This left us with 22969 genes

3. pyDESeq2: we apply a variance stabilizing transformation (vst) to accommodate the problem of unequal variance across the range of mean values.


4. limma: we used control samples to estimate the batch effect of the site, that we subsequently removed in both controls and cases. In experimental research, a batch effect is a systematic variation in data that can occur when data is collected from multiple sites (clinical centers). These factors can include differences in equipment, reagents, operators, or experimental conditions. Examples of batch effects: 
 - Differences in the equipment used to collect the data. For example, if you are using two different microarray platforms to measure gene expression, there may be differences in the way that the platforms detect and quantify gene expression.
 - Differences in the operators who collect the data. For example, if two different people are collecting RNA-seq data, they may have different levels of experience or expertise, which could lead to differences in the way that they process the samples.
 

5. using limma: we removed further confounding effects due to sex and RIN value. RIN value is a measure of the quality of RNA samples, and it can vary depending on the sample preparation method. Sex can also affect gene expression. If the effects of sex and RIN value are not removed, then the results of the analysis may be biased.


## Model Training
The code uses a Random Forest model to identify the most important features in a dataset. The code first performs
repeated stratified k-fold cross validation to train the Random Forest and compute the permutaion featute importanes. Then, the code counts the occurances of each features in the selected top features ist. Finally, the code gets the name of the final selected top features.

## Results and Evaluation
We present the results of the trained models, including performance metrics, accuracy, or any relevant evaluation measures. The model without preprocessing is with high recall score and low roc and auc score, and this means that the model is good to distinguishing the person with parkinson but not healthy people, therefore the model sounds very random.


## Conclusion
Summarize the key findings, limitations of the analysis, and potential future work or improvements. Offer closing remarks or suggestions for further exploration.

## References
- [**Parkinson’s Progression Markers Initiative (PPMI)**](https://www.ppmi-info.org/)

- [**A Machine Learning Approach to Parkinson’s Disease Blood Transcriptomics**](https://www.mdpi.com/2073-4425/13/5/727)

- [**Quality Control Metrics for Whole Blood Transcriptome Analysis in the Parkinson’s Progression Markers Initiative (PPMI)**](https://www.medrxiv.org/content/10.1101/2021.01.05.21249278v1)

