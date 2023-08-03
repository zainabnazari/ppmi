#!/usr/bin/env python
# coding: utf-8

# # <span style="color:#8B4513;"> Machine Learning and RNA-Seq Data of Parkinson Disease
# </span>
# 
# 
# 
# [<span style="color:#8B4513;">Author: **Zainab Nazari**</span>](mailto:z.nazari@ebri.com)
#  
#  <span style="color:#8B4513;">EBRI – European Brain Research Institute Rita Levi-Montalcini | MHPC - Master in High Performance Computing</span>
#  
# 
# 
# ## Introduction
# By employing machine learning in PPMI clinical data set, we can develop predictive models that aid in the early diagnosis of the disease. These models can potentially identify specific genetic markers or gene signatures that correlate with disease progression or response to treatment.
# 
# ## Table of Contents
# - [Matrix of Gene IDs and Counts for Pateints](#matrixcreation)
# - [Data Preprocessing STEP I](#preprocessing)
# - [Data Preprocessing STEP II](#preprocessing2)
# - [Model Training](#training)
# - [Results and Evaluation](#results)
# 
# ## Matrix of Gene IDs and Counts for Pateints
# - Loading the data from IR3/counts folder and extracting the associated last column (counts) of each patient file for their BL visit.
# 
# 
# ## Data Preprocessing STEP I
# - We remove patients that have these diseases: SNCA (ENRLSNCA), GBA (ENRLGBA), LRRK2 (ENRLLRRK2).
# -  We only keep genes with the intersection of counts and quants with proteing coding and RNAincs.
# - We remove the duplicated gene IDs in which they are also lowly expressed.
# - We keep only patients with diagnosis of Health control or Parkinson disease.
# - We check if there are some patients were they were taking dopamine drug, so we exclude them. Dopaminergic medication can impact the interpretation of experimental data or measurements and can alter gene expression patterns, so we need to remove them to have less biased data.
# 
# ## Data Preprocessing STEP II
# 1. We remove lowely expressed genes, by keeping only genes that had more than five counts in at least 10% of the individuals, which left us with 21,273 genes
# 
# 2. Similar DESeq2 but with numpy:  we estimated size factors, normalized the library size bias using these factors, performed independent filtering to remove lowly expressed genes using the mean of normalized counts as a filter statistic. This left us with 22969 genes
# 
# 3. pyDESeq2: we apply a variance stabilizing transformation (vst) to accommodate the problem of unequal variance across the range of mean values.
# 
# 
# 4. limma: we used control samples to estimate the batch effect of the site, that we subsequently removed in both controls and cases. In experimental research, a batch effect is a systematic variation in data that can occur when data is collected from multiple sites (clinical centers). These factors can include differences in equipment, reagents, operators, or experimental conditions. Examples of batch effects: 
#  - Differences in the equipment used to collect the data. For example, if you are using two different microarray platforms to measure gene expression, there may be differences in the way that the platforms detect and quantify gene expression.
#  - Differences in the operators who collect the data. For example, if two different people are collecting RNA-seq data, they may have different levels of experience or expertise, which could lead to differences in the way that they process the samples.
#  
# 
# 5. using limma: we removed further confounding effects due to sex and RIN value. RIN value is a measure of the quality of RNA samples, and it can vary depending on the sample preparation method. Sex can also affect gene expression. If the effects of sex and RIN value are not removed, then the results of the analysis may be biased.
# 
# 
# ## Model Training
# The code uses a Random Forest model to identify the most important features in a dataset. The code first performs
# repeated stratified k-fold cross validation to train the Random Forest and compute the permutaion featute importanes. Then, the code counts the occurances of each features in the selected top features ist. Finally, the code gets the name of the final selected top features.
# 
# ## Results and Evaluation
# We present the results of the trained models, including performance metrics, accuracy, or any relevant evaluation measures. The model without preprocessing is with high recall score and low roc and auc score, and this means that the model is good to distinguishing the person with parkinson but not healthy people, therefore the model sounds very random.
# 
# 
# ## Conclusion
# Summarize the key findings, limitations of the analysis, and potential future work or improvements. Offer closing remarks or suggestions for further exploration.
# 
# ## References
# - [**Parkinson’s Progression Markers Initiative (PPMI)**](https://www.ppmi-info.org/)
# 
# - [**A Machine Learning Approach to Parkinson’s Disease Blood Transcriptomics**](https://www.mdpi.com/2073-4425/13/5/727)
# 
# - [**Quality Control Metrics for Whole Blood Transcriptome Analysis in the Parkinson’s Progression Markers Initiative (PPMI)**](https://www.medrxiv.org/content/10.1101/2021.01.05.21249278v1)
# 
# 

# In[1]:


# In case you do not have following packages installed, uncomment instalisation.

import pandas as pd
import numpy as np
import os
import glob
import functools
from pathlib import Path
import matplotlib.pyplot as plt

#!pip install dask[complete];
# you need to run these in case dask gives you error, it might need update.
#!pip install --upgrade pandas "dask[complete]"
#python -m pip install "dask[dataframe]" --upgrade
import dask.dataframe as dd

from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, accuracy_score
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance       

from sklearn.feature_selection import SelectFromModel
from sklearn.utils import class_weight

#!pip3 install xgboost
from xgboost import XGBClassifier

#!pip install conorm
import conorm # for tmm normalisation

#!pip3 install pydeseq2 or pip install pydeseq2
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from pydeseq2.utils import load_example_data



#to install R :
#conda install -c r r-irkernel

#to install a library from R
#!pip install library edgeR


# In[2]:


# Note that the counts file in the IR3 is around 152 G, and the files are located in scratch area.

path_to_files="/scratch/znazari/PPMI_ver_sep2022/RNA_Seq_data/star_ir3/counts/"
path1=Path("/scratch/znazari/PPMI_ver_sep2022/RNA_Seq_data/star_ir3/counts/")
path2 = Path("/home/znazari/data") # where the output data will be saved at the end.
path3=Path("/scratch/znazari/PPMI_ver_sep2022/study_data/Subject_Characteristics/")


# <a id="matrixcreation"></a>
# ## Matrix of Gene IDs and Counts for Pateints
#  Loading the data from IR3/counts folder and extracting the associated last column (counts) of each patient file for their BL visit.

# In[ ]:


#reading the files which are in BL (Base line) visit.
specific_word = 'BL'
ending_pattern = '*.txt'
file_pattern = f'*{specific_word}*.{ending_pattern}'
file_paths = glob.glob(path_to_files + file_pattern)
# 'bl.txt' is a file that ccontains the name of the files with patient, BL, IR3, counts.
filename = 'bl.txt'
file_path_2 = os.path.join(path_to_files, filename)
bl_files = pd.read_csv(file_path_2,header=None)

# We define a function where we can take the second phrase seperated by dot. The second phrase 
# is the patient ID. So with this functin we want to get the patient IDs from their file's name
def function_names(fname):
    tokens=fname.split('.')
    return tokens[1]

# we create a list with the name of the each patients.
bl_list = [function_names(bl_files.iloc[i][0]) for i in range(len(bl_files))]

# here we read all the files with with base visit(BL) from the counts folder (where we have all the files
# for all the patients and all the visit).
list_bl_files = [dd.read_csv(path1/bl_files.iloc[i][0],skiprows=1,delimiter='\t') for i in range(len(bl_files))]


# we get th last columns of each file in the list
last_columns = [ddf.iloc[:, -1:] for ddf in list_bl_files]

# concatinating the list of the columns in a single file.
single_file = dd.concat(last_columns, axis=1)

# we change the name of the each columns with the patient numbers.
single_file.columns = bl_list

# we get the Geneid column and convert it to dask dataframe
pd_tmp_file = list_bl_files[3].compute()
geneid = pd_tmp_file['Geneid']
ddf_geneid = dd.from_pandas(geneid, npartitions=1)

# here we set the Geneid column as the index of the matrix.
ddf_new_index = single_file.set_index(ddf_geneid)

# converting to pandas data frame and saving.
ir3_counts = ddf_new_index.compute()
ir3_counts.to_csv(path2/"matrix_ir3_counts_bl.csv")


# <a id="preprocessing"></a>
# ## Data Preprocessing STEP I
# 
# - We remove patients that have these diseases: SNCA (ENRLSNCA), GBA (ENRLGBA), LRRK2 (ENRLLRRK2).
# - dopamin drug using
# -  We only keep genes with the intersection of counts and quants with proteing coding and RNAincs.
# - We remove the duplicated gene IDs in which they are also lowly expressed.
# - We keep only patients with diagnosis of Health control or Parkinson disease.
# - We check if there are some patients were they were taking dopomine drug, so we exclude them.
# 

# In[5]:


# reading the file
read_ir3_counts = pd.read_csv(path2/"matrix_ir3_counts_bl.csv")
# setting the geneid as indexing column
read_ir3_counts.set_index('Geneid', inplace=True)
# result with removing the after dot (.) value, i.e. the version of the geneIDs is removed.
read_ir3_counts.index =read_ir3_counts.index.str.split('.').str[0]


#here we delete the duplicated gene IDs, first we find them then remove them from the gene IDs
# as they are duplicated and also they are very lowly expressed either zero or one in rare caes.

# Check for duplicate index values
is_duplicate = read_ir3_counts.index.duplicated()

# Display the duplicate index values
duplicate_indices = read_ir3_counts.index[is_duplicate]

# drop them (duplicated indices and their copies are deleted, 45 duplicatd indices and 90 are dropped)
to_be_deleted = list(duplicate_indices)
read_ir3_counts = read_ir3_counts.drop(to_be_deleted)

# we read the file where we have an intersection of geneIDs in IR3, counts, quant
intersect = pd.read_csv(path2/"intersect_IR3_ENG_IDs_LincRNA_ProtCoding_counts_quant_gene_transcript_only_tot_intsersect.txt")
intersection = read_ir3_counts.index.intersection(intersect['[IR3_gene_counts] and [IR3_quant_gene] and [IR3_quant_trans] and [lncRNA+ProtCod]: '])
filtered_read_ir3_counts = read_ir3_counts.loc[intersection]

# reading the file which contains diagnosis
diago=pd.read_csv(path3/"Participant_Status.csv", header=None )
diago1=diago.rename(columns=diago.iloc[0]).drop(diago.index[0]).reset_index(drop=True)

#this is to remove patients that have these diseases: SNCA (ENRLSNCA), GBA (ENRLGBA), LRRK2 (ENRLLRRK2)
filtered_SNCA_GBA_LRRK2 = diago1[(diago1['ENRLSNCA'] == "0")& (diago1['ENRLGBA'] == "0")& (diago1['ENRLLRRK2'] == "0")]

#patients with their diagnosis
patinets_diagnosis = filtered_SNCA_GBA_LRRK2[['PATNO','COHORT_DEFINITION']].reset_index(drop=True)

# Define the particular names to keep
names_to_keep = ['Healthy Control', "Parkinson's Disease"]


# Filter the dataframe based on the specified names
PK_HC_pateints = patinets_diagnosis[patinets_diagnosis['COHORT_DEFINITION'].isin(names_to_keep)]

# Get the list of patient IDs with diagnosis from the second dataframe
patient_ids_with_diagnosis = PK_HC_pateints['PATNO']
list_patients=list(patient_ids_with_diagnosis)

# Filter the columns in the first dataframe based on patient IDs with diagnosis
rna_filtered = filtered_read_ir3_counts.filter(items=list_patients)

# We read a file that contains the Patient IDs that they were taking dopomine drugs, so they needed to be excluded.
patient_dopomine = pd.read_csv(path2/'Patient_IDs_taking_dopamine_drugs.txt',delimiter='\t',  header=None)
patient_dopomine = patient_dopomine.rename(columns={0: 'Pateint IDs'})
ids_to_remove = patient_dopomine['Pateint IDs'].tolist() # put the patient IDs to list
strings = [str(num) for num in ids_to_remove] # convert them as string

# The code is iterating over each column name in rna.columns and checking if any of the strings in the strings list 
# are present in that column name. If none of the strings are found in the column name,
# then that column name is added to the new_columns list.
new_columns = [col for col in rna_filtered.columns if not any(string in col for string in strings)] 
rna_filtered = rna_filtered[new_columns]
# there were no column name (patints that use druf in this list) to be excluded in our case.
# IN CASE THERE WERE SOME PATIENTS TO BE REMOVED, the diagnosis file below needs to be amended too.

rna_filtered.to_csv(path2/'ir3_rna_step1.csv', index=True)

# we keep only the patients that are common in the two dataframes:
common_patient_ids = list(set(PK_HC_pateints['PATNO']).intersection(rna_filtered.columns))
patient11_filtered = PK_HC_pateints[PK_HC_pateints['PATNO'].isin(common_patient_ids)]
patient11_filtered.reset_index(drop=True)

# we save the output into data folder
patient11_filtered.to_csv(path2/'patients_HC_PK_diagnosis.csv', index=False)


# <a id="preprocessin2"></a>
# ## Data Preprocessing STEP II
# 
# 1. Removing lowely expressed genes, by keeping only genes that had more than five counts in at least 10% of the individuals, which left us with 25317 genes
# 
# 2. Similar DESeq2: we estimated size factors, normalized the library size bias using these factors, performed independent filtering to remove lowly expressed genes using the mean of normalized counts as a filter statistic. This left us with 22969 genes
# 
# 3. DESeq2: we apply a variance stabilizing transformation to accommodate the problem of unequal variance across the range of mean values.
# 
# 4. limma: we used control samples to estimate the batch effect of the site, that we subsequently removed in both controls and cases 
# 
# 5. limma: we removed further confounding effects due to sex and RIN value.

# In[26]:


rna_step1 = pd.read_csv(path2/'ir3_rna_step1.csv')
rna_step1.set_index('Geneid', inplace=True)


# In[22]:


# 1. Removing lowely expressed genes, by keeping only genes that had more than five counts in 
#at least 10% of the individuals, which left us with 25317 genes
gene_counts = rna_step1.sum(axis=1)
gene_mask = gene_counts > 5
gene_percentage = (rna_step1 > 5).mean(axis=1)
percentage_mask = gene_percentage >= 0.1
filtered_data = rna_step1[gene_mask & percentage_mask]

# we estimated size factors, normalized the library size bias using these factors,
# performed independent filtering to remove lowly expressed genes using the mean of normalized counts as a filter statistic.
#This left us with 22969 genes
# Step 1: Estimating Size Factors
library_sizes = filtered_data.sum(axis=0)
median_library_size = np.median(library_sizes)
size_factors = library_sizes / median_library_size

# Step 2: Normalizing Library Size Bias
normalized_data = filtered_data.divide(size_factors, axis=1)

# Step 3: Performing Independent Filtering
mean_normalized_counts = normalized_data.mean(axis=1)
threshold = 5  # Adjust this threshold as desired
filtered_data2 = normalized_data.loc[mean_normalized_counts >= threshold]


#we need to round and make the counts values integer because that what deseq2 type requires.
filtered_data2 = filtered_data2.round().astype(int)
filtered_data2 = filtered_data2.T
# we make the patient ids as string type otherwise we get warning when transforming to deseq data set.
filtered_data2.index = filtered_data2.index.astype(str)
filtered_data2.to_csv(path2/'ir3_rna_step2.csv', index=True)


diagnosis = pd.read_csv(path2/'patients_HC_PK_diagnosis.csv')
patnn=diagnosis.set_index("PATNO")
# renaming the column as "condition" is necessary for deseq transformation.
patnn.rename(columns={'COHORT_DEFINITION': 'condition'}, inplace=True)
patnn.index = patnn.index.astype(str)


# In[14]:


# here is to make a dese data set:
dds = DeseqDataSet(
    counts=filtered_data2,
    clinical=patnn,
    design_factors="condition"
)
#dds.obs # show patients diagnosis
#dds.X # show array of counts
# dds.var # show Geneids

# Perform VST transformation
dds.vst()

# Here we get the VST data which are in the numpy form.
vst_transformed_dds=dds.layers["vst_counts"]

# We convert the numpy data to pandas dataframe
pd_vst= pd.DataFrame(vst_transformed_dds)

# the above file does not have patient IDs name as well as Gene IDs so we need to take it from the other
# file and then add it to bare dataframe file

ir3_rna_step2 = pd.read_csv(path2/'ir3_rna_step2.csv')
# patient IDs 
patient_ids = ir3_rna_step2['Unnamed: 0']

# set them as the index of rows: 
pd_vst.set_index(patient_ids, inplace = True)

# taking the gene IDs properly as a list format
geneids = list(ir3_rna_step2.columns)[1:]

# add the list of columns to the pandas dataframe file:
pd_vst.columns = geneids
# Saving the matrix with vst applied into csv file.
pd_vst.to_csv(path2/'ir3_rna_step_vst.csv', index=True)


# # RIN and SEX effects to be removed 

# In[ ]:


sex_rin_data = pd.read_csv(path2/'Patient_IDs_RNA_sample_RIN_sex_CNO_diagnosis.txt',delimiter='\t')
sex_rin_data 


# In[ ]:


unique_values = sex_rin_data['CLINICAL_EVENT'].unique()
unique_values  


# In[ ]:


# we need to keep only base line visit data of patient.
#baseline_df = sex_rin_data[sex_rin_data['CLINICAL_EVENT'] == 'BL']
# Filter the DataFrame to keep rows with 'BL' and 'SC' values in the 'CLINICAL_EVENT' column
baseline_df = sex_rin_data[sex_rin_data['CLINICAL_EVENT'].isin(['BL'])]


# In[ ]:


diagnosis = pd.read_csv(path2/'patients_HC_PK_diagnosis.csv')
diagnosis


# In[ ]:


patient_ids = diagnosis['PATNO']
filtered_df_sex_rin = baseline_df[baseline_df['ALIAS_ID'].isin(patient_ids)]
filtered_df_sex_rin


# In[ ]:


# Check if all patient IDs in diagnosis exist in filtered_df_sex_rin
all_exist = diagnosis['PATNO'].isin(baseline_df['ALIAS_ID']).all()

# Print the result
if all_exist:
    print("All patient IDs in other_df exist in big_df.")
else:
    print("Not all patient IDs in other_df exist in big_df.")
# I note that all the patient IDs that I analysis are note only in BL some of them are in V01.
# I need to find those that are in the v01 and only add them not the rest.


# In[ ]:


duplicates = filtered_df_sex_rin['ALIAS_ID'].duplicated()
duplicate_rows = filtered_df_sex_rin[duplicates]


# In[ ]:


duplicate_rows


# In[ ]:


duplicates


# In[ ]:


# Find the duplicated patient IDs in the filtered DataFrame
duplicates = filtered_df_sex_rin['ALIAS_ID'].duplicated(keep=False)

# Filter the DataFrame to keep only the duplicated rows
duplicate_rows = filtered_df_sex_rin[duplicates]

# Sort the duplicate rows by the patient ID
duplicate_rows_sorted = duplicate_rows.sort_values('ALIAS_ID')

# Print the duplicate rows
duplicate_rows_sorted


# <a id="training"></a>
# ## Model Training 
# 
# Build and train machine learning models on the prepared data. Explain the choice of models, feature engineering techniques, and hyperparameter tuning. Provide code and comments to walk through the model training process.

# In[3]:


ir3_rna_step_vst =  pd.read_csv(path2/'ir3_rna_step_vst.csv')

ir3_rna_step_vst.rename(columns={"Unnamed: 0": "PATNO"}, inplace=True)

ir3_rna_step_vst.set_index("PATNO", inplace = True)

diagnosis = pd.read_csv(path2/'patients_HC_PK_diagnosis.csv')

# mapping diagnosis to zero and one.
diagnosis['COHORT_DEFINITION'] = diagnosis['COHORT_DEFINITION'].map({'Healthy Control': 0, "Parkinson's Disease": 1})


# In[ ]:


# to see how many Parkinson and Healthy patients we have
diagnosis['COHORT_DEFINITION'].value_counts()


# In[16]:


# Initialize evaluation metric lists
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
roc_auc_scores = []
confusion_matrices = []

# Perform the iterations
for _ in range(20):
    # Split the data into training and test sets
    # X_train: train data of RNA,  X_test: test data of RNA 
    # y_train: train diagnosis, y_test: test diagnosis
    X_train, X_test, y_train, y_test = train_test_split(ir3_rna_step_vst, diagnosis['COHORT_DEFINITION'], test_size=.3)
    
    # Create and fit the Random Forest model
    model_rf = RandomForestClassifier(n_estimators=100)
    model_rf.fit(X_train, y_train)
    
    # Evaluate the model and store the metrics
    accuracy_scores.append(accuracy_score(y_test, model_rf.predict(X_test)))
    precision_scores.append(precision_score(y_test, model_rf.predict(X_test)))
    recall_scores.append(recall_score(y_test, model_rf.predict(X_test)))
    f1_scores.append(f1_score(y_test, model_rf.predict(X_test)))
    roc_auc_scores.append(roc_auc_score(y_test, model_rf.predict(X_test)))
    confusion_matrices.append(confusion_matrix(y_test, model_rf.predict(X_test)))

# Calculate the average evaluation metrics
avg_accuracy = np.mean(accuracy_scores)
avg_precision = np.mean(precision_scores)
avg_recall = np.mean(recall_scores)
avg_f1 = np.mean(f1_scores)
avg_roc_auc = np.mean(roc_auc_scores)
avg_confusion_matrix = np.mean(confusion_matrices, axis=0)

# Print the average evaluation metrics
print('Average Evaluation Metrics')
print('Accuracy:  ', avg_accuracy)
print('Precision: ', avg_precision)
print('Recall:    ', avg_recall)
print('F1:        ', avg_f1)
print('ROC-AUC:   ', avg_roc_auc)
print('\nAverage Confusion Matrix\n', avg_confusion_matrix)


# In[ ]:


get_ipython().run_cell_magic('time', '', '# \'ir3_rna_step_vst\' is my RNA-seq data and \'diagnosis\' is the corresponding labels\n\n# Set the number of repetitions for Random Forest training\nnum_repetitions = 1\n\n# Set the number of folds for repeated stratified 10-fold cross-validation\nnum_folds = 2\n\n# Set the random seed for reproducibility\nrandom_seed = 42\n\n# Define the function to train a Random Forest model and compute permutation feature importances\ndef train_rf_with_permutation_importance(X_train, y_train, X_test, y_test, num_features):\n    rf_model = RandomForestClassifier(n_estimators=10, random_state=random_seed)\n    rf_model.fit(X_train, y_train)\n\n    perm_importance = permutation_importance(rf_model, X_test, y_test, n_repeats=1, random_state=random_seed)\n    feature_importance = perm_importance.importances_mean\n\n    # Get the indices of the top features based on their importance\n    top_feature_indices = np.argsort(feature_importance)[-int(np.sqrt(num_features)):]\n    \n    return top_feature_indices\n\n# Initialize lists to store the selected top features across all splits\nselected_top_features = []\n\n# Create the repeated stratified k-fold cross-validator\ncv = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_seed)\n\n# Repeat the process for a total of \'num_repetitions\' times\nfor repetition in range(num_repetitions):\n    print(f"Repetition {repetition+1}/{num_repetitions}")\n\n    # Perfo0rm repeated stratified k-fold cross-validation\n    for train_idx, test_idx in cv.split(ir3_rna_step_vst, diagnosis[\'COHORT_DEFINITION\']):\n        X_train, X_test = ir3_rna_step_vst.iloc[train_idx], ir3_rna_step_vst.iloc[test_idx]\n        y_train, y_test = diagnosis[\'COHORT_DEFINITION\'].iloc[train_idx], diagnosis[\'COHORT_DEFINITION\'].iloc[test_idx]\n\n        # Get the selected top features for this split\n        top_features_split = train_rf_with_permutation_importance(X_train, y_train, X_test, y_test, len(ir3_rna_step_vst.columns))\n\n        # Add the selected top features to the list\n        selected_top_features.extend(top_features_split)\n\n# Count the occurrences of each feature in the selected_top_features list\nfeature_counts = pd.Series(selected_top_features).value_counts()\n\n# Get the indices of the top features based on their overall occurrence\nfinal_top_feature_indices = feature_counts.index[:int(np.sqrt(len(ir3_rna_step_vst.columns)))]\n\n# Get the names of the final selected top features\nfinal_top_features = ir3_rna_step_vst.columns[final_top_feature_indices]\nfinal_top_features.to_csv(path2/"important_features.csv", index = True)\n\n')


# In[ ]:





# In[ ]:


roc_auc_scores


# In[32]:


from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Assuming you have X_train and y_train prepared
X_train, X_test, y_train, y_test = train_test_split(ir3_rna_step_vst, diagnosis['COHORT_DEFINITION'], test_size=.3)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Get feature importances
importances = model.feature_importances_

# Sort the importances in descending order
sorted_indices = importances.argsort()[::-1]
sorted_importances = importances[sorted_indices]
sorted_feature_names = [feature_names[i] for i in sorted_indices]

# Select only the top 50 features
top_50_indices = sorted_indices[:50]
top_50_importances = sorted_importances[:50]
top_50_feature_names = sorted_feature_names[:50]

# Visualize the top 50 feature importances
plt.figure(figsize=(15, 8))
plt.bar(range(len(top_50_importances)), top_50_importances, tick_label=top_50_feature_names)
plt.xticks(rotation=90)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Top 50 Feature Importances')
plt.show()

# Select the top 50 features from the original dataset
X_train_top_50 = X_train[top_50_feature_names]


# In[14]:


len(feature_names)


# In[10]:


top_50_feature_names


# <a id="results"></a>
# ## Results and Evaluation 
# 
# Present the results of the trained models, including performance metrics, accuracy, or any relevant evaluation measures. Interpret the findings and discuss the implications. Include visualizations or tables to support the results.

# In[ ]:


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve

# Initialize lists to store the ROC and precision-recall data
roc_curves = []
precision_recall_curves = []

# Perform the iterations
for _ in range(20):
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(rna_ir3, diagnosis['COHORT_DEFINITION'], test_size=.3)
    
    # Create and fit the Random Forest model
    model_rf = RandomForestClassifier(n_estimators=100)
    model_rf.fit(X_train, y_train)
    
    # Calculate the ROC curve
    fpr, tpr, _ = roc_curve(y_test, model_rf.predict_proba(X_test)[:, 1])
    roc_curves.append((fpr, tpr))
    
    # Calculate the precision-recall curve
    precision, recall, _ = precision_recall_curve(y_test, model_rf.predict_proba(X_test)[:, 1])
    precision_recall_curves.append((precision, recall))

# Plot ROC curves
plt.figure(figsize=(8, 6))
for fpr, tpr in roc_curves:
    plt.plot(fpr, tpr, lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.show()

# Plot precision-recall curves
plt.figure(figsize=(8, 6))
for precision, recall in precision_recall_curves:
    plt.plot(recall, precision, lw=1)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves')
plt.show()


# In[ ]:


#ploting roc curve and precision recall curve
roc = roc_curve(y_test,model_rf.predict_proba(X_test)[:,1])
pr  = precision_recall_curve(y_test,model_rf.predict_proba(X_test)[:,1])

f = plt.figure(figsize=(20,7))
ax = f.add_subplot(121)
ax.plot(roc[0],roc[1])
ax.set_ylabel('True Positive Rate')
ax.set_xlabel('False Positive Rate')
ax.set_title('ROC Curve')
ax.grid(which='both')
ax = f.add_subplot(122)
ax.plot(pr[1],pr[0])
ax.set_ylabel('Precision')
ax.set_xlabel('Recall')
ax.set_title('PR-curve')
ax.grid(which='both')
plt.show()


# In[ ]:




