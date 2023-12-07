import numpy as np
import pandas as pd
import numpy as np
import os
import glob
import functools
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
'''
Machine Learning paradigm

Within a repeated stratified, (to tackle 1: the control-patient mismatch, 2: 10-fold cross-validation framework, with 3: 20 iterations), we trained multiple RF models with 100 repetitions, where each repetition used a different seed of the random generation process  to evaluate permutation feature importance measures. 

Each forest was grown using 1000 trees, a sufficient value to allow the algorithm to reach a stable plateau of the out-of-bag internal error. The features selected at each split were square root of (f) with f being the overall number of genes, which is the default value for this parameter.

We determined the overall feature importance ranking by averaging over the 100 repetitions. 

We are training random forest for 200 * 100 = 20,000 times. In the following code each of 100 random forest will be trained with 200 different indexing for taining and testing data. In other words we use a single state of random forest for 200 different combination of taining and testing.

Cosidering that each permutaion importance uses 10 times shuffling of the value and resuing the random forest, in overal random forest will be used 200,000 times.


Note: The n_repeats parameter in the permutation_importance() function in Scikit-Learn controls how many times a feature is randomly shuffled and the model is retrained to evaluate its importance. A higher value of n_repeats will result in a more accurate estimate of feature importance, but it will also be more computationally expensive.

'''
# Loading the already preprocessed data:

path2 = Path("/home/znazari/data") # where the output data will be saved at the end.

# with all the filtered genes:

ir3_rna_step_vst =  pd.read_csv(path2/'mydata_Log_CPM_filtered_bact_sex_effect_removed_RIN_covariate.txt',delimiter='\t' )

diagnosis = pd.read_csv(path2/'patients_HC_PK_diagnosis.csv')

# mapping diagnosis to zero and one.
diagnosis['COHORT_DEFINITION'] = diagnosis['COHORT_DEFINITION'].map({'Healthy Control': 0, "Parkinson's Disease": 1})

# X: feature matrix, y: the target variable

X=ir3_rna_step_vst.T
y=diagnosis['COHORT_DEFINITION']

# Initialize repeated stratified K-fold cross-validation
n_splits = 10 
n_repeats = 20
rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)



# Define the parameters for the Random Forest
n_estimators = 1000
max_features = int(np.sqrt(X.shape[1]))  # Square root of the total number of features

# Initialize arrays to store feature importance scores
feature_importance_scores = np.zeros((1, X.shape[1]))  # 100 repetitions, n_features

# Loop through each repetition
for repetition in range(100):
    
    # Loop through each fold
    for train_index, test_index in rkf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Initialize a Random Forest model
        rf_model = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, random_state=repetition)

        # Fit the Random Forest model, we fit the X training sample with its associated y labels.
        rf_model.fit(X_train, y_train)

        # Calculate permutation feature importance
        permutation_importance_result = permutation_importance(rf_model, X_test, y_test, n_repeats=10, scoring='roc_auc')

        # Accumulate feature importance scores
        feature_importance_scores[repetition] += permutation_importance_result.importances_mean

# Average the feature importance scores across repetitions
average_feature_importance_scores = np.mean(feature_importance_scores, axis=0)

# Get the feature indices sorted by importance
sorted_feature_indices = np.argsort(average_feature_importance_scores)[::-1]


output_file = "feature_importance_ranking.txt"  # Name of the output file

with open(output_file, 'w') as file:
    file.write("Feature Importance Ranking:\n")
    for i in sorted_feature_indices:
        file.write(f"Feature {i}: {average_feature_importance_scores[i]}\n")
