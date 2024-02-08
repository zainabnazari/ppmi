import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import permutation_importance, roc_auc_score
import numpy as np
from joblib import Parallel, delayed
from dask.distributed import Client
from dask_mpi import initialize

# Loading the already preprocessed data:

path2 = Path("/home/znazari/data")  # Adjust the path to your data directory

# with all the filtered genes:
ir3_rna_step_vst = pd.read_csv(path2/'mydata_Log_CPM_filtered_bact_sex_effect_removed_RIN_covariate.txt', delimiter='\t')

# Read diagnosis data
diagnosis = pd.read_csv(path2/'patients_HC_PK_diagnosis.csv')

# mapping diagnosis to zero and one.
diagnosis['COHORT_DEFINITION'] = diagnosis['COHORT_DEFINITION'].map({'Healthy Control': 0, "Parkinson's Disease": 1})

# X: feature matrix, y: the target variable
X = ir3_rna_step_vst.T
y = diagnosis['COHORT_DEFINITION']

# Initialize repeated stratified K-fold cross-validation
n_splits = 10
n_repeats = 20
rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

# Define the parameters for the Random Forest
n_estimators = 1000
max_features = int(np.sqrt(X.shape[1]))  # Square root of the total number of features

# Initialize arrays to store feature importance scores
feature_importance_scores = np.zeros((1, X.shape[1]))  # 100 repetitions, n_features

# Function to train Random Forest models and calculate feature importance
def train_rf_and_get_importance(X_train, y_train, X_test, y_test, repetition):
    rf_model = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, random_state=repetition)
    rf_model.fit(X_train, y_train)
    permutation_importance_result = permutation_importance(rf_model, X_test, y_test, n_repeats=10, scoring='roc_auc')
    return permutation_importance_result.importances_mean

# Initialize Dask cluster and client interface
initialize()

dask_client = Client()

# Loop through each repetition
for repetition in range(100):
    # Loop through each fold in parallel
    results = Parallel(n_jobs=-1)(delayed(train_rf_and_get_importance)(
        X.iloc[train_index], y.iloc[train_index], X.iloc[test_index], y.iloc[test_index], repetition
    ) for train_index, test_index in rkf.split(X, y))

    # Accumulate feature importance scores
    feature_importance_scores[repetition] += np.mean(results, axis=0)

# Average the feature importance scores across repetitions
average_feature_importance_scores = np.mean(feature_importance_scores, axis=0)

# Get the feature indices sorted by importance
sorted_feature_indices = np.argsort(average_feature_importance_scores)[::-1]

output_file = "feature_importance_ranking.txt"  # Name of the output file

with open(output_file, 'w') as file:
    file.write("Feature Importance Ranking:\n")
    for i in sorted_feature_indices:
        file.write(f"Feature {i}: {average_feature_importance_scores[i]}\n")

