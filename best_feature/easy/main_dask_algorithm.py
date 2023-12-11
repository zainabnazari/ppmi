from dask_mpi import initialize
from dask.distributed import Client
import joblib
from sklearn.ensemble import RandomForestClassifier
import sklearn.datasets 
import pandas as pd
from pathlib import Path
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.inspection import permutation_importance
import numpy as np
import os

if __name__ == "__main__":

    print('I am before client initialization')

    # Initialize Dask cluster and client interface
    n_tasks = int(os.getenv('SLURM_NTASKS'))
    mem = os.getenv('SLURM_MEM_PER_CPU')
    mem = str(int(mem)) + 'MB'

    initialize(memory_limit=mem)

    dask_client = Client()

    dask_client.wait_for_workers(n_workers=(n_tasks-2))
    num_workers = len(dask_client.scheduler_info()['workers'])
    print("%d workers available and ready" % num_workers)

    # Loading the already preprocessed data:
    path2 = Path("/home/znazari/data")

    # Read data
    ir3_rna_step_vst = pd.read_csv(path2/'mydata_Log_CPM_filtered_bact_sex_effect_removed_RIN_covariate.txt', delimiter='\t')
    diagnosis = pd.read_csv(path2/'patients_HC_PK_diagnosis.csv')

    # Map diagnosis to zero and one.
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

    # Loop through each repetition
    def process_repetition(repetition):
        # Loop through each fold
        for train_index, test_index in rkf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Initialize a Random Forest model
            with joblib.parallel_backend("dask"):
                rf_model = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, random_state=repetition)

                # Fit the Random Forest model
                rf_model.fit(X_train, y_train)

                # Calculate permutation feature importance
                permutation_importance_result = permutation_importance(rf_model, X_test, y_test, n_repeats=10, scoring='roc_auc')

                # Accumulate feature importance scores
                return permutation_importance_result.importances_mean

    # Use Dask parallelization to distribute repetitions across workers
    repetitions = range(100)
    feature_importance_scores = dask_client.map(process_repetition, repetitions)

    # Compute results
    feature_importance_scores = dask.compute(*feature_importance_scores)

    # Average the feature importance scores across repetitions
    average_feature_importance_scores = np.mean(feature_importance_scores, axis=0)

    # Get the feature indices sorted by importance
    sorted_feature_indices = np.argsort(average_feature_importance_scores)[::-1]

    # Output file
    output_file = "feature_importance_ranking.txt"
    with open(output_file, 'w') as file:
        file.write("Feature Importance Ranking:\n")
        for i in sorted_feature_indices:
            file.write(f"Feature {i}: {average_feature_importance_scores[i]}\n")

