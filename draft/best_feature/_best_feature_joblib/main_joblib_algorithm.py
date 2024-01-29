import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.inspection import permutation_importance

def process_repetition(repetition, X, y):
    # Loop through each fold
    for train_index, test_index in rkf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Initialize a Random Forest model
        rf_model = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, random_state=repetition)

        # Fit the Random Forest model
        rf_model.fit(X_train, y_train)

        # Calculate permutation feature importance
        permutation_importance_result = permutation_importance(rf_model, X_test, y_test, n_repeats=10, scoring='roc_auc')

        # Accumulate feature importance scores
        return permutation_importance_result.importances_mean

if __name__ == "__main__":
    print('I am before joblib parallelization')

    # Loading the already preprocessed data:
    path2 = Path("/home/znazari/data")

    # Read data
    ir3_rna_step_vst = pd.read_csv(path2 / 'mydata_Log_CPM_filtered_bact_sex_effect_removed_RIN_covariate.txt', delimiter='\t')
    diagnosis = pd.read_csv(path2 / 'patients_HC_PK_diagnosis.csv')

    # Map diagnosis to zero and one.
    diagnosis['COHORT_DEFINITION'] = diagnosis['COHORT_DEFINITION'].map({'Healthy Control': 0, "Parkinson's Disease": 1})

    # X: feature matrix, y: the target variable
    X = ir3_rna_step_vst.T
    y = diagnosis['COHORT_DEFINITION']

    # Initialize repeated stratified K-fold cross-validation
    n_splits = 10
    n_repeats = 10
    rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

    # Define the parameters for the Random Forest
    n_estimators = 1000
    max_features = int(np.sqrt(X.shape[1]))  # Square root of the total number of features

    # Initialize arrays to store feature importance scores
    feature_importance_scores = np.zeros((1, X.shape[1]))  # 100 repetitions, n_features

    # Use Joblib parallelization to distribute repetitions across workers
    repetitions = range(10)
    feature_importance_scores = joblib.Parallel(n_jobs=-1, backend='threading')(joblib.delayed(process_repetition)(r, X, y) for r in repetitions)

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

