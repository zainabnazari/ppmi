import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

# Initialize repeated stratified K-fold cross-validation
n_splits = 10
n_repeats = 20
rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

# X: feature matrix, y: the target variable

# Define the parameters for the Random Forest
n_estimators = 1000
max_features = int(np.sqrt(X.shape[1]))  # Square root of the total number of features

# Initialize arrays to store feature importance scores
feature_importance_scores = np.zeros((100, X.shape[1]))  # 100 repetitions, n_features

# Loop through each repetition
for repetition in range(100):
    
    # Loop through each fold
    for train_index, test_index in rkf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Initialize a Random Forest model
        rf_model = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features)

        # Fit the Random Forest model, we fit the X training sample with its associated y labels.
        rf_model.fit(X_train, y_train)

        # Calculate permutation feature importance
        permutation_importance_result = permutation_importance(rf_model, X_test, y_test, n_repeats=10, random_state=repetition, scoring='roc_auc')

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
