import pandas as pd
import numpy as np
import os
import glob
import functools
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, accuracy_score
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SelectFromModel
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFECV
from xgboost import XGBClassifier
import time
import joblib
from joblib import Parallel, delayed
# ### Data for Parkinson ###



path2 = Path("/home/znazari/data")# where the output data will be saved at the end.
path3 = Path("/home/znazari/data/open_proteomic/")
proteomic_annotation = pd.read_csv(path3/"PPMI_Project_151_pqtl_Analysis_Annotations_20210210.csv",delimiter=',')
proteomic_annotation;


# In[4]:


proteomic = pd.read_csv(path3/"Project_151_pQTL_in_CSF_1_of_7_Batch_Corrected_.csv",delimiter=',')
proteomic


# In[5]:


# Specify the base file name and path
base_file_name = "Project_151_pQTL_in_CSF_{}_of_7_Batch_Corrected_.csv"

# Number of files
num_files = 7

# List to store DataFrames
dfs = []

# Loop through the file indices and read each file
for file_index in range(1, num_files + 1):
    file_name = base_file_name.format(file_index)
    file_path = path3 / file_name

    # Check if the file exists before attempting to read it
    if file_path.is_file():
        # Read the CSV file and append it to the list
        df = pd.read_csv(file_path, delimiter=',')
        dfs.append(df)
    else:
        print(f"File {file_name} not found.")



# Concatenate all DataFrames into a single DataFrame
result_df = pd.concat(dfs, ignore_index=True)

# Filter out patients diagnosed as Prodromal
result_df = result_df[result_df['COHORT'] != 'Prodromal']

# Pivot the DataFrame to get the desired format
result_pivot = result_df.pivot(index='TESTNAME', columns='PATNO', values='TESTVALUE')

patient_diagnosis_df = result_df[['PATNO', 'COHORT']].drop_duplicates()


# Assuming you have two DataFrames: result_pivot and patient_diagnosis_df

# Merge the two DataFrames on 'PATNO'
merged_df = pd.merge(result_pivot.T, patient_diagnosis_df, on='PATNO')

# Separate features (X) and target variable (y)
X = merged_df.drop(['PATNO', 'COHORT'], axis=1)
y = merged_df['COHORT']

# Convert diagnosis labels to numeric values using LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Initialize and train the Random Forest model for feature selection
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Initialize RFECV with repeated stratified k-fold cross-validation
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
selector = RFECV(estimator=rf_model, step=1, cv=cv, scoring='roc_auc')

start_time=time.time()	
# Fit RFECV on training data
#selector = selector.fit(X_train, y_train)
# Define a function for parallel fitting
def fit_rfecv(X_train, y_train, selector):
    return selector.fit(X_train, y_train)
# Parallelize the RFECV fitting process using joblib
with joblib.parallel_backend('multiprocessing'):  # Change 'threading' to 'multiprocessing' for parallelization using multiple processes
    selector = fit_rfecv(X, y, selector)

# Transform the data to include only selected features
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

# Initialize and train the XGBoost model on the selected features
xgb_model = XGBClassifier()
xgb_model.fit(X_train_selected, y_train)

# Make predictions on the testing set
y_pred_proba = xgb_model.predict_proba(X_test_selected)[:, 1]

# Calculate AU-ROC score
au_roc_score = roc_auc_score(y_test, y_pred_proba)
end_time=time.time()
elapsed_time=end_time-start_time
# Print the AU-ROC score
print(f'AU-ROC Score: {au_roc_score}')
print(f'elapsed time: {elapsed_time}')

