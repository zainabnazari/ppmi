import pandas as pd
import dask
import dask.dataframe as dd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFECV
from xgboost import XGBClassifier
import time
from pathlib import Path

path2 = Path("/home/znazari/data")
path3 = Path("/home/znazari/data/open_proteomic/")

# Read proteomic annotation
proteomic_annotation = pd.read_csv(path3 / "PPMI_Project_151_pqtl_Analysis_Annotations_20210210.csv", delimiter=',')

# Read and concatenate proteomic data
base_file_name = "Project_151_pQTL_in_CSF_{}_of_7_Batch_Corrected_.csv"
num_files = 7
dfs = []

for file_index in range(1, num_files + 1):
    file_name = base_file_name.format(file_index)
    file_path = path3 / file_name

    if file_path.is_file():
        df = pd.read_csv(file_path, delimiter=',')
        dfs.append(df)
    else:
        print(f"File {file_name} not found.")

result_df = pd.concat(dfs, ignore_index=True)

# Filter out patients diagnosed as Prodromal
result_df = result_df[result_df['COHORT'] != 'Prodromal']

# Pivot the DataFrame
result_pivot = result_df.pivot(index='TESTNAME', columns='PATNO', values='TESTVALUE')

# Merge diagnosis information
patient_diagnosis_df = result_df[['PATNO', 'COHORT']].drop_duplicates()
merged_df = pd.merge(result_pivot.T, patient_diagnosis_df, on='PATNO')

# Separate features (X) and target variable (y)
X = merged_df.drop(['PATNO', 'COHORT'], axis=1)
y = merged_df['COHORT']

# Convert diagnosis labels to numeric values using LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets using Dask
X_train, X_test, y_train, y_test = train_test_split(dd.from_pandas(X, npartitions=10), y_encoded, test_size=0.3, random_state=42)

# Initialize and train the Random Forest model for feature selection
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Initialize RFECV with stratified k-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
selector = RFECV(estimator=rf_model, step=1, cv=cv, scoring='roc_auc')

start_time = time.time()

# Fit RFECV on training data using Dask
with dask.config.set(scheduler='threads'):  # You can change 'threads' to 'processes' for parallelism
    selector = selector.fit(X_train, y_train)

# Transform the data to include only selected features
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

# Initialize and train the XGBoost model on the selected features
xgb_model = XGBClassifier()
xgb_model.fit(X_train_selected.compute(), y_train.compute())

# Make predictions on the testing set
y_pred_proba = xgb_model.predict_proba(X_test_selected.compute())[:, 1]

# Calculate AU-ROC score
au_roc_score = roc_auc_score(y_test.compute(), y_pred_proba)

end_time = time.time()
elapsed_time = end_time - start_time

# Print the AU-ROC score and elapsed time
print(f'AU-ROC Score: {au_roc_score}')
print(f'Elapsed Time: {elapsed_time} seconds')

