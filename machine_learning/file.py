from xgboost import XGBClassifier
import numpy as np
import pandas as pd
# read data
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data['data'], data['target'], test_size=.2)
# create model instance
bst = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')
# fit model
bst.fit(X_train, y_train)
# make predictions
preds = bst.predict(X_test)
preds_p=pd.DataFrame(preds)
data_p=pd.DataFrame(data)
data_p.to_csv("data.csv", index =False)
preds_p.to_csv("result_x.csv",index=False)
print(preds)
