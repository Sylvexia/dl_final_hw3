from data_proc import get_data, standardize_data, get_label_names, gen_confusion_matrix, get_class_report

from sklearn.ensemble import RandomForestClassifier
import pandas as pd

data_frame = pd.read_csv('feature_time_48k_2048_load_1.csv')
train, test = get_data(data_frame)

train_label = train.iloc[:, -1].to_numpy()
test_label = test.iloc[:, -1].to_numpy()

train_data = train.iloc[:, :-1].to_numpy()
test_data = test.iloc[:, :-1].to_numpy()

train_data, test_data = standardize_data(train_data, test_data)

model = RandomForestClassifier()
model.fit(train_data, train_label)

train_pred = model.predict(train_data)
test_pred = model.predict(test_data)

dir = "random_forest"

get_class_report(train_pred, train_label, test_pred, test_label, dir)

gen_confusion_matrix(train_label, train_pred, test_label,
                     test_pred, get_label_names(data_frame), 'RandomForest', dir)

# get feature importance
importances = model.feature_importances_

# plot feature importance

import matplotlib.pyplot as plt
import numpy as np
features = data_frame.columns[:-1]
num_features = len(features)

indices = np.argsort(importances)[::-1] # reverse the sequence

sorted_features = []
for k in indices:
    sorted_features=np.append(sorted_features, features[k])

plt.figure(figsize=(16, 8))
plt.title("Feature importances")

plt.bar(range(num_features), importances[indices], color="blue", align="center")

plt.xticks(range(num_features),sorted_features)
plt.ylabel("importance")
plt.xlabel("feature names")
plt.savefig(f'{dir}/feature_importance.png')
plt.show()

from joblib import dump, load

dump(model, f'{dir}/model.joblib')