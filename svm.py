from data_proc import get_data, standardize_data, get_label_names, gen_confusion_matrix, get_class_report
from sklearn.svm import SVC

import pandas as pd

data_frame = pd.read_csv('feature_time_48k_2048_load_1.csv')
train, test = get_data(data_frame)

train_label = train.iloc[:, -1].to_numpy()
test_label = test.iloc[:, -1].to_numpy()

train_data = train.iloc[:, :-1].to_numpy()
test_data = test.iloc[:, :-1].to_numpy()

train_data, test_data = standardize_data(train_data, test_data)

svc_model = SVC()
svc_model.fit(train_data, train_label)

train_pred = svc_model.predict(train_data)
test_pred = svc_model.predict(test_data)

dir = "svm"

get_class_report(train_pred, train_label, test_pred, test_label, dir)

gen_confusion_matrix(train_label, train_pred, test_label, test_pred, get_label_names(data_frame), 'SVM', dir)

from joblib import dump, load

dump(svc_model, f'{dir}/model.joblib')