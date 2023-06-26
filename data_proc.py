from sklearn.metrics import confusion_matrix, classification_report

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_data(data_frame) -> tuple[list, list]:
    train, test = train_test_split(
        data_frame, test_size=460, stratify=data_frame.iloc[:, -1], random_state=42069)
    return train, test


def standardize_data(train_data, test_data) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)

    test_data = (test_data - scaler.mean_)/np.sqrt(scaler.var_)

    return train_data, test_data


def get_label_names(data_frame):
    return data_frame.fault.unique()


def get_class_report(train_pred, train_label, test_pred, test_label, dir):
    train_report = classification_report(
        y_pred=train_pred, y_true=train_label, output_dict = True)
    test_report = classification_report(
        y_pred=test_pred, y_true=test_label, output_dict = True)
    
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    with open(f'{dir}/result.json', 'w') as f:
        json.dump({"train": train_report, "test": test_report}, f)
    
def gen_confusion_matrix(train_label, train_pred, test_label, test_pred, labels, title, dir):
    train_confu_matrix = confusion_matrix(train_label, train_pred)
    test_confu_matrix = confusion_matrix(test_label, test_pred)

    plt.figure(1, figsize=(18, 8))

    plt.subplot(121)
    sns.heatmap(train_confu_matrix, annot=True, fmt="d",
                xticklabels=labels, yticklabels=labels, cmap="Blues", cbar=False)
    plt.title(f'{title} Training:')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.subplot(122)

    plt.subplot(122)
    sns.heatmap(test_confu_matrix, annot=True,
                xticklabels=labels, yticklabels=labels, cmap="Blues", cbar=False)
    plt.title(f'{title} Testing:')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    if not os.path.exists(dir):
        os.makedirs(dir)

    plt.savefig(f'{dir}/{title}confusion_matrix.png')

    plt.show()
    
def gen_loss_plt(train_loss, test_loss, dir):
    plt.figure(2, figsize=(18, 8))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(test_loss, label='Testing Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Plot')
    
    if not os.path.exists(dir):
        os.makedirs(dir)

    plt.savefig(f'{dir}/loss_plot.png')

    plt.show()
