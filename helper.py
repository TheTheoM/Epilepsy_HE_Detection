import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import os

def read_data_file(date_file_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    EEG_CSV_dir = f'{current_dir}/{date_file_name}'
    ESR = pd.read_csv(EEG_CSV_dir) 
    return ESR

def graph_seizure_labels(seizure_labels):
    sn.countplot(
        x=seizure_labels,
        hue=seizure_labels,
        palette={0: 'steelblue', 1: 'orange'},
        legend=False
    )
    plt.xticks([0, 1], ['Non-Seizure', 'Seizure'])
    plt.title("Seizure vs Non-Seizure Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of Samples")
    plt.show()

