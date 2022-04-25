import data
import path
import numpy as np
import pandas as pd

import sklearn.preprocessing as preprocessing

path = path.Path("../datasets")

def normalize_data(fdata, data):
    norm = preprocessing.MinMaxScaler()
    x_norm = norm.fit(fdata)
    return x_norm.transform(data)

def encodeData(flabel, label):

    lab_enc = preprocessing.LabelEncoder()
    lab_enc = lab_enc.fit(flabel.values)
    data_enc = lab_enc.transform(label.values)
    ohe = preprocessing.OneHotEncoder()
    one_label = ohe.fit(data_enc.reshape(-1,1))
    
    return one_label.transform(data_enc.reshape(-1,1)).toarray()
    