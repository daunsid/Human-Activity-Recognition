import os
import path
import pickle

import numpy as np
import pandas as pd


def loadData(dpath):
    #data_path=path.Path(dpath)
    train_set = pd.read_csv(dpath/"train.csv")
    test_set = pd.read_csv(dpath/"test.csv")
    
    return (train_set.iloc[:, :-2], train_set.iloc[:, -1]), (test_set.iloc[:, :-2], test_set.iloc[:, -1])

def loadInferData(path):
    df = pd.read_csv(path)
    sdf = df[df["Activity"] == "STANDING"]
    wdf = df[df["Activity"] == "WALKING"]
    swdf = pd.concat([sdf, wdf])
    swdf = swdf.set_index(np.arange(len(swdf)))
    
    return swdf.iloc[:, :-2], swdf.iloc[:, -1] 