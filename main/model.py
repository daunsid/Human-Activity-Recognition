import time
import path
import data
import preprocessing
from sklearn.utils import shuffle

import sklearn.ensemble as ensemble
import sklearn.neighbors as neighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel

dpath = path.Path("../datasets")

datasets = data.loadData(dpath)

(train_samples,train_label), (test_samples, test_label) = datasets

infer_samples,infer_label = data.loadInferData(dpath/"test.csv")

x_train, y_train = preprocessing.normalize_data(train_samples, train_samples), preprocessing.encodeData(train_label, train_label)

x_test, y_test = preprocessing.normalize_data(train_samples, test_samples), preprocessing.encodeData(train_label,
                                                                                                     test_label)

x_ifer, y_ifer = preprocessing.normalize_data(train_samples, infer_samples), preprocessing.encodeData(train_label,
                                                                                                      infer_label)


# Baseline Model

trees = 500
max_feat = 10
max_depth = 50
min_sample = 2

clf = ensemble.RandomForestClassifier(n_estimators=trees,
                             max_features=max_feat,
                             max_depth=max_depth,
                             min_samples_split= min_sample, random_state=0)

def getAccuracy(pre,ytest): 
    count = 0
    for i in range(len(ytest)):
        if (ytest[i]==pre[i]).all(): 
            count+=1
    acc = float(count)/len(ytest)
    return acc

def evaluate_model(trainx, trainy, testx, testy, model):
    start = time.time()
    model.fit(trainx, trainy)
    end = time.time()
    print("Execution time for building the Tree is: %f"%(float(end)- float(start)))
    
    #Evaluate the model performance for the test data
    
    pred = model.predict(testx)
    
    accuracy = getAccuracy(pred, testy)
    return "Accuracy of model before feature selection is %.2f"%(100*accuracy)

evaluate_model(x_train, y_train, x_test, y_test, clf)
