#encoding: utf-8

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np
import time
from sklearn.externals import joblib


# from data_processer import DataProcessor
# data = DataProcessor("/train_dir")
# feature_vectors, bilabels = data.get_dataset()
# feature_vectors = np.array(feature_vectors)
# bilabels = np.array(bilabels)


data_dir = '/train_dir/'

from data_processer2 import Data
data = Data(data_dir+'features1.csv',data_dir+'features2.csv')
feature_vectors = data.norm_features
bilabels = data.bilabels

clf = RandomForestClassifier()
clf.fit(feature_vectors ,bilabels)


results = cross_val_score(clf, feature_vectors, bilabels, cv=10)
print(results)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))



joblib.dump(clf, 'rf.pkl')
