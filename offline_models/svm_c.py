#encoding: utf-8
'''https://blog.csdn.net/rocling/article/details/93717335'''
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.model_selection import KFold # 交叉验证所需的子集划分方法
from sklearn.externals import joblib
import time
import numpy as np


logfile = open('timelog', 'a')

logfile.write('begin time: '+str(time.time())+'\n')

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


clf = svm.SVC(kernel='rbf', C=1) # 高斯核 rbf
clf.fit(feature_vectors ,bilabels)

scores = cross_val_score(clf, feature_vectors, bilabels, cv=10)  # cv为迭代次数。


print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))  # 获取置信区间。（也就是均值和方差）
joblib.dump(clf, "svc.pkl")

logfile.write('call other validation: '+str(time.time())+'\n')


logfile.write('end:  '+str(time.time())+'\n')
logfile.close()