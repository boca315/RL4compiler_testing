#encoding: utf-8

from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import xgboost as xgb
import numpy as np
from sklearn.externals import joblib
# import time


# from data_processer import DataProcessor
# data = DataProcessor("/train_dir")
# feature_vectors, bilabels = data.get_dataset()
# feature_vectors = np.array(feature_vectors)
# nc = len(np.unique(bilabels))
# bilabels = np.array(bilabels)

data_dir = '/train_dir/'

from data_processer2 import Data
data = Data(data_dir+'features1.csv',data_dir+'features2.csv')
# data = Data(data_dir+'features1.csv')

feature_vectors = data.norm_features
bilabels = data.bilabels
nc = len(np.unique(bilabels))


### data split
# x_train,x_test,y_train,y_test = train_test_split(feature_vectors,
#                                                  bilabels,
#                                                  test_size = 0.2,
#                                                  random_state = 3)
### fit model for train data
''' self.classes_ = np.unique(y) 
so no need to specify classes number'''
model = xgb.XGBClassifier( # booster=gbtree
    learning_rate=0.01,
    n_estimators=100,         # 树的个数--100棵树建立xgboost
    max_depth=6,               # 树的深度
    min_child_weight = 1,      # 叶子节点最小权重
    gamma=0.,                  # 惩罚项中叶子结点个数前的参数
    subsample=0.8,             # 随机选择80%样本建立决策树
    colsample_btree=0.8,       # 随机选择80%特征建立决策树
    # objective='binary:logistic', # 指定损失函数
    num_class=nc,
    objective='multi:softmax', # 指定损失函数
    scale_pos_weight=1,        # 解决样本个数不平衡的问题
    random_state=27,            # 随机数
    )
# print(feature_vectors)
# print(bilabels)
# print(len(feature_vectors))
# print(len(bilabels))

model.fit(feature_vectors ,bilabels)

# model.fit(feature_vectors ,bilabels, sample_weight= 3)


results = cross_val_score(model, feature_vectors, bilabels, cv=10)
print(results)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

joblib.dump(model, 'xgb_sample_weight3.pkl')

# clf = joblib.load('xgb.pkl')
