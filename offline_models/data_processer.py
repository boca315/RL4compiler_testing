#encoding: utf-8
import numpy as np
import torch
import os, gc
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler




batch_size = 32 # 64
iftest = True


categories_dir={
    'pass':0,
    'fail':1,
    # 'crash':1,
    # 'wrongcode':2
}

def feature2vector(feature):
    # print("feature is")
    # print(feature)
    # print(len(feature))
    feature_vector = []
    for index in range(len(feature) - 1):
        # print(feature[index])
        # print(feature[index].split(" ")[-1])
        feature_vector.append(int(feature[index].split(" ")[-1]))
    feature_vector = np.array(feature_vector)
    fv_normed = feature_vector / feature_vector.max(axis=0)  # nomalization
    return fv_normed.tolist()

def exccmd2arr(cmd):
    p = os.popen(cmd, "r")
    rs = []
    line = ""
    while True:
        line = p.readline()
        if not line:
            break
        rs.append(line)
    gc.collect() # free mem
    return rs
def exccmd2string(cmd):
    p = os.popen(cmd, "r")
    rs = []
    line = ""
    while True:
        line = p.readline()
        if not line:
            break
        rs.append(line)
    ss = ''
    for item in rs:
        ss += item
    gc.collect()  # free mem
    return ss


class DataProcessor:
    def __init__(self, root_dir):
        self.train_data_dir = root_dir + "/train/"
        self.test_data_dir = root_dir+"/test/"


    def get_dataset(self):
        feature_vectors = []
        bilabels = []
        label_ids = []

        # train dir
        size = len(exccmd2arr('ls '+ self.train_data_dir))
        perm = range(1, size + 1)
        bias = 13118
        for i in perm:
            if(iftest):
                i += bias
                if i>30+bias:
                    break
            if(i% 100 == 0):
                print('<<<<<< '+ str(i) +' <<<<<< ')
            cmd_feature = "ls " + self.train_data_dir + " | grep \'\\." + str(i) + "\\.\' | xargs -I file cat " + self.train_data_dir + "file"
            cmd_label = "ls " + self.train_data_dir + " | grep \'\\." + str(i) + "\\.\' "

            feature = exccmd2arr(cmd_feature)
            fv = feature2vector(feature)
            label_info = exccmd2string(cmd_label).split('.')
            label = label_info[0]
            label_id = label_info[1]
            label_type_id = categories_dir[label]


            feature_vectors.append(fv)
            bilabels.append(label_type_id)
            label_ids.append(label_id)
            if (iftest):
                i -= bias

        # test dir
        s1 = len(feature_vectors)
        size = len(exccmd2arr('ls ' + self.test_data_dir))
        perm = range(1, size + 1)
        bias = 933
        for i in perm:
            if (iftest):
                i += bias
                if(i>30+bias):
                    break
            if(i% 100 == 0):
                print('<<<<<< ' + str(i+s1/100) + ' <<<<<< ')

            cmd_feature = "ls " + self.test_data_dir + " | grep \'\\." + str(
                i) + "\\.\' | xargs -I file cat " + self.test_data_dir + "file"
            cmd_label = "ls " + self.test_data_dir + " | grep \'\\." + str(i) + "\\.\' "

            feature = exccmd2arr(cmd_feature)
            fv = feature2vector(feature)
            label_info = exccmd2string(cmd_label).split('.')
            label = label_info[0]
            label_id = label_info[1]
            label_type_id = categories_dir[label]
            # fv = torch.Tensor([fv])

            feature_vectors.append(fv)
            bilabels.append(label_type_id)
            label_ids.append(label_id)
            if (iftest):
                i -= bias

        logfile = open('logfile', 'w')
        logfile.write('data distribution: pass '+str(bilabels.count(categories_dir['pass']))
                         +' fail '+str(bilabels.count(categories_dir['fail']))+'\n')
        for index  in range(len(bilabels)):
            if index%10==0:
                logfile.write('\n')
            logfile.write(str(label_ids[index])+'('+str(bilabels[index])+'),')
        logfile.write('\n')
        logfile.flush()
        logfile.close()
        return feature_vectors, bilabels


    def get_torch_dataset(self):
        fvs, bilabels = self.get_dataset()
        feature_vectors = []
        for fv in fvs:
            feature_vectors.append(torch.Tensor([fv]))
        feature_vectors = torch.cat(feature_vectors, dim=0)
        bilabels = torch.tensor(bilabels)

        print("begin get_data_loaders")
        from torch.utils.data import TensorDataset
        # 将输入数据合并为 TensorDataset 对象

        dataset = TensorDataset(feature_vectors, bilabels)
        return dataset


    '''construct training and validation sets'''
    def split(self,dataset):
        # calculate size of training set and validation set
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        from torch.utils.data import random_split


        # 按照数据大小随机拆分训练集和测试集
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        print("begin get_data_loaders -train")
        # construct Dataloader，shuffule training set
        train_dataloader = DataLoader(
            train_dataset,
            sampler=RandomSampler(train_dataset),  # 随机小批量
            batch_size=batch_size  # 以小批量进行训练
        )

        print("begin get_data_loaders -validation")
        # 验证集不需要随机化，这里顺序读取就好
        validation_dataloader = DataLoader(
            val_dataset,  # 验证样本
            sampler=SequentialSampler(val_dataset),  # 顺序选取小批量
            batch_size=batch_size
        )

        print("done get_data_loaders")

        return train_dataloader, validation_dataloader

