#encoding: utf-8
import pandas as pd
import numpy as np

iftest = True


categories_dir={
    'pass':0,
    'crash':1, # fail
    'wrongcode':1, # fail
    'generation':-1, # invalid
    'compile':-1,
    'execute':-1,
}

class Data:
    def __init__(self, *csv_files):
        self.dfs = []
        for csv_file in csv_files:
            # self.dfs.append(pd.read_csv(csv_file,header=None,error_bad_lines=False))
            self.dfs.append(pd.read_csv(csv_file,header=None))
        self.dfs = pd.concat(self.dfs, axis=0,ignore_index=True)

        labels = self.dfs[[1]].values.tolist()
        self.bilabels = [categories_dir[cat[0]] for cat in labels]

        features = []
        for indexs in self.dfs.index:
            feature = self.dfs.loc[indexs].values[2:-1].tolist()
            # print(feature)

            features.append(feature)
        self.bilabels = np.array(self.bilabels)
        features = np.array(features)
        self.norm_features = self.norm(features)
        self.bilabels = np.array(self.bilabels)
        self.norm_features = np.array(self.norm_features)
        # print(features)
        # print(len(features))
        # print(self.norm_features)

    def norm(self,x):

        x_normed = x / x.max(axis=0)
        where_are_nan = np.isnan(x_normed) # xmax=0, div by 0
        x_normed[where_are_nan] = 0
        return x_normed.tolist()


