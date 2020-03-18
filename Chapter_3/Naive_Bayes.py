import numpy as np
import pandas as pd
from NB_preprocessing import preprocessing

class NB():
    def __init__(self):
        self.data = None
        self.labels = None
        self.vocab = []
        self.num_classes = None
        self.p_ci = {}

    def fit(self, data, labels):
        self.data = data
        self.labels = labels

    def prob_Classes(self):
        seen_list = []
        for label in self.labels:
            if(label not in seen_list):
                count = self.labels.count(label)
                seen_list.append(label)
                self.p_ci[label] = count/len(self.labels)

    def calc_NB_prob(self):
        vec_sum_class = np.zeros(shape=self.data[0].shape)
        vec_sum_total = np.zeros(shape=self.data[0].shape)

        for vec in self.data:
            vec_sum_total+=vec

        for cls in set(self.labels):
            current_class = cls
            for vec_num in range(len(self.data)):
                if(self.labels[vec_num]==current_class):
                    vec_sum_class+=self.data[vec_num]
                







postingList=[['my', 'dog', 'has', 'flea', \
'problems', 'help', 'please'],
['maybe', 'not', 'take', 'him', \
'to', 'dog', 'park', 'stupid'],
['my', 'dalmation', 'is', 'so', 'cute', \
'I', 'love', 'him'],
['stop', 'posting', 'stupid', 'worthless', 'garbage'],
['mr', 'licks', 'ate', 'my', 'steak', 'how',\
'to', 'stop', 'him'],
['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
classVec = [0,1,0,1,0,1]

prep = preprocessing()
prep.fit(postingList,classVec)
prep.create_vocab()
myDat = prep.vectorize()
#print(myDat)

NBClassifier = NB()
NBClassifier.fit(myDat,classVec)
NBClassifier.prob_Classes()
print(NBClassifier.calc_NB_prob())