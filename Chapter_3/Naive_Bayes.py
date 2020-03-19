import numpy as np
import pandas as pd
from NB_preprocessing import preprocessing
from pprint import pprint

class NB():
    def __init__(self):
        self.data = None
        self.labels = None
        self.vocab = []
        self.num_classes = None
        self.p_ci = {}
        self.p_cond = {}

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


        self.prob_Classes()

        for cls in set(self.labels):
            current_class = cls
            vec_sum_class = np.zeros(shape=self.data[0].shape)
            vec_sum_total = 0
            #vec_sum_total = np.zeros(shape=self.data[0].shape)
            for vec_num in range(len(self.data)):
                if(self.labels[vec_num]==current_class):
                    vec_sum_class+=self.data[vec_num]
                    vec_sum_total+=sum(self.data[vec_num])

            cond_prob = (vec_sum_class)*(float)(self.p_ci[cls])/vec_sum_total

            self.p_cond[cls] = cond_prob

    def predict(self,test_vec):
        self.calc_NB_prob()
        final_prob = 0
        cond_class = None
        for cls in self.p_cond.keys():
            cond_vec = self.p_cond[cls]
            mul = np.multiply(cond_vec,test_vec)
            cond_prob = sum(mul)
            if(cond_prob>final_prob):
                final_prob = cond_prob
                cond_class = cls
        return final_prob,cond_class






def main():
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
    #pprint(NBClassifier.calc_NB_prob())



    pred_prob,pred_class = NBClassifier.predict(myDat[5])
    print(pred_class)
if __name__=="__main__":
    main()