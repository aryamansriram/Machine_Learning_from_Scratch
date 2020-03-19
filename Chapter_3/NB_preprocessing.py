import numpy as np
import pandas as pd

class preprocessing():
    def __init__(self):
        self.data = None
        self.labels = None
        self.vocab = []

    def fit(self, data, labels):
        self.data = data
        self.labels = labels

    def create_vocab(self):
        temp_vocab = set({})
        for doc in self.data:
            for word in doc:
                temp_vocab.add(word)
        self.vocab = list(temp_vocab)

    def vectorize(self):
        hor_lim = len(self.vocab)
        ver_lim = len(self.data)
        vec_list = np.zeros(shape=(ver_lim,hor_lim))
        for i in range(ver_lim):
            doc = self.data[i]
            for word in doc:
                for j in range(hor_lim):
                    if self.vocab[j] == word:
                        vec_list[i][j]+=1
        self.data = vec_list
        return vec_list

    def vectorize_test(self,data):
        hor_lim = len(self.vocab)
        encoded = np.zeros(shape=(1,hor_lim))
        for index in range(hor_lim):
            for index_l in range(len(data)):
                if(self.vocab[index]==data[index_l]):
                    encoded[:,index]+=1
        return encoded

vec_test = ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
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
print(prep.vectorize_test(vec_test))



