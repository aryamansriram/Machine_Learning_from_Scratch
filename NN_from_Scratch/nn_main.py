import numpy as np
import pandas as pd

from layer import Layer

class NN:
    def __init__(self):
        self.input_shape = None
        self.input = None
        self.labels = None
        self.layers = 0
        self.network = []
        self.layer_shapes = None

    def add(self,num_nodes,inp_shape=None):
        layer = Layer()
        if(self.layers==0):

            layer.weights = np.ones(shape=(num_nodes,inp_shape[0]))
            layer.bias = np.ones(shape=(num_nodes,1))
            self.layers+=1
            self.layer_shapes = layer.weights.shape[0]
            self.network.append(layer)

        else:
            layer.weights = np.ones(shape=(num_nodes,self.layer_shapes))
            layer.bias = np.ones(shape=(num_nodes,1))
            self.layers+=1
            self.layer_shapes = layer.weights.shape[0]
            self.network.append(layer)

    def fit(self,data,labels):
        self.input = data
        self.labels = labels

    def forward(self):
        for record in self.input:
            print(record.shape)
            break

if __name__=="__main__":

    nn = NN()
    data = pd.read_csv("test_set.txt",sep="\t")
    labels = np.array(data.iloc[:,-1])
    dat = np.array(data.iloc[:,:-1])
    nn.fit(dat,labels)
    nn.forward()