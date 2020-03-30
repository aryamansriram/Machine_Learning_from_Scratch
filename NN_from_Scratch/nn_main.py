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
        self.last_op = None

    def add(self,num_nodes,activation="relu"):
        layer = Layer()
        if(self.layers==0):

            layer.weights = np.ones(shape=(num_nodes,self.input.shape[1]))
            layer.bias = np.ones(shape=(num_nodes,self.input.shape[0]))
            layer.activation = activation
            self.layers+=1
            self.layer_shapes = layer.weights.shape
            self.network.append(layer)

        else:
            layer.weights = np.ones(shape=(num_nodes,self.layer_shapes[0]))
            layer.bias = np.ones(shape=(num_nodes,self.input.shape[0]))
            layer.activation = activation
            self.layers+=1
            self.layer_shapes = layer.weights.shape
            self.network.append(layer)

    def fit(self,data,labels):
        self.input = data
        self.labels = labels

    def fun(self,activation,inp):
        if(activation=="relu"):
            inp[inp<0] = 0
        return inp


    def forward(self):
        inp_flag = 0
        count=0
        for layer in self.network:
            print(count)
            activation = layer.activation
            if(inp_flag==0):
                op = np.dot(layer.weights,self.input.transpose())+layer.bias
                op = self.fun(activation,op)
                print(op.shape)
                inp_flag = 1
                self.last_op = op
            else:

                op = np.dot(layer.weights,self.last_op)+layer.bias
                op = self.fun(activation,op)
                self.last_op = op
                print(op.shape)
            count+=1
        return op
    def calc_loss(self):
        op = self.forward()
        self.labels = self.labels.reshape(-1,1)
        op = op.reshape(-1,1)
        loss = sum(abs(self.labels-op))
        print(loss)
        return loss





if __name__=="__main__":

    nn = NN()
    data = pd.read_csv("test_set.txt",sep="\t")
    labels = np.array(data.iloc[:,-1])
    dat = np.array(data.iloc[:,:-1])
    nn.fit(dat,labels)
    nn.add(6)
    nn.add(4)
    nn.add(1)
    #for lay in nn.network:
    #    print(lay.weights.shape)
    nn.calc_loss()
