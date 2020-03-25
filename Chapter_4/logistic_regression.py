import numpy as np
import pandas as pd

class LogReg:
    def __init__(self):
        self.data = None
        self.labels = None
        self.weights = None
        self.lr = 0.00001

    def fit(self,data,labels):
        self.data = data
        self.labels = labels

    def train(self,epochs):
        bias = np.ones(shape=(self.data.shape[0],1))
        self.data = np.append(self.data,bias,axis=1)
        num_vars = len(self.data[0])

        self.weights = np.ones(shape=(1,num_vars))

        for epoch in range(epochs):

            theta_x = self.weights.dot(self.data.transpose())
            h_theta = 1/(1+np.exp(-theta_x))

            loss_per = -(np.multiply(self.labels,np.log(h_theta+1e-7)) + np.multiply((1-self.labels),np.log(((1-h_theta)+1e-7))))/(float)(len(self.labels))
            total_loss = np.sum(loss_per)
            print("LOSS",total_loss)
            ###### WEIGHT UPDATE #####
            updater = h_theta - self.labels
            for i in range(num_vars):
                req_x = self.data[:,i]
                self.weights[:,i] = self.weights[:,i] - self.lr*np.sum(np.multiply(updater,req_x))
    def predict(self,test):
        pred_list = []

        tx_pred = self.weights.dot(test.transpose())
        h_pred = 1 / (1 + np.exp(-tx_pred))
        for i in h_pred.transpose():
            if(i>0.5):
                pred_list.append(1)
            else:
                pred_list.append(0)
        print(h_pred.shape)
        return pred_list

    def check_accuracy(self,pred,true):
        corr=0.0
        incorr=0.0
        for p,t in zip(pred,true):
            if(p==t):
                corr+=1
            else:
                incorr+=1
        return (float)(corr)/(float)(corr+incorr)



def main():
    df = pd.read_csv("test_set.txt", sep="\t")
    df.columns = ["1", "2", "3"]
    lc = LogReg()
    lc.fit(np.array(df.iloc[:,:2],dtype=np.float128),np.array(df.iloc[:,2],dtype=np.float128))
    print(lc.data)
    lc.train(1500)
    print(lc.weights.shape)
    #pred = lc.predict(np.array(df.iloc[:,:2],dtype=np.float128))
    #print(lc.check_accuracy(pred,list(df.iloc[:,2])))

if __name__=="__main__":
    main()