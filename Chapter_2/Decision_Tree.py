import numpy as np 
import math
from pprint import pprint
def createDataSet():
    dataSet = [["Sunny", "Mood","yellow"],
    ["Sunny", "Mood","red"],
    ["Sunny", "No Mood","yellow"],
    ["Rainy", "Mood", "red"],
    ["Rainy", "Mood", "yellow"]]
    labels = ["yes","yes","no","no","no"]
    return dataSet, labels

myDat,labels = createDataSet()
#print(myDat)

class DTreeClassifier:
    ######Default Constructor####
    def __init__(self):
        
        self.data = None
        self.labels = None
        self.min_entropy = None
        self.levels = 0
        self.tree = {}

    ####Calculates Entropy using Shannon Entropy Formula####
    def CalcEntropy(self,data):
        target_variable = data
        #print(len(target_variable))
        den = (float)(len(target_variable))
        unique_vals = list(set(target_variable))
        #print(data_point,unique_vals)
        entropy = 0
        for i in unique_vals:
            #print(data_point,i,list(target_variable).count(i))
            count_val = (float)(list(target_variable).count(i))
            
            #print(data_point,(float)(count_val/len(target_variable)))
            entropy-=(count_val/den)*(math.log(count_val/den)/math.log(2))
        #print(data_point,entropy)
        return entropy

    ### Fit the object to data ####
    def fit(self,data,labels):
        self.data = np.array(data)
        self.labels = np.array(labels)

    #### Finds the entropy of a particular column ####
    def column_entropy(self,feature_index):
        ###Splits data and labels based on the feature index###
        unique_vals = list(set(self.data[:,feature_index]))
        final_entropy = 0
        den = (float)(len(self.data))
        #print(unique_vals)
        for data_point in unique_vals:
            cond_arr = self.data[:,feature_index]==data_point
            
            masked = self.labels[np.array(cond_arr)]
        
            num = (float)(len(masked))
            entropy_feature = self.CalcEntropy(masked)
            #print(data_point,entropy_feature)
            final_entropy+=(num/den)*entropy_feature
        return final_entropy    

    ### Finds the index of column to split on in the entire dataset####
    def find_split_index(self):
        final_entropy = None
        final_index = None
        
        for i in range(len(self.data[0])):
            temp_final_entropy = self.column_entropy(i)
            #print("I",i)
            #print(i,temp_final_entropy)
            if(final_entropy==None or temp_final_entropy<final_entropy):
                final_entropy = temp_final_entropy
                final_index = i
            else:
                continue
            
        self.min_entropy = final_entropy
        
        
        return final_index
    #### Splits on the desired index ####
    def split_on_index(self,split_index):
        unique_vals = list(set(self.data[:,split_index]))
        glob_mask_x = []
        glob_mask_y = []
        glob_dps = []
        dc = {}

        for data_point in unique_vals:
            cond_arr = self.data[:,split_index]==data_point

            masked_x = self.data[cond_arr]
            masked_y = self.labels[cond_arr]
            #print(np.delete(masked_x,split_index,1))

            masked_x = np.delete(masked_x,split_index,1)
            if(masked_x.size!=0):
                dc[(data_point)] = {"x":masked_x,"y":masked_y}
            else:
                dc[data_point] = {"labels":masked_y}

            #glob_mask_x.append(masked_x)
            #glob_mask_y.append(masked_y)
        self.levels+=1
        
        return dc,masked_x

    def recursive_build(self):
        index = self.find_split_index()
        split_dict,masked_x = self.split_on_index(index)
        if(masked_x.size==0):
            return split_dict
        else:
            for key in split_dict.keys():
                dtree = DTreeClassifier()
                dtree.fit(split_dict[key]["x"],split_dict[key]["y"])
                temp_dict = dtree.recursive_build()
                split_dict[key] = temp_dict
            self.tree = split_dict
            return split_dict
    def predict(self,features,trav_dict=None):
        if(trav_dict==None):
            trav_dict=self.tree

        for feature_name in features:

            for key in trav_dict.keys():


                if(key==feature_name):
                    print(key,feature_name)
                    features.remove(feature_name)
                    if("labels" in trav_dict[key].keys()):

                        return trav_dict[key]["labels"]
                    else:
                        pred = self.predict(features,trav_dict[key])
                        return pred
                else:
                    return "Not enough data to make prediction"






    
dtree = DTreeClassifier()
import pandas as pd
data = pd.read_csv("lenses.txt",sep="\t")
myDat = np.array(data.iloc[:,:4])
labels = np.array(data.iloc[:,-1])
print(myDat)
dtree.fit(myDat,labels)
s_dict = dtree.recursive_build()
pprint(s_dict)



