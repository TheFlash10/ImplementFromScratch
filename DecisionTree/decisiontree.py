# Max height of tree is 5

import pandas as pd
import numpy as np
buff = np.finfo(float).eps
import random

def entropy(data):
    target = data.columns[-1]
    entropy = 0
    values = pd.unique(data[target])
    for value in values:
        prob = data[target].value_counts()[value]/len(data[target])
        entropy += -prob * (np.log(prob))    
    #print(entropy)

    return entropy

def entropy_attribute(data,attribute):
    target = data.columns[-1]
    
    values = pd.unique(data[attribute]) 
    targets = pd.unique(data[target])
    entropy = 0
    for value in values:
        temp = 0
        for t in targets:
            n = len(data[attribute][data[attribute]==value][data[target]==t])
            d = len(data[attribute][data[attribute]==value])
            f = n/(d+buff)  # buff added to not make denominator zero
            temp += -f * (np.log(f + buff))
            
        entropy += (d/len(data))*temp
        #print(entropy)
        return abs(entropy)
            
def best(data):
    entopy_attr = []
    attributes = list(data.columns)[:-1]
    l = [] 
    for attribute in attributes:
        #print(attribute)
        l.append(entropy(data) - entropy_attribute(data,attribute))
    
    #print(ig)
    return data.columns[:-1][np.argmax(l)]


def get_table(data, node,value):
    return data[data[node] == value].reset_index(drop=True)
            
def makedtree(data,height,dtree=None): 
    target = data.columns[-1]
    #print(target)
    node = best(data)
    
    values = np.unique(data[node])
    
    if dtree is None:                    
        dtree={}
        dtree[node] = {}

    for value in values:
        
        table = get_table(data,node,value)
        classes,count = np.unique(table[target],return_counts=True)                        
        if len(count)==1 or height==5:
            if(len(count) == 1):
                dtree[node][value] = classes[0]                                                                    
            else:            
                dtree[node][value] = classes[np.argmax(count)]                                                   
        else:               
            dtree[node][value] = makedtree(table,height+1) 
    
    return dtree        
    

data1 = pd.read_csv('weather.csv')
data2 = pd.read_csv('train.csv')


#data1.head(4)

#data2.head(4)


# Building dtree 1
tree1 = makedtree(data1,0,None)
#print(tree1)

# Building dtree 2
tree2 = makedtree(data2,0,None)
#print(tree2)


def print_tree(dtree,width):
    
    for key in dtree.keys():
        for value in list(dtree[key].keys()):
            tree_value = dtree[key][value]
            if(type(tree_value) is dict):                
                print("| "*width + key + ' = ' + str(value))
            else:
                print("| "*width + key + ' = ' + str(value),end = '')
                
            #print(tree_value)
            if(type(tree_value) is dict):
                print_tree(tree_value,width+1)
            else:
                print(": " + tree_value)



print("Tree 1")
print_tree(tree1,0)
print()

print("Tree 2")
print_tree(tree2,0)
print()

# Prediction

def predict(data,dtree):
    
    prediction = -1
    for value in dtree.keys():
        recurse_value = data[value]
        
        tree_value = dtree[value][recurse_value]
        if type(tree_value) is dict:
            #print(tree_value)
            prediction = predict(data,tree_value)
        
        else:
            prediction = tree_value
        
    if(prediction == -1):
        return None
    
    return prediction
    
print()
print("Data 1 Predictions")
p = 0
n = 0
for j in range(0,5):
    random_no = random.randint(0,len(data1)-1)
    print("Test Input " + str(j+1))
    print(data1.iloc[random_no][:-1])
    print('\033[1m',end='')
    pred = predict(data1.iloc[random_no][:-1],tree1)
    print("Prediction = {}".format(pred))
    print('\033[0m')    
    if(data1.iloc[random_no][-1] == pred):
          p += 1
    else:
          n += 1
    print()
print('Accuracy = {}'.format(p/(p+n)))          
print()

print("Data 2 Predictions")
p = 0
n = 0
for j in range(0,5):
    random_no = random.randint(0,len(data2)-1)
    print("Test Input " + str(j+1))
    print(data2.iloc[random_no][:-1])
    print('\033[1m',end='')
    pred = predict(data2.iloc[random_no][:-1],tree2)    
    print("Prediction = {}".format(pred))
    print('\033[0m')    
    if(data2.iloc[random_no][-1] == pred):
          p += 1
    else:
          n += 1                    
    print()
    #print(p,n)

print('Accuracy = {}'.format(p/(p+n)))          


# Accuracy coming high because we are using random examples from the training dataset

