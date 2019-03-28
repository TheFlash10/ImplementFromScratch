import pandas as pd
import operator




print('Working on the weather data...')
print()

data = pd.read_csv('weather.csv')
data.head(4)




columns = list(data.columns)
columns




target_col = columns[-1]
del columns[-1]
targets = pd.unique(data[target_col])
targets



dict_data1 = {}
dict_data2 = {}

for i in data[target_col]:
    if i not in list(dict_data2.keys()):
        dict_data2[i] = 0
        dict_data2[i] += 1        
            
    else:
        dict_data2[i] += 1
        
total = 0
for value in list(dict_data2.values()):
    total += value

for key,value in dict_data2.items():
    dict_data2[key] /= total
    
#print(dict_data2)



for column in columns:
    data_concat = list(zip(data[column],data[target_col]))
    values = pd.unique(data[column])
    matrix = pd.DataFrame(0,columns=targets,index=values)
    for (value,target) in data_concat:
        matrix.loc[value,target] += 1
    
    #print(matrix)
    for row in list(matrix.columns):
        matrix[row] /= sum(matrix[row])
    
    dict_data1[column] = matrix
    #print(matrix)
    
    #print()



# Working on the two examples

prediction = {}
new_data1 = pd.Series(['sunny','cool','high',True],index=['outlook','temperature','humidity','windy'])
print('Example 1')
print(new_data1)

print()
divide = 0

for target in targets:
    p = 1
    for index in new_data1.index:
        if index in list(dict_data1.keys()):                               
            p = p * (dict_data1[index].loc[new_data1[index],target]) 
        
    p = p * (dict_data2[target])
    divide += p    
    prediction[target] = p    
    
for value in prediction.keys():
    prediction[value] = (prediction[value] + 1)/(divide + len(targets))

    

print('Output Probabilities = {}'.format(prediction))
print()
print('Prediction = {}'.format(max(prediction.items(), key=operator.itemgetter(1))[0]))
print()


prediction = {}
new_data2 = pd.Series(['overcast','hot','high',True],index=['outlook','temperature','humidity','windy'])
print('Example 2')
print(new_data2)

print()
divide = 0

for target in targets:
    p = 1
    for index in new_data2.index:
        if index in list(dict_data1.keys()):                               
            p = p * (dict_data1[index].loc[new_data2[index],target]) 
        
    p = p * (dict_data2[target])
    divide += p    
    prediction[target] = p    
    
    
for value in prediction.keys():
    prediction[value] = (prediction[value] + 1)/(divide + len(targets))                    

print('Output Probabilities = {}'.format(prediction))
print()
print('Prediction = {}'.format(max(prediction.items(), key=operator.itemgetter(1))[0]))


print()
print('Working on other train_data...')
print()

data = pd.read_csv('train_data.csv')



data.head(4)



columns = list(data.columns)


columns


target_col = columns[-1]
del columns[-1]



targets = pd.unique(data[target_col])


targets



dict_data1 = {}
dict_data2 = {}

for i in data[target_col]:
    if i not in list(dict_data2.keys()):
        dict_data2[i] = 0
        dict_data2[i] += 1        
            
    else:
        dict_data2[i] += 1
        
total = 0
for value in list(dict_data2.values()):
    total += value

for key,value in dict_data2.items():
    dict_data2[key] /= total
    
#print(dict_data2)


for column in columns:
    data_concat = list(zip(data[column],data[target_col]))
    values = pd.unique(data[column])
    matrix = pd.DataFrame(0,columns=targets,index=values)
    for (value,target) in data_concat:
        matrix.loc[value,target] += 1

    #print(matrix)
    for row in list(matrix.columns):
        matrix[row] /= sum(matrix[row])
    
    dict_data1[column] = matrix
    #print(matrix)
    
    #print()


# Working on the two examples

prediction = {}
new_data1 = pd.Series(['Weekday','Winter','High','Heavy'],index=['Day','Season','Wind','Rain'])
print('Example 1')
print(new_data1)

print()
divide = 0

for target in targets:
    p = 1
    for index in new_data1.index:
        if index in list(dict_data1.keys()):                               
            p = p * (dict_data1[index].loc[new_data1[index],target]) 
        
    p = p * (dict_data2[target])
    divide += p    
    prediction[target] = p    
    
    
for value in prediction.keys():
    prediction[value] = (prediction[value] + 1)/(divide + len(targets))                    

print('Output Probabilities = {}'.format(prediction))
print()
print('Prediction = {}'.format(max(prediction.items(), key=operator.itemgetter(1))[0]))
print()


prediction = {}
new_data2 = pd.Series(['Saturday','Winter','Normal','Heavy'],index=['Day','Season','Wind','Rain'])
print('Example 2')
print(new_data2)

print()
divide = 0

for target in targets:
    p = 1
    for index in new_data2.index:
        if index in list(dict_data1.keys()):                               
            p = p * (dict_data1[index].loc[new_data2[index],target]) 
        
    p = p * (dict_data2[target])
    divide += p    
    prediction[target] = p    
    
    
for value in prediction.keys():
    prediction[value] = (prediction[value] + 1)/(divide + len(targets))                    

print('Output Probabilities = {}'.format(prediction))
print()
print('Prediction = {}'.format(max(prediction.items(), key=operator.itemgetter(1))[0]))

