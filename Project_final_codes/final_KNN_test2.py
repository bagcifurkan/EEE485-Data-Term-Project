import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import csv
import time

start = time.time()


train_data = pd.read_csv('train.zip')
test_data = pd.read_csv('test.zip')

A = int(len(test_data))-1
B = len(train_data)-1


prediction = pd.DataFrame(index=range(A), columns=range(1), dtype='float64')
predictions = pd.DataFrame(index=range(A), columns=range(1), dtype='float64')
test_y = pd.DataFrame(index=range(A), columns=range(1))

distance = pd.DataFrame(index=range(B), columns=range(A))
output = pd.DataFrame(index=range(B), columns=range(A))

MSEs = pd.DataFrame(index=range(1), columns=range(1))
MSEs.columns = [0]

test_y[0] = pd.read_csv('test_Y.zip')
output = pd.read_csv('train_Y.zip')
a = 0
for neighbor in [3]:
    for i in range(A):

        distance = pd.DataFrame(train_data - test_data.loc[i])
        distance[i] = pd.DataFrame(((distance * distance).sum( axis = 1))**(1/2),columns=['mean'])

        dist_out = pd.concat([distance[i], output], axis='columns')
        dist_out.columns = ['distance','output']
        dist_out = dist_out.sort_values(by='distance')
        dist_out = dist_out.reset_index()

        count = 0;
        for k in range(neighbor):
            if dist_out['output'][k] == 1:
                count +=1

        if count > neighbor*0.5 :
            prediction.loc[i] = 1
        else:
            prediction.loc[i] = 0

        predictions[a] = prediction



    error = test_y - prediction
    predictions[1] = test_y
    MSE = (error.T.dot(error))/A
    MSEs[a] = MSE
    print(MSE)
    a += 1


predictions[0] = predictions[0].astype(int)

true_pos = pd.DataFrame(index=range(A), columns=range(1))
false_pos = pd.DataFrame(index=range(A), columns=range(1))
true_neg = pd.DataFrame(index=range(A), columns=range(1))
false_neg = pd.DataFrame(index=range(A), columns=range(1))

tp = 0
fp = 0
tn = 0
fn = 0
for i in range(A):
    if (predictions[0].loc[i]==1) & (predictions[1].loc[i]==1):
        tp += 1
    #true_pos[0].loc[i] = (predictions[0].loc[i]==1) & (predictions[1].loc[i]==1)
    #true_pos_per = true_pos.value_counts()[1]/A
    if (predictions[0].loc[i]==1) & (predictions[1].loc[i]==0):
        fp += 1
    #false_pos[0].loc[i] = (predictions[0].loc[i]==0) & (predictions[1].loc[i]==1)
    #false_pos_per = false_pos.value_counts()[1] / A
    if (predictions[0].loc[i] == 0) & (predictions[1].loc[i] == 0):
        tn += 1
    #true_neg[0].loc[i] = (predictions[0].loc[i] == 0) & (predictions[1].loc[i] == 0)
    #true_neg_per = true_neg.value_counts()[0] / A
    if (predictions[0].loc[i] == 0) & (predictions[1].loc[i] == 1):
        fn += 1
    #false_neg[0].loc[i] = (predictions[0].loc[i] == 1) & (predictions[1].loc[i] == 0)
    #false_neg_per = false_neg.value_counts()[1] / A


print(true_pos.value_counts())
print(false_pos.value_counts())
print(true_neg.value_counts())
print(false_neg.value_counts())

#df.groupby('a').count()
F1 = tp / (tp + 1/2 * (fp + fn ))
print(F1)

end = time.time()
print(end-start)