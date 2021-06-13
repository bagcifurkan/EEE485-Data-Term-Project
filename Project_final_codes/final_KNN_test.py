import pandas as pd
import numpy as np
import math
import csv

#data =  pd.read_csv('data_normalized2.zip')
data =  pd.read_csv('train.csv')


""" Defining method for splitting data into k folds"""
def splitting_data(data, k_folds):
    datasets = []
    data_copy = data;
    fold_size = int(len(data) / k_folds)

    for i in range(k_folds-1):
        fold = data_copy.iloc[:fold_size,:]
        datasets.append(fold)
        data_copy = data_copy.iloc[fold_size:]

    datasets.append(data_copy)
    return datasets

k_fold = 5
data = data.sample(frac=1)
#datasets = splitting_data(data,k_fold)

for k_fold_i in range(1):

    #datasets_temp = list(datasets)
    #test_data = datasets_temp[k_fold_i]
    test_data = pd.read_csv("test.csv")
    train_data = pd.read_csv("train.csv")
    #del datasets_temp[k_fold_i]
    #train_data = pd.concat(datasets_temp, axis='rows')

    train_data = train_data.reset_index()
    test_data = test_data.reset_index()
    #above code ads additional feature as index, should be deleted
    del train_data['index']
    del test_data['index']


    A = int(len(test_data))-1
    #A = len(test_data)-1
    B = len(train_data)-1


    prediction = pd.DataFrame(index=range(A), columns=range(1), dtype='float64')
    predictions = pd.DataFrame(index=range(A), columns=range(1), dtype='float64')
    test_y = pd.DataFrame(index=range(A), columns=range(1))

    distance = pd.DataFrame(index=range(B), columns=range(A))
    output = pd.DataFrame(index=range(B), columns=range(A))

    MSEs = pd.DataFrame(index=range(1), columns=range(1))
    MSEs.columns = [0]

    test_y[0] = test_data['y']
    output = train_data['y']
    a = 0
    #for neighbor in [2,3,4,5,10,25,50,75,100,150,250]:
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
    print('fold : '+str(k_fold_i))
    if(k_fold_i==0):
        pred_kfold = predictions
        MSE_kfold = MSEs
    else:
        pred_kfold = pd.concat([pred_kfold, predictions], axis='columns')
        MSE_kfold = pd.concat([MSE_kfold, MSEs], axis='rows')

    #MSE_kfold.columns = ['k=1','k=2','k=5','k=25','k=50']
    #MSE_kfold.plot(x=[1,2,5,25,50], y = ['k=1','k=2','k=5','k=25','k=50'],kind = 'scatter')
    #plt.show()
    #MSE_kfold.describe().loc['mean'])

CV = pd.DataFrame(MSE_kfold.describe().loc['mean'])
CV = pd.concat([CV, pd.DataFrame([3])], axis='columns')
CV.columns = ['CV','k']
""""
CV.plot.line(x='k', y='CV')
print(CV)
print(MSE)
plt.show()

#pred_kfold.to_csv('predictions.zip', index=False, compression=dict(method='zip', archive_name='predictions.csv'))
"""

A = len(test_data)
true_pos = pd.DataFrame(index=range(A), columns=range(1))
false_pos = pd.DataFrame(index=range(A), columns=range(1))
true_neg = pd.DataFrame(index=range(A), columns=range(1))
false_neg = pd.DataFrame(index=range(A), columns=range(1))

tp = 0
fp = 0
tn = 0
fn = 0
for i in range(A-1):
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




#df.groupby('a').count()
F1 = tp / (tp + 1/2 * (fp + fn ))
print(F1)