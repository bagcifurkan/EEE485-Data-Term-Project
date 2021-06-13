import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import csv
import time

start = time.time()
data = pd.read_csv('train.csv')

data['y'] = data['y'].replace({0: -1})

"""Splitting data method"""


def splitting_data(data, k_folds):
    datasets = []
    data_copy = data;
    fold_size = int(len(data) / k_folds)

    for i in range(k_folds - 1):
        fold = data_copy.iloc[:fold_size, :]
        datasets.append(fold)
        data_copy = data_copy.iloc[fold_size:]

    datasets.append(data_copy)
    return datasets


# k_fold = 10
# data = data.sample(frac=1)
# datasets = splitting_data(data,k_fold)
# validation için

for k_fold_i in range(1):

    # datasets_temp = list(datasets)
    # test_data = datasets_temp[k_fold_i]
    # validation için
    test_data = pd.read_csv('test_projection_PCA.zip')
    test_data['y'] = test_data['y'].replace({0: -1})

    # del datasets_temp[k_fold_i]
    # train_data = pd.concat(datasets_temp, axis='rows')
    # validation için
    train_data = pd.read_csv('projection_PCA.zip')
    train_data['y'] = train_data['y'].replace({0: -1})

    train_data = train_data.reset_index()
    test_data = test_data.reset_index()
    # above code ads additional feature as index, should be deleted
    del train_data['index']
    del test_data['index']

    train_data_copy = train_data.copy()
    test_data_copy = test_data.copy()

    y_train_data = pd.DataFrame(train_data['y'])
    y_test_data = pd.DataFrame(test_data['y'])
    del train_data['y']
    del test_data['y']

    np_y_train = y_train_data.to_numpy()
    np_y_test = y_test_data.to_numpy()

    np_train = train_data.to_numpy()
    np_test = test_data.to_numpy()

    # rates = [0.000001,0.0001,0.01]

    rates = [0.01]

    for rate in rates:
        B = 0
        iterations = 1000
        # lams = [0.01,0.1,1,10,100]
        lams = [0.01]
        for lam in lams:
            print(lam)
            n, p = np_train.shape

            W = np.zeros(p)
            B = 0

            for i in range(iterations):

                """
                np_train = pd.DataFrame(np_train).sample(frac=1)
                np_y_train = pd.DataFrame(np_y_train).sample(frac=1)
                np_train = np_train.reset_index()
                np_y_train = np_y_train.reset_index()
                del np_train['index']
                del np_y_train['index']

                np_train = np_train.to_numpy()
                np_y_train = np_y_train.to_numpy()
                """

                for instance in range(int(len(np_train)) - 1):
                    # dist = np_y_train[instance] * (np.dot( np_train[instance], W) )
                    dist = np_y_train[instance] * (np.dot(np_train[instance], W) - B)

                    if dist >= 1:
                        W = W - rate * (2 * lam * W)
                        # B = 0
                    else:
                        W = W + rate * (np_y_train[instance] * np_train[instance]) - rate * (2 * lam * W)
                        B = B - rate * np_y_train[instance]

            pred = np.dot(np_test, W) - B

            pred = pred.reshape((pred.size, 1))
            print(pred.min())
            print(pred.max())
            pred[pred < 0] = 0
            pred[pred > 0] = 1

            np_y_test[np_y_test < 0] = 0

            error = pred - np_y_test
            MSE = (error.T.dot(error)) / len(np_y_test)
            print(MSE)
            pred = np.concatenate((pred, np_y_test), axis=1)

            pred = pd.DataFrame(pred)
            if (k_fold_i == 0) & (lam == lams[0]) & (rate == rates[0]):
                pred_10fold = pred
                MSEs = MSE
            else:
                pred_10fold = pd.concat((pred_10fold, pred), axis=1)
                MSEs = np.concatenate((MSEs, MSE), axis=1)

            print(k_fold_i)

"""np_train = pd.DataFrame(np_train).sample(frac=1)
    np_y_train = pd.DataFrame(np_y_train).sample(frac=1)
    np_train = np_train.reset_index()
    np_y_train = np_y_train.reset_index()
    del np_train['index']
    del np_y_train['index']

    np_train = np_train.to_numpy()
    np_y_train = np_y_train.to_numpy()"""

"""""
MSElamb = [0] * 5
for i in range(5):
    for j in range(3):
        MSElamb[i] += (MSEs[0,j*5+i])
MSElamb = np.array(MSElamb)"""

A = len(test_data)
true_pos = pd.DataFrame(index=range(A), columns=range(1))
false_pos = pd.DataFrame(index=range(A), columns=range(1))
true_neg = pd.DataFrame(index=range(A), columns=range(1))
false_neg = pd.DataFrame(index=range(A), columns=range(1))

tp = 0
fp = 0
tn = 0
fn = 0
for i in range(A):
    if (pred[0].loc[i] == 1) & (pred[1].loc[i] == 1):
        tp += 1
    # true_pos[0].loc[i] = (predictions[0].loc[i]==1) & (predictions[1].loc[i]==1)
    # true_pos_per = true_pos.value_counts()[1]/A
    if (pred[0].loc[i] == 1) & (pred[1].loc[i] == 0):
        fp += 1
    # false_pos[0].loc[i] = (predictions[0].loc[i]==0) & (predictions[1].loc[i]==1)
    # false_pos_per = false_pos.value_counts()[1] / A
    if (pred[0].loc[i] == 0) & (pred[1].loc[i] == 0):
        tn += 1
    # true_neg[0].loc[i] = (predictions[0].loc[i] == 0) & (predictions[1].loc[i] == 0)
    # true_neg_per = true_neg.value_counts()[0] / A
    if (pred[0].loc[i] == 0) & (pred[1].loc[i] == 1):
        fn += 1
    # false_neg[0].loc[i] = (predictions[0].loc[i] == 1) & (predictions[1].loc[i] == 0)
    # false_neg_per = false_neg.value_counts()[1] / A

print(true_pos.value_counts())
print(false_pos.value_counts())
print(true_neg.value_counts())
print(false_neg.value_counts())

# df.groupby('a').count()
F1 = tp / (tp + 1 / 2 * (fp + fn))
print(F1)


end = time.time()
print(end - start)