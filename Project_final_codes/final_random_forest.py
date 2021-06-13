from __future__ import print_function
from _csv import reader
from math import sqrt
from random import randrange, seed

import pandas as pd
import random
import numpy as np
import time

start = time.time()
test_X = pd.read_csv('test_X.zip')
train_X = pd.read_csv('train_X.zip')
test_Y = pd.read_csv('test_Y.zip')
train_Y = pd.read_csv('train_Y.zip')
data = pd.read_csv('train.zip')
train_data = pd.read_csv('train.zip')

#training_data = df.to_numpy();


header = ['age', 'education', 'default', 'housing', 'loan', 'contact', 'campaign', 'pdays', 'previous', 'poutcome',
          'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m',
          'nr.employed', 'job_admin.', 'job_blue-collar', 'job_entrepreneur', 'job_housemaid', 'job_management',
          'job_retired', 'job_self-employed', 'job_services', 'job_student', 'job_technician', 'job_unemployed',
          'job_unknown', 'marital_divorced', 'marital_married',
          'marital_single', 'marital_unknown', 'y']



def tree_data(train_data):
    """Creating a random data for a tree"""
    size1, size2 = train_data.shape
    tree = pd.DataFrame(index=range(0), columns=range(size2))
    for j in range(size1):
        a = train_data.sample()
        if j == 0:
            tree = a
        else:
            tree = pd.concat([tree,a ], axis='rows')
    training_data = tree.to_numpy();
    return training_data

def classes(train_np):
    #print("class_distribution")
    """Find and returns the number of class in given data"""
    size1 , size2 = train_np.shape
    dist = {'0': 0 , '1': 0}
    for i in range(size1):
        if train_np[i][31] == 0:
            dist['0']  = dist['0'] + 1
        else:
            dist['1'] = dist['1'] + 1
    return dist


class Node_Question:
    """ A question is needed to split data in each node"""

    def __init__(self, feature, feature_value):
        self.feature = feature
        self.feature_value = feature_value

    def compare(self, sample):
        return self.feature_value <= sample[self.feature]


def splitting(train_np, Q):
    """Splitting data into left(false) and right(true) according to the question"""
    right = []
    left = []
    for row in train_np:
        if Q.compare(row):
            right.append(row)
        else:
            left.append(row)
    return np.array(right), np.array(left)

def gini(train_np):
    """Calculating gini metric"""
    dict = classes(train_np)
    zeros = dict['0']
    ones = dict['1']
    total = zeros + ones
    if (ones == 0) or (zeros == 0) :
        gini = 0
    else:
        gini = 1 - (zeros/total)**2 -  (ones/total)**2
    return gini

def best_question(train_np):
    #print("best_question")
    """Finds the best question with best gain to split data into two with using random 5 Features"""
    size1, size2 = train_np.shape
    random_5_features = []
    for i in range(0, 7):
        n = random.randint(1, 30)
        random_5_features.append(n)
    best_gain = 0
    best_question = None

    for i in random_5_features:
        if i == 13:
            continue
        unique_vals =  np.unique(train_np[:,i])

        for j in unique_vals:
            node_question = Node_Question(i,j)
            right, left = splitting(train_np, node_question)

            if len(right) == 0 or len(left) == 0:
                continue

            ratio = float(len(left)) / (len(left) + len(right))
            gain = gini(train_np) - ratio * gini(left) - (1 - ratio) * gini(right)

            if gain >= best_gain:
                best_question, best_gain = node_question, gain
            else:
                continue

    return best_gain, best_question


class node:
    """Every node in a tree as a class"""
    def __init__(self,Q, left, right, classes):
        self.Q = Q
        self.left = left
        self.right = right
        self.classes_leaf = classes


def build_tree(train_np):
    #print("tree")
    """Building a tree recursively"""
    bestGain,bestQuestion,  = best_question(train_np)
    #print("best split is done")
    if bestGain == 0:
        #return class_distribution(train_np)
        return node(None, None, None, classes(train_np))
    else:
        right,left = splitting(train_np, bestQuestion)
        left_tree = build_tree(left)
        right_tree = build_tree(right)
        return node(bestQuestion, left_tree, right_tree, None)



def tree():
     training_data = tree_data(train_data)
     my_tree = build_tree(training_data)
     return my_tree
     #print_tree(my_tree)

def classify(test_instance, tree):
    #print("classification")
    if tree.classes_leaf is not None:
        pred = tree.classes_leaf
        values_pred = list(pred.values())
        if values_pred[0] > values_pred[1]:
            return 0
        else:
            return 1
    else:
        if tree.Q.compare(test_instance):
            return classify(test_instance, tree.right)
        else:
            return classify(test_instance, tree.left)


#df = train_data.iloc[:6590]
#train_data = train_data.iloc[6590:]

df = pd.read_csv('test.zip')
testing_data = df.to_numpy()

"""
for i in range(5):
    my_tree = tree()
    for row in testing_data:
        print("Actual: %s. Predicted: %s" %
              (row[-1], print_leaf(classify(row, my_tree))))
"""
alltrees = [0] * len(testing_data)
alltrees = np.array(alltrees)
alltrees.reshape(len(testing_data),1)
count = 0

tree_count = 10
result_data = np.ones((len(testing_data),tree_count+1))
result_data[:,0] = testing_data[:,-1]


for i in range(tree_count):
    print(i)
    my_tree = tree()
    count = 0
    for row in testing_data:
        pred = classify(row, my_tree)
        result_data[count][i+1] = pred
        count +=1


predictions = np.ones((len(testing_data),2))
predictions[:,0] = testing_data[:,-1]

count_acc = 0
for i in range(len(testing_data)):
    count_selam = 0
    for j in range(tree_count):
        count_selam += result_data[i][j+1]

        if count_selam >= tree_count/2:
            predictions[i][1] = 1
        else:
            predictions[i][1] = 0

        if predictions[i][1] == predictions[i][0]:
            count_acc += 1

print(str(count_acc/len(testing_data) * 100 / tree_count) + '%')


predictions = pd.DataFrame(predictions)
A = len(testing_data)
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
    if (predictions[0].loc[i]==0) & (predictions[1].loc[i]==1):
        fp += 1
    #false_pos[0].loc[i] = (predictions[0].loc[i]==0) & (predictions[1].loc[i]==1)
    #false_pos_per = false_pos.value_counts()[1] / A
    if (predictions[0].loc[i] == 0) & (predictions[1].loc[i] == 0):
        tn += 1
    #true_neg[0].loc[i] = (predictions[0].loc[i] == 0) & (predictions[1].loc[i] == 0)
    #true_neg_per = true_neg.value_counts()[0] / A
    if (predictions[0].loc[i] == 1) & (predictions[1].loc[i] == 0):
        fn += 1
    #false_neg[0].loc[i] = (predictions[0].loc[i] == 1) & (predictions[1].loc[i] == 0)
    #false_neg_per = false_neg.value_counts()[1] / A




#df.groupby('a').count()
F1 = tp / (tp + 1/2 * (fp + fn ))
print(F1)



end = time.time()
print(end-start)


#result_df = pd.DataFrame(result_data)
#result_df.to_csv("3tree_result_data_2.csv")

