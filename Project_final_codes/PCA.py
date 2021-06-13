import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import csv

X =  pd.read_csv('train_X.zip')

A_pca = np.array(X)
cov = np.cov(A_pca.T)
values, vectors = np.linalg.eig(cov)

index = values.argsort()[::-1]
values = values[index]
vectors = vectors[:,index]
projection = A_pca.dot(vectors)

total_var = 0

for i in range(0,31):
    total_var += np.var(A_pca[:,i])

PVE = np.zeros(31)
PVE_m = np.zeros(31)
for i in range(0,31):
    print(np.var(projection[:,i])/total_var)
    PVE_m[i] = (np.var(projection[:,i])/total_var)

    if i >= 1:
        PVE[i]= PVE[i-1] + (np.var(projection[:, i]) / total_var)
    else:
        PVE[0]=PVE_m[0]

    print(PVE)
    print("""""")

PVE_m = pd.DataFrame(PVE_m)
PVE_m = PVE_m.reset_index()
PVE_m.columns = ['Principal Component', 'PVE']
PVE_m.plot.line(x='Principal Component', y='PVE')

PVE = pd.DataFrame(PVE)
PVE = PVE.reset_index()
PVE.columns = ['Principal Component', 'Cumulative PVE']
PVE.plot.line(x='Principal Component', y='Cumulative PVE')

plt.show()

values = values[0:14]
vectors = vectors[:,0:14]
projection = A_pca.dot(vectors)

projection =  pd.DataFrame(projection)
#projection.to_csv('projection_PCA.zip', index=False, compression=dict(method='zip', archive_name='projection_PCA.csv'))