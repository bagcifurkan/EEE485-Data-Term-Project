import pandas as pd
import csv

data0 = pd.read_csv('bank-additional-full-copy.csv', delimiter=";")


data0.columns = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'duration', 'campaign',
                 'pdays', 'previous','poutcome', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed', 'y']

#Changing strings into 0-1 for output feature - 'y'
data0['y'] = data0['y'].replace({'yes':1, 'no':0})

#Did nothing for 'age'

#One hot encoding for jobs - 'jobs'
data0 = pd.get_dummies(data0, columns=['job'])

#One hot encoding for marital - 'marital'
data0 = pd.get_dummies(data0, columns=['marital'])

#Changing strings into integer values for education - 'education'
data0['education'] = data0['education'].replace({'illiterate':0, 'basic.4y':1, 'basic.6y':2, 'basic.9y':3,
                                                 'high.school':4, 'professional.course':5, 'university.degree':6, 'unknown':float("NaN") })
data0['education'] = data0['education'].replace({ float("NaN"): data0['education'].mean()})

#Changing strings into 0-1 for defaul - 'default'
data0['default'] = data0['default'].replace({'yes':1, 'no':0, 'unknown':float("NaN")})
data0['default'] = data0['default'].replace({float("NaN"):data0['default'].mean()})

#Changing strings into 0-1 for housing - 'housing'
data0['housing'] = data0['housing'].replace({'yes':1, 'no':0, 'unknown':float("NaN")})
data0['housing'] = data0['housing'].replace({float("NaN"):data0['housing'].mean()})

#Changing strings into 0-1 for loan - 'loan'
data0['loan'] = data0['loan'].replace({'yes':1, 'no':0, 'unknown':float("NaN")})
data0['loan'] = data0['loan'].replace({float("NaN"):data0['loan'].mean()})

#Changing strings into 0-1 for contact - 'contact'
data0['contact'] = data0['contact'].replace({'cellular':1, 'telephone':0, })

#Removing 'month'
del data0['month']

#Removing 'day_of_week'
del data0['day_of_week']

#Removing 'duration'
del data0['duration']

#Did nothing for 'campaign'
#Did nothing for 'pdays'
#Did nothing for 'previous'

#Changing strings into 0-1 for poutcome - 'poutcome'
data0['poutcome'] = data0['poutcome'].replace({'failure':0, 'nonexistent':1, 'success':2 })


#Did nothing for 'emp.var.rate'
#Did nothing for 'cons.price.idx'
#Did nothing for 'cons.conf.idx'
#Did nothing for 'euribor3m'
#Did nothing for 'nr.employed'



with pd.option_context('display.max_rows', 6, 'display.max_columns', 35):  # more options can be specified also
    print(data0)



data0.to_csv('before_normalizing1.zip', index=False, compression=dict(method='zip', archive_name='before_normalizing1.csv'))

data_response = data0['y']
del data0['y']


data_ilk15 = data0[['age', 'education', 'default', 'housing', 'loan', 'contact',  'campaign',
                 'pdays', 'previous','poutcome', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']]

data_son16 = data0[['job_admin.', 'job_blue-collar', 'job_entrepreneur', 'job_housemaid', 'job_management', 'job_retired', 'job_self-employed','job_services',
                    'job_student', 'job_technician', 'job_unemployed', 'job_unknown', 'marital_divorced', 'marital_married', 'marital_single', 'marital_unknown']]

print(data_ilk15)
print(data_son16)


data_normalized = (data_ilk15-data_ilk15.describe().loc["mean"])/data_ilk15.describe().loc["std"]
data_normalized = pd.concat([data_normalized,data_son16 ], axis='columns')
data_normalized['y'] = data_response

data_normalized.to_csv('data_normalized2.zip', index=False, compression=dict(method='zip', archive_name='data_normalized2.csv'))

#print(data_normalized.describe())
#print(data0.loc[3])
#csv0 = data0.to_csv(index=False)
#print(data0['education'][7:15])

with pd.option_context('display.max_rows', 6, 'display.max_columns', 35):  # more options can be specified also
    print(data_normalized)

    print(data_normalized.describe())
