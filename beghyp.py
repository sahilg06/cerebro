
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

x_train=pd.read_csv('train.csv',parse_dates=['Date']).iloc[:,:14]
y_train=pd.read_csv('train.csv').iloc[:,14:15]
x_train=x_train.set_index('TID')
x_test=pd.read_csv('test.csv',parse_dates=['Date'])
x_test=x_test.set_index('TID')
x_test['year']=x_test['Date'].apply(lambda x:x.year)
x_test['month']=x_test['Date'].apply(lambda x:x.month)
x_test['day']=x_test['Date'].apply(lambda x:x.day)
x_train['year']=x_train['Date'].apply(lambda x:x.year)
x_train['month']=x_train['Date'].apply(lambda x:x.month)
x_train['day']=x_train['Date'].apply(lambda x:x.day)



x_train=x_train.drop(columns=['Date','AddressLine1','AddressLine2','Street','Locality','Town','Taluka','District','Price Category'])
x_test=x_test.drop(columns=['Date','AddressLine1','AddressLine2','Street','Locality','Town','Taluka','District','Price Category'])
  
  

#missing data
from sklearn.impute import SimpleImputer
imp=SimpleImputer( strategy='most_frequent')
imp1=SimpleImputer( strategy='most_frequent')
x_train=imp.fit_transform(x_train)
x_train = pd.DataFrame(x_train)
x_test=imp1.fit_transform(x_test)
x_test = pd.DataFrame(x_test)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
le1=LabelEncoder()
# Categorical boolean mask
categorical_feature_mask = x_train.dtypes==object
categorical_feature_mask1= x_test.dtypes==object
# filter categorical columns using mask and turn it into a list
categorical_cols = x_train.columns[categorical_feature_mask].tolist()
categorical_cols1 = x_test.columns[categorical_feature_mask1].tolist()

# apply le on categorical feature columns
x_train[categorical_cols] = x_train[categorical_cols].apply(lambda col: le.fit_transform(col))
x_test[categorical_cols1] = x_test[categorical_cols1].apply(lambda col: le1.fit_transform(col))

# import OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(categorical_features=[0,1,2])
ohe1=OneHotEncoder(categorical_features=[0,1,2])
x_train = ohe.fit_transform(x_train).toarray()
x_test = ohe1.fit_transform(x_test).toarray()
 # It returns an numpy array
 
 
x_test = pd.DataFrame(x_test)
x_train = pd.DataFrame(x_train)
x_train=x_train.drop(columns=[0,5,7])
x_test=x_test.drop(columns=[0,5,7])


from sklearn.decomposition import PCA 
  
pca = PCA(n_components = 2) 
  
x_train = pca.fit_transform(x_train) 
x_test = pca.transform(x_test) 
  
explained_variance = pca.explained_variance_ratio_ 
from sklearn.linear_model import LinearRegression

reg=LinearRegression()
reg.fit(x_train,y_train)

y_predict=reg.predict(x_test)






