import ast

import numpy as np
import matplotlib as plt
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('Birds.csv')

#  encoding gender coulmn
encoder = TargetEncoder(cols=['gender'])
print(df['gender'])

cdf = df[['gender','body_mass','beak_length','beak_depth','fin_length']]

# define classes belongs to a,b,c which is defined targets

class1 = df.iloc[:50,-1:]
class2 = df.iloc[51:100,-1:]
class3 = df.iloc[101:151,-1:]


# Extracting featuers

feature1 = df.iloc[:50,:]
feature2 = df.iloc[51:100,:]
feature3 = df.iloc[101:151,:]


# define masks to generate 30 sample of each class for training and 20 samples for testing

msk1 = np.random.rand(len(feature1)) < 0.6
msk2 = np.random.rand(len(feature2)) < 0.6
msk3 = np.random.rand(len(feature3)) < 0.6

# train test split

train_class1 = cdf[msk1]
test_class1 =  df[~msk1]
train_class2 = df[msk2]
test_class2 =  df[~msk2]
train_class3 = df[msk3]
test_class3 =  df[~msk3]


# concatenate the train data and test data to train and test model

train_x = np.concatenate(train_class1,train_class2,test_class3)
test_x = np.concatenate(test_class1,test_class2,test_class3)


print(train_x)




