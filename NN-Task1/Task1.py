import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('Birds.csv')

# Encoding the gender column
le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])

# Extract relevant features
cdf = df[['gender', 'body_mass', 'beak_length', 'beak_depth', 'fin_length']]

# Define classes
class1 = df.iloc[:50]
class2 = df.iloc[50:100]
class3 = df.iloc[100:150]

print(class1)

# Define masks to generate 30 samples of each class for training and 20 samples for testing
msk1 = np.random.rand(len(class1)) < 0.6
msk2 = np.random.rand(len(class2)) < 0.6
msk3 = np.random.rand(len(class3)) < 0.6

# Train-test split
train_class1 = class1[msk1]
test_class1 = class1[~msk1]
train_class2 = class2[msk2]
test_class2 = class2[~msk2]
train_class3 = class3[msk3]
test_class3 = class3[~msk3]

# Concatenate training data and testing data
train_x = pd.concat([train_class1, train_class2, train_class3]).reset_index(drop=True)
test_x = pd.concat([test_class1, test_class2, test_class3]).reset_index(drop=True)

# Prepare training and testing labels
train_y = train_x['bird category']
test_y = test_x['bird category']

# Drop the label from feature data
train_x = train_x.drop(columns=['gender'])
test_x = test_x.drop(columns=['gender'])

# print("Training features:\n", train_x.head())
# print("Training labels:\n", train_y.head())
# print("Testing features:\n", test_x.head())
# print("Testing labels:\n", test_y.head())

