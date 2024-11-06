import random

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
# Load the dataset
df = pd.read_csv('Birds.csv')
df['gender'] = df['gender'].fillna(df['gender'].mode()[0])
# Encoding the gender column
le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])
df['bird category'] = le.fit_transform(df['bird category'])
# print(df['bird category'].head(10))

# Extract relevant features
cdf = df[['gender', 'body_mass', 'beak_length', 'beak_depth', 'fin_length']]

# Define classes
class1 = df.iloc[:50]
class2 = df.iloc[50:100]
class3 = df.iloc[100:150]

# print(class1)

# Define masks to generate 30 samples of each class for training and 20 samples for testing
msk1 = np.random.rand(len(class1)) <= 0.6
msk2 = np.random.rand(len(class2)) <= 0.6
msk3 = np.random.rand(len(class3)) <= 0.6



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
train_x = train_x.drop(columns=['bird category'])
test_x = test_x.drop(columns=['bird category'])

# print("Training features:\n", train_x.head())
# print("Training labels:\n", train_y.head())
# print("Testing features:\n", test_x.head())
# print("Testing labels:\n", test_y.head())







# implement the required function that will use in perceptron algorithm

# 1 - calculate net value which represent linear combination between inputs and weights
def calculateNetValue(inputs, weights, bias):
    if bias == 0:
        return np.sum(inputs * weights)
    else:
        x = random.uniform(-1, 1)
        bias = x
        return np.sum(inputs * weights) + bias


# 2 - activation function that receive z = linear combination between inputs and weights
def signum(z):
    if z >= 0:
        return 1
    else:
        return -1


# 3 - calculate error between t => what actually the output should look like - y=> what output I get (output of act fn)
def calculateError(t, y):
    return t - y


# 4 - update weight if needed
def updateWeight(weights, learningRate, error, input):
    return weights + (learningRate * error * input)

# combine all functions to implement single layer preceptron algorithm

def singelLayerPreceptron(input, output, weights , learningRate, epochs, bias=0):
    for epoch in range(epochs):
        for i in range(len(input)):
            z = calculateNetValue(input.iloc[i], weights, bias)
            y = signum(z)
            error = calculateError(output.iloc[i], y)

            if error == 0:
                continue

            weights = updateWeight(weights, learningRate, error, input.iloc[i])
    return weights


# train the model


# Initialize weights and bias
weights = np.random.rand(train_x.shape[1])
learning_rate = 0.01
epochs = 500

singelLayerPreceptron(train_x,train_y,weights,learning_rate,epochs)

def tetsing(test_x, weights):
    predictions = []
    for i in range(len(test_x)):
        z = calculateNetValue(test_x.iloc[i], weights, 0)  # No bias in this case
        y_pred = signum(z)
        predictions.append(y_pred)
    return predictions

# Get predictions
predictions = tetsing(test_x, weights)

# Calculate confusion matrix and accuracy
conf_matrix = confusion_matrix(test_y, predictions)
accuracy = accuracy_score(test_y, predictions)

print("Confusion Matrix:\n", conf_matrix)
print("Model Accuracy:", accuracy)

