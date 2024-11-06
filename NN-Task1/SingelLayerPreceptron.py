import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report, mean_squared_error

df = pd.read_csv('birds.csv')
df['gender'] = df['gender'].fillna(df['gender'].mode()[0])

scaler = MinMaxScaler()
encoder = OrdinalEncoder()

mapping = {'A': 1, 'B': -1, 'C': 0}
df['bird category'] = df['bird category'].map(mapping)

df['gender'] = encoder.fit_transform(df[['gender']])

Y = df['bird category']
X = df.drop(columns='bird category')
X = X.astype(float)  # Ensure all features are floats

X = X[0:100]
Y = Y[0:100]
X = scaler.fit_transform(X)
X = pd.DataFrame(X)

# print(type(X))
#
# print(type(Y))
# Need to choose about 2 features
# Proceed with train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, shuffle=True, random_state=42, stratify=Y)


# Helper functions for the perceptron
def calculateNetValue(inputs, weights, bias):
    return np.dot(inputs, weights) + bias


def signum(z):
    return 1 if z >= 0 else -1


def calculateError(t, y):
    return t - y


def updateWeight(weights, learningRate, error, input):
    return weights + (learningRate * error * input)


# Implementing the perceptron algorithm
def singleLayerPerceptron(input, output, weights, learningRate, epochs, bias=0):
    for epoch in range(epochs):
        for i in range(len(input)):
            z = calculateNetValue(input.iloc[i], weights, bias)
            y = signum(z)
            error = calculateError(output.iloc[i], y)

            if error != 0:
                weights = updateWeight(weights, learningRate, error, input.iloc[i])
    return weights





def Adaline(input, output, weights, learningRate, epochs, threshold, bias=0):
    for epoch in range(epochs):
        epoch_predictions = []

        for i in range(len(input)):
            z = calculateNetValue(input.iloc[i], weights, bias)  # Linear output
            y = z  # No activation function in Adaline, just the linear output
            error = calculateError(output.iloc[i], y)

            # Update weights based on the error
            weights = updateWeight(weights, learningRate, error, input.iloc[i])

            # Collect predictions for MSE calculation
            epoch_predictions.append(y)

        # Calculate mean squared error for the entire epoch
        mse = mean_squared_error(output, epoch_predictions)
        print(f"Epoch {epoch + 1}, MSE: {mse}")

        # Early stopping if MSE is below the threshold
        if mse < threshold:
            print("Training stopped early due to MSE threshold.")
            break

    return weights


# Testing function
def evaluate_preceptron(test_x, weights):
    predictions = []
    for i in range(len(test_x)):
        z = calculateNetValue(test_x.iloc[i], weights, 0)  # No bias in this case
        y_pred = signum(z)
        predictions.append(y_pred)
    return predictions


def evaluate_Adaline(test_x, weights, threshold=0.00001):
    predictions = []
    for i in range(len(test_x)):
        z = calculateNetValue(test_x.iloc[i], weights, 0)  # No bias in this case
        print(z)
        y_pred = z  # Continuous output from Adaline
        # Apply threshold to convert to binary label
        binary_pred = 1 if y_pred >= threshold else -1
        predictions.append(binary_pred)
    return predictions


# **************************************plotting functions********************************


def plot_decision_boundary(weights, bias=0, x_range=(0, 1)):
    slope = -weights[0] / weights[1]
    intercept = -bias / weights[1]
    x_vals = np.linspace(x_range[0], x_range[1], 100)
    y_vals = slope * x_vals + intercept
    return x_vals, y_vals


def plot_with_train_data(X_train, Y_train, weights, bias=0):
    # Calculate x and y values for the decision boundary
    x_vals, y_vals = plot_decision_boundary(weights, bias, x_range=(X_train.min().min(), X_train.max().max()))

    # Scatter plot of the training data
    plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=Y_train, cmap='viridis', edgecolor='k',
                label='Training Data Points')
    # Plot the decision boundary
    plt.plot(x_vals, y_vals, 'r-', label='Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Perceptron Decision Boundary (Training Data)')
    plt.legend()
    plt.show()


# Initialize weights and bias for preceptron model
weights = np.random.rand(X_train.shape[1]).astype(float)  # Ensure weights are floats
learning_rate = 0.01
epochs = 1000
# Train the model
weights = singleLayerPerceptron(X_train, Y_train, weights, learning_rate, epochs)
# Get predictions for preceptron model
predictions = evaluate_preceptron(X_test, weights)

# Calculate confusion matrix and accuracy
conf_matrix = confusion_matrix(Y_test, predictions)
accuracy = accuracy_score(Y_test, predictions)
f1 = f1_score(Y_test, predictions)
print("Confusion Matrix for preceptron algorithm :\n", conf_matrix)
print("Model Accuracy for preceptron :", accuracy)
print("f1 score  for preceptron = ", f1)

# Initialize weights and bias for Adaline model
weights = np.random.rand(X_train.shape[1]).astype(float)  # Ensure weights are floats
learning_rate = 0.01
epochs = 1000
threshold = 0.01
# Train the model
weights = Adaline(X_train, Y_train, weights, learning_rate, epochs, threshold)

# Get predictions for preceptron model
predictions = evaluate_Adaline(X_test, weights, 0.0001)

# Calculate confusion matrix and accuracy
# classification_report = classification_report(Y_test,predictions)
conf_matrix = confusion_matrix(Y_test, predictions)
accuracy = accuracy_score(Y_test, predictions)
# # f1 = f1_score(Y_test,predictions)
# print("Confusion Matrix for adaline algorithm :\n", conf_matrix)
# print("Model Accuracy for adaline :", accuracy)
# print("f1 score  for adaline = ",f1)

print("conf Matrix ********* \n", conf_matrix)
print("Model Accuracy for adaline :", accuracy)
