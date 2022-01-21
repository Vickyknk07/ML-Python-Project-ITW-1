# Importing Modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Loading Data and Splitting it into Training and Test Data
filename = 'Problem2.csv'
data = pd.read_csv(filename)
data_train, data_test = train_test_split(data, test_size=0.3)

# Extracting the Feature vectors and Label vector from Training Dataset
# Scaling the Feature vectors so that their values lies in (0,1) range
X_train = np.array(data_train[data_train.columns[:-1]])
scaler = preprocessing.MinMaxScaler().fit(X_train)
X_train_scaled = scaler.fit_transform(X_train)
y_train = np.array(data_train['Outcome'])

# Extracting the Feature vectors and Label vector from Test Dataset
X_test = np.array(data_test[data_test.columns[:-1]])
X_test_scaled = scaler.fit_transform(X_test)
y_test = np.array(data_test['Outcome'])

# Declaring Neural Network Model
# Training the Model with given Dataset
nn = MLPClassifier(solver='lbfgs',
                   alpha=1e-5,
                   hidden_layer_sizes=(6, 6),
                   random_state=1,
                   max_iter=1000)
nn.fit(X_train_scaled, y_train)

# Making predictions using the Trained model
# Finding the Accuracy of the predictions and printing it
y_train_pred = nn.predict(X_train_scaled)
y_test_pred = nn.predict(X_test_scaled)
accuracy_train = (accuracy_score(y_train_pred, y_train)) * 100
accuracy_test = (accuracy_score(y_test_pred, y_test)) * 100
print("Accuracy on Training data: {}%".format(accuracy_train))
print("Accuracy on Test data: {}%".format(accuracy_test))

# Plotting and Visualizing the Data
fig, ax = plt.subplots(2, 2)
fig.set_size_inches(15, 8)
pos = np.squeeze(np.where(y_train == 1))
neg = np.squeeze(np.where(y_train == 0))
x = np.array(data_train['Age'])
y = np.array(data_train['DiabetesPedigreeFunction'])
z = np.array(data_train['Glucose'])

# Histogram of Diabetic and Non-Diabetic people distributed
# on the basis of Age, from Training Dataset
ax[0, 0].hist([x[pos], x[neg]], bins=10, label=['Diabetic', 'Not Diabetic'])
ax[0, 0].set_title('Distribution by Age')
ax[0, 0].set_xlabel('Age')
ax[0, 0].set_ylabel('Frequency')
ax[0, 0].legend()

# Histogram of Diabetic and Non-Diabetic people distributed
# on the basis of DPF, from Training Dataset
ax[0, 1].hist([y[pos], y[neg]], bins=10, label=['Diabetic', 'Not Diabetic'])
ax[0, 1].set_title('Distribution by DPF')
ax[0, 1].set_xlabel('Diabetes Pedigree Function (DPF)')
ax[0, 1].set_ylabel('Frequency')
ax[0, 1].legend()

# Histogram of Diabetic and Non-Diabetic people distributed
# on the basis of Glucose, from Training Dataset
ax[1, 0].hist([z[pos], z[neg]], bins=10, label=['Diabetic', 'Not Diabetic'])
ax[1, 0].set_title('Distribution by Glucose')
ax[1, 0].set_xlabel('Glucose')
ax[1, 0].set_ylabel('Frequency')
ax[1, 0].legend()

# Scatter plot of predictions
correct_pred = np.where(y_test == y_test_pred)
wrong_pred = np.where(y_test != y_test_pred)
ax[1, 1].scatter(X_test[correct_pred, 6], X_test[correct_pred, 7], c='r', label='Correct Prediction')
ax[1, 1].scatter(X_test[wrong_pred, 6], X_test[wrong_pred, 7], c='k', label='Wrong Prediction')
ax[1, 1].set_title('Predictions')
ax[1, 1].set_xlabel('DPF')
ax[1, 1].set_ylabel('Age')
ax[1, 1].legend()

plt.show()