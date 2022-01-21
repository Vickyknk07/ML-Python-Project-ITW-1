# Importing Modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Loading Data
fruits = pd.read_csv('problem1.txt', delimiter='\t')
X = fruits.iloc[:, -4:]
Y = fruits.iloc[:, 1]

# Data Normalization
X = preprocessing.StandardScaler().fit_transform(X)

# Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
Data_train, Data_test = train_test_split(fruits, test_size=0.3, random_state=1)

# Training and Predicting using K-Nearest Neighbour Algorithm
knnmodel = KNeighborsClassifier(n_neighbors=3)
knnmodel.fit(X_train, Y_train)
knn_y_predict = knnmodel.predict(X_test)

# Accuracy of K-Nearest Neighbour Algorithm
acc_knn = accuracy_score(Y_test, knn_y_predict)
print('Accuracy of K-Nearest Neighbour Algorithm : ', str(acc_knn * 100) + '%')

# Training and Predicting using Logistic Regression Algorithm
lrmodel = LogisticRegression(solver='liblinear')
lrmodel.fit(X_train, Y_train)
lr_y_predict = lrmodel.predict(X_test)

# Accuracy of Logistic Regression Algorithm
acc_lr = lrmodel.score(X_test, Y_test)
print('Accuracy of Logistic Regression Algorithm : ', str(acc_lr * 100) + '%')

print('K Neighbour is better than Logistic Regression')

# Output Visualization
# prediction_output = pd.DataFrame(data = [Y_test.values, knn_y_predict], index = ['Y_test', 'knn_y_predict'])
# print(prediction_output.transpose())
figure, axis = plt.subplots(2, 2)
figure.set_size_inches(15, 8)

# Histogram of Fruits type and their Frequency in Training Dataset
axis[0, 0].hist(Y_train, 4, width=0.5)
axis[0, 0].set_title('Fruit Distribution in Training Dataset')
axis[0, 0].set_xlabel('Fruits')
axis[0, 0].set_ylabel('Frequency')

# Scatter plot of Fruit Distribution according
# to their Width and Height, from Training Dataset
groups = Data_train.groupby("fruit_name")
for name, group in groups:
    axis[0, 1].plot(group["width"], group["height"], marker="o", linestyle="", label=name)
    axis[0, 1].set_title('Fruit Distribution according to Width and Height')
    axis[0, 1].set_xlabel('Width')
    axis[0, 1].set_ylabel('Height')
    axis[0, 1].legend()

# Scatter plot of Fruit Predictions
# using K-Nearest Neighbour Algorithm
knn_correct_pred = np.where(Y_test == knn_y_predict)
knn_wrong_pred = np.where(Y_test != knn_y_predict)
axis[1, 0].scatter(X_test[knn_correct_pred, 1], X_test[knn_correct_pred, 2], c='g', label='Correct Prediction')
axis[1, 0].scatter(X_test[knn_wrong_pred, 1], X_test[knn_wrong_pred, 2], c='r', label='Wrong Prediction')
axis[1, 0].set_title('K-Nearest Neighbour Algorithm Predictions')
axis[1, 0].set_xlabel('Width')
axis[1, 0].set_ylabel('Height')
axis[1, 0].legend()

# Scatter plot of Fruit Predictions
# using Logistic Regression Algorithm
lr_correct_pred = np.where(Y_test == lr_y_predict)
lr_wrong_pred = np.where(Y_test != lr_y_predict)
axis[1, 1].scatter(X_test[lr_correct_pred, 1], X_test[lr_correct_pred, 2], c='g', label='Correct Prediction')
axis[1, 1].scatter(X_test[lr_wrong_pred, 1], X_test[lr_wrong_pred, 2], c='r', label='Wrong Prediction')
axis[1, 1].set_title('Logistic Regression Algorithm Predictions')
axis[1, 1].set_xlabel('Width')
axis[1, 1].set_ylabel('Height')
axis[1, 1].legend()

plt.show()








