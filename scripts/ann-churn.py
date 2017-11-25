## Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Importing the dataset
dataset = pd.read_csv('C:\\Users\\mpatel\\Documents\\MP_Personal\\Udemy_DeepLearning\\Artificial_Neural_Networks\\Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values  ## Removing unnecessary columns
y = dataset.iloc[:, 13].values

## Plotting some variables to get the sense of the dataset:
## Gender vs. Geography
fig, ax = plt.subplots(figsize = (12,8))
temp_df = dataset.groupby(by = ['Geography', 'Gender']).size().unstack()
temp_df.plot(ax = ax, kind = 'bar')
ax.set_title('Distribution of Gender and Geography')
ax.set_ylabel('Count')
plt.show()

## Age, Credit Score, Bank Balance, and  Estimated Salary
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize = (12,8))
ax1.scatter(x = dataset['Age'], y = dataset['CreditScore'])
ax2.scatter(x = dataset['Age'], y = dataset['Balance'])
ax3.scatter(x = dataset['Age'], y = dataset['EstimatedSalary'])
ax4.scatter(x = dataset['Balance'], y = dataset['EstimatedSalary'])

ax1.set_xlabel('Age')
ax2.set_xlabel('Age')
ax3.set_xlabel('Age')
ax4.set_xlabel('Balance')

ax1.set_ylabel('Credit Score')
ax2.set_ylabel('Bank Balance')
ax3.set_ylabel('Estimated Salary')
ax4.set_ylabel('Estimated Salary')

fig.tight_layout()
plt.show()

## Distribution of Age, Credit Score, Balance, Salary:
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize = (12,8))
ax1.hist(dataset['Age'])
ax2.hist(dataset['CreditScore'])
ax3.hist(dataset['EstimatedSalary'])
ax4.hist(dataset['Balance'])

ax1.set_xlabel('Age')
ax2.set_xlabel('Credit Score')
ax3.set_xlabel('Estimated Salary')
ax4.set_xlabel('Bank Balance')
 
ax1.set_ylabel('Count')
ax2.set_ylabel('Count')
ax3.set_ylabel('Count')
ax4.set_ylabel('Count')

fig.tight_layout()
plt.show()

## Checking mean credit score of those who 'Exited' and those who didn't:
fig, ax = plt.subplots(figsize = (12, 8))
dataset.groupby(by = 'Exited')['CreditScore'].mean().plot(kind = 'bar', ax = ax)
ax.set_ylabel('Average Credit Score')
plt.show()

## Checking how many males and females exited/stayed:
fig, ax = plt.subplots(figsize = (12, 8))
dataset.groupby(by = ['Exited', 'Gender']).size().unstack().plot(kind = 'bar', ax = ax)
ax.set_ylabel('Count')
plt.show()

## Encoding categorical data: consider two categorical columns: Geography and Gender
## Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

## Encoding Geography
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

## Encoding Gender
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

## Create dummy variables after encoding categorical variables:
## For Geography:
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

## Remove one of the dummy columns of country variable to avoid dummy variable trap:
X = X[:, 1:]

## Splitting the dataset into Training and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

## Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

## Start building ANN:
import keras
from keras.models import Sequential  ## Required to initialize the neural-network
from keras.layers import Dense  ## Required to build layers of ANN

## Initialize ANN: This ANN will work as a classifier.
classifier_nn = Sequential()

## Add two hidden layers in NN:
classifier_nn.add(Dense(units = 6, kernel_initializer = 'uniform', 
                     activation = 'relu', input_shape = (11,)))

classifier_nn.add(Dense(units = 6, kernel_initializer = 'uniform', 
                     activation = 'relu'))

## Add o/p layer: If dependent variables has more than 2 categories then we'll use 'softmax' instead of 'sigmoid':
classifier_nn.add(Dense(units = 1, kernel_initializer = 'uniform', 
                     activation = 'sigmoid'))

## Compiling the NN:
classifier_nn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


## Fitting the ANN to data:
classifier_nn.fit(x = X_train, y = y_train, batch_size = 10, epochs = 100)


## Predicting the Test set results
y_pred = classifier_nn.predict(X_test)
y_pred = (y_pred > 0.5)

## Checking the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm
