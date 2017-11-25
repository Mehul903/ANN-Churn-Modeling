## Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Importing the dataset
dataset = pd.read_csv('../data/Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values  ## Removing unnecessary columns
y = dataset.iloc[:, 13].values

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
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def construct_classifier():   
    
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

    return classifier_nn

classifier = KerasClassifier(build_fn = construct_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
mean_ac = np.mean(accuracies)
var_ac = np.var(accuracies)

## Predicting the Test set results
y_pred = classifier_nn.predict(X_test)
y_pred = (y_pred > 0.5)

## Checking the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm


## Make prediction on one new observation:
new_obs = np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])  
new_obc_scaled = scaler.transform(new_obs)
classifier_nn.predict(new_obs)