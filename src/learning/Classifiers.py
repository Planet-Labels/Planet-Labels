__author__ = 'rishabh'

import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

class RandomForest_c():
    '''
    This is a class that creates a Random Forest model classifier.
    '''
    def __init__(self, n_trees=100):
        '''
        Creates multiple linear regression models with l2 penalty
        '''
        self.n_estimators = n_trees
        self.model = RandomForestClassifier(n_estimators = n_trees)

    def description(self):
        return 'Random Forest with ' + str(self.n_estimators) + ' trees. '

    def train(self, X_train, y_train):
        '''
        Train the model on training set

        Parameters:
        X_train: 2-D numpy array  that contains each training example as a row
        y_train: 1-D numpy array labels associated with each training example

        Returns:
        The trained model using X_train and y_train
        '''
        return self.model.fit(X_train, y_train)

    def predict(self, X_test):
        '''
        Predict the labels for the test set

        Parameters:
        X_test: 2-D numpy array (the test set) to predict labels for

        Returns:
        List of predicted labels for X_test
        '''
        return self.model.predict(X_test)

    def write(self, file):
        """Writes the model to a pickle"""
        with open(file, 'w') as filehandler:
            pickle.dump(self.model, filehandler)

    def load(self, file):
        """Loads the model from a pickle"""
        with open(file, 'r') as filehandler:
            self.model = pickle.load(filehandler)

class SVM_OnevsRest_c():
    '''
    This is a class that creates a One vs Rest classifier for multilabel classification.
    As of now, it uses an SVM to train n classifeiers, where n is the number of classes
    '''
    def __init__(self):
        '''
        Creates a Multilabel SVM classifier model
        '''
        self.model = OneVsRestClassifier(SVC(C=0.001, kernel='rbf'))

    def description(self):
        return 'OneVsRestClassifier using SVMs'

    def train(self, X_train, y_train):
        '''
        Train the model on training set

        Parameters:
        X_train: 2-D numpy array  that contains each training example as a row
        y_train: 1-D numpy array labels associated with each training example

        Returns:
        The trained model using X_train and y_train
        '''
        return self.model.fit(X_train, y_train)

    def predict(self, X_test):
        '''
        Predict the labels for the test set

        Parameters:
        X_test: 2-D numpy array (the test set) to predict labels for

        Returns:
        List of predicted labels for X_test
        '''
        return self.model.predict(X_test)

    def write(self, file):
        """Writes the model to a pickle"""
        with open(file, 'w') as filehandler:
            pickle.dump(self.model, filehandler)

    def load(self, file):
        """Loads the model from a pickle"""
        with open(file, 'r') as filehandler:
            self.model = pickle.load(filehandler)

class LogisticRegression_c():
    '''
    This is a class that creates a one vs rest classifier where each individual predictor is logistic regression.
    '''
    def __init__(self):
        '''
        Creates multiple linear regression models with l2 penalty
        '''
        self.model = OneVsRestClassifier(LogisticRegression(C=1, penalty='l2'))

    def description(self):
        return 'OneVsRestClassifier using Logistic Regressions'

    def train(self, X_train, y_train):
        '''
        Train the model on training set

        Parameters:
        X_train: 2-D numpy array  that contains each training example as a row
        y_train: 1-D numpy array labels associated with each training example

        Returns:
        The trained model using X_train and y_train
        '''
        return self.model.fit(X_train, y_train)

    def predict(self, X_test):
        '''
        Predict the labels for the test set

        Parameters:
        X_test: 2-D numpy array (the test set) to predict labels for

        Returns:
        List of predicted labels for X_test
        '''
        return self.model.predict(X_test)

    def write(self, file):
        """Writes the model to a pickle"""
        with open(file, 'w') as filehandler:
            pickle.dump(self.model, filehandler)

    def load(self, file):
        """Loads the model from a pickle"""
        with open(file, 'r') as filehandler:
            self.model = pickle.load(filehandler)


