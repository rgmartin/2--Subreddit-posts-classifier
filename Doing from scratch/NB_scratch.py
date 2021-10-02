import numpy as np
import pandas as pd
import time

class MultinomialNB_scratch:
    def __init__(self, alpha=1):
        self.alpha = alpha 
    def fit(self, X_train, y_train):
        # m = no. of instances
        # n = no. of features
        m, n = X_train.shape
        self._classes = np.unique(y_train)
        n_classes = len(self._classes)

        # init: Prior & Likelihood
        self._priors = np.zeros(n_classes)
        self._likelihoods = np.zeros((n_classes, n))

        for idx, c in enumerate(self._classes):
            # X_train_c =  matrix containing only instances with class c
            X_train_c = X_train[c == y_train]
            # compute priorprobability of class c as no. of instances of class c
            # over all instances.
            self._priors[idx] = X_train_c.shape[0] / m 
            # compute likelihoods with self.alpha for smoothing
            self._likelihoods[idx, :] = ((X_train_c.sum(axis=0)) + self.alpha) / (np.sum(X_train_c.sum(axis=0) + self.alpha))
    def predict(self, X_test):
        return [self._predict(x_test) for x_test in X_test]
    def _predict(self, x_test):
        # Calculate posterior for each class
        posteriors = []
        for idx, c in enumerate(self._classes):
            prior_c = np.log(self._priors[idx])
            likelihoods_c = self.calc_likelihood(self._likelihoods[idx,:], x_test)
            posteriors_c = np.sum(likelihoods_c) + prior_c
            posteriors.append(posteriors_c)
        return self._classes[np.argmax(posteriors)]
    def calc_likelihood(self, cls_likeli, x_test):
        return np.log(cls_likeli) * x_test
    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return np.sum(y_pred == y_test)/len(y_test)
        
class BernoulliNB_scratch:
    def __init__(self, alpha=1, binarize=0.5):
        self.alpha = alpha
        self.binarize = binarize
    def fit(self, X_train, y_train):
        # m = no. of instances
        # n = no. of features
        m, n = X_train.shape
        self._classes = np.unique(y_train)
        n_classes = len(self._classes)

        #convert X_train to binary with threshold at 
        X_train = np.where(X_train > self.binarize, 1, 0)

        # init: Prior & Likelihood
        self._priors = np.zeros(n_classes)
        self._likelihoods = np.zeros((n_classes, n))

        for idx, c in enumerate(self._classes):
            # X_train_c =  matrix containing only instances with class c
            X_train_c = X_train[c == y_train]
            # compute priorprobability of class c as no. of instances of class c
            # over all instances.
            self._priors[idx] = X_train_c.shape[0] / m 

            # compute likelihoods with self.alpha for smoothing
            self._likelihoods[idx, :] = ((X_train_c.sum(axis=0)) + self.alpha) / (X_train_c.shape[0] + 2*self.alpha)
          
    def predict(self, X_test):
        return [self._predict(x_test) for x_test in X_test]
    def _predict(self, x_test):

        x_test = np.where(x_test > self.binarize, 1, 0)

        # Calculate posterior for each class
        posteriors = []
        for idx, c in enumerate(self._classes):
            prior_c = np.log(self._priors[idx])
            likelihoods_c = self.calc_likelihood(self._likelihoods[idx,:], x_test)
            posteriors_c = np.sum(likelihoods_c) + prior_c
            posteriors.append(posteriors_c)
        return self._classes[np.argmax(posteriors)]
    def calc_likelihood(self, cls_likeli, x_test):
        return np.log(cls_likeli) * x_test + np.log(1-cls_likeli) * (1-x_test)
    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return np.sum(y_pred == y_test)/len(y_test)