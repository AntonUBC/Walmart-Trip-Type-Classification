# This script produces custom wrappers for XGBoost and Keras modules (to generate sklearn-like interface)

# Description:

# 1. XGBoost_multilabel: gradient boosting model for estimation of multi-class probabilities. 

# 3. Keras_NN_Classifier: deep feed-forward network with categorical crossentropy objective.

import numpy as np
import theano
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils
import xgboost as xgb
from sklearn import preprocessing
from sklearn.base import BaseEstimator

class XGBoost_multilabel(BaseEstimator):
    def __init__(self, nthread, eta,
                 gamma, max_depth, min_child_weight, max_delta_step,
                 subsample, colsample_bytree, silent, seed,
                 l2_reg, l1_reg, num_round):
        self.silent = silent
        self.nthread = nthread
        self.eta = eta
        self.gamma = gamma
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step
        self.subsample = subsample
        self.silent = silent
        self.colsample_bytree = colsample_bytree
        self.seed = seed
        self.l2_reg = l2_reg
        self.l1_reg = l1_reg
        self.num_round=num_round
        self.num_class = None
        self.model = None

    def fit(self, X, y):
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)
        self.num_classes = np.unique(y).shape[0]
        sf = xgb.DMatrix(X, y)
        params = {"objective": 'multi:softprob',
          "eta": self.eta,
          "gamma": self.gamma,
          "max_depth": self.max_depth,
          "min_child_weight": self.min_child_weight,
          "max_delta_step": self.max_delta_step,
          "subsample": self.subsample,
          "silent": self.silent,
          "colsample_bytree": self.colsample_bytree,
          "seed": self.seed,
          "lambda": self.l2_reg,
          "alpha": self.l1_reg,
          "num_class": self.num_classes}
        self.model = xgb.train(params, sf, self.num_round)

        return self

    def predict_proba(self, X):
        X=xgb.DMatrix(X)
        preds = self.model.predict(X)
        return preds
        
class Keras_NN_Classifier(BaseEstimator):
    def __init__(self, batch_norm, hidden_units, hidden_layers, input_dropout, hidden_dropout, prelu,
                 hidden_activation, batch_size, nb_epoch, optimizer, learning_rate, momentum, decay,
                 rho, epsilon, validation_split):
        self.batch_norm = batch_norm
        self.hidden_units = hidden_units
        self.hidden_layers = hidden_layers
        self.input_dropout = input_dropout
        self.prelu = prelu
        self.hidden_dropout = hidden_dropout
        self.hidden_activation = hidden_activation
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.decay = decay
        self.rho = rho
        self.epsilon = epsilon
        self.validation_split = validation_split
        self.model = None

    def fit(self, X, y):
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y).astype(np.int32)
        y  = np_utils.to_categorical(y)
        self.model = Sequential()
        ## input layer
        nb_classes = y.shape[1]
        self.model.add(Dropout(self.input_dropout))
        ## hidden layers
        first = True
        hidden_layers = self.hidden_layers
        while hidden_layers > 0:
            if first:
                dim = X.shape[1]
                first = False
            else:
                dim = self.hidden_units
            self.model.add(Dense(dim, self.hidden_units, init='uniform'))
            if self.batch_norm:
                self.model.add(BatchNormalization((self.hidden_units,)))
            if self.prelu == True:
                self.model.add(PReLU((self.hidden_units,)))
            else:
                self.model.add(Activation(self.hidden_activation))
            self.model.add(Dropout(self.hidden_dropout))
            hidden_layers -= 1

        ## output layer
        self.model.add(Dense(self.hidden_units, nb_classes, init='uniform'))
        self.model.add(Activation('softmax'))

        ## Optimizers
        if self.optimizer == "sgd":
            sgd = keras.optimizers.SGD(lr=self.learning_rate, decay=self.decay, momentum=self.momentum, nesterov=True)
            self.model.compile(loss='categorical_crossentropy', optimizer=sgd)
        if self.optimizer == "rmsprop":
            rmsprop = keras.optimizers.RMSprop(self.learning_rate, rho=self.rho, epsilon=self.epsilon)
            self.model.compile(loss='categorical_crossentropy', optimizer=rmsprop)
        if self.optimizer == "adagrad":
            adagrad = keras.optimizers.Adagrad(self.learning_rate, epsilon=self.epsilon)
            self.model.compile(loss='categorical_crossentropy', optimizer=adagrad)
        if self.optimizer == "adadelta":
            adadelta = keras.optimizers.Adadelta(self.learning_rate, rho=self.rho, epsilon=self.epsilon)
            self.model.compile(loss='categorical_crossentropy', optimizer=adadelta)
        if self.optimizer == "adam":
            adam = keras.optimizers.Adam(self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=self.epsilon)
            self.model.compile(loss='categorical_crossentropy', optimizer=adam)
                
        self.model.fit(X, y, nb_epoch=self.nb_epoch, batch_size=self.batch_size,
                      validation_split=self.validation_split)
        return self                     

    def predict_proba(self, X):
        preds = self.model.predict_proba(X)
        return preds