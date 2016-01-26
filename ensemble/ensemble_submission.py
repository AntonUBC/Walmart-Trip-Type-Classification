# This script generates meta-features and distance features, trains two separate models and ensembles their predictions
# to construct the vector of predicted probabilities for the final submission

import numpy as np
np.random.seed(1337) # for reproducibility
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

# load user modules
from walmart_utils import utils
from wrappers import models

# The 2d layer model is an ensemble of XGBoost and Neural Net Classifiers
# which are trained using the stacked data

Bagging = True
if Bagging==True:
   bagging_size = 50 # set the number of bagging rounds for stabilizing NN predictions

n_folds = 3 # set the number of folders for generating meta-features

def Load_Data_raw():
         
    train, fineline_list, department_list, department_dummies_list = utils.Load_Train()
    test =  utils.Load_Test(fineline_list, department_list, department_dummies_list)
    y = train['TripType']
    le = LabelEncoder()
    y = le.fit_transform(y)
    filter_out_train=['TripType', 'VisitNumber']
    filter_out_test=['VisitNumber']
    train.drop(filter_out_train, axis = 1, inplace = True)
    test.drop(filter_out_test, axis = 1, inplace = True)
    return train, test, y
    
def Load_Data_log():
         
    log_scale = True    
    train, fineline_list, department_list, department_dummies_list = utils.Load_Train(log_scale)
    test =  utils.Load_Test(fineline_list, department_list, department_dummies_list, log_scale)
    filter_out_train=['TripType', 'VisitNumber']
    filter_out_test=['VisitNumber']
    train.drop(filter_out_train, axis = 1, inplace = True)
    test.drop(filter_out_test, axis = 1, inplace = True)
    return train, test    
    

def KNN_stacking(train, test, y):
    
    pca = PCA(n_components=100)
    scaler = StandardScaler()
    train=scaler.fit_transform(train)
    test=scaler.transform(test)
    train=pca.fit_transform(train)
    test=pca.transform(test)
    
    clf1=KNeighborsClassifier(n_neighbors=10, weights='uniform', algorithm='auto', leaf_size=30, 
                              p=2, metric='minkowski', metric_params=None)
                          
    clf2=KNeighborsClassifier(n_neighbors=20, weights='uniform', algorithm='auto', leaf_size=30,
                              p=2, metric='minkowski', metric_params=None)  
                          
    clf3=KNeighborsClassifier(n_neighbors=40, weights='uniform', algorithm='auto', leaf_size=30, 
                              p=2, metric='minkowski', metric_params=None)
                          
    clf4=KNeighborsClassifier(n_neighbors=80, weights='uniform', algorithm='auto', leaf_size=30,  
                              p=2, metric='minkowski', metric_params=None)

    clf5=KNeighborsClassifier(n_neighbors=120, weights='uniform', algorithm='auto', leaf_size=30,
                              p=2, metric='minkowski', metric_params=None) 

    clf6=KNeighborsClassifier(n_neighbors=160, weights='uniform', algorithm='auto', leaf_size=30,
                              p=2, metric='minkowski', metric_params=None) 

    clfs = [clf1, clf2, clf3, clf4, clf5, clf6] 

    train_probs, test_probs = utils.StackModels(train, test, y, clfs, n_folds) 
    
    return train, test, train_probs, test_probs
    

def NN_stacking(train, test, y):
        
    sd_scaler = True
        
    clf1 =  models.Keras_NN_Classifier(batch_norm=True, hidden_units=1024,
                          hidden_layers=2, input_dropout=0.5,
                          prelu=True, hidden_dropout=0.5, hidden_activation='tanh', batch_size=200, #256
                          nb_epoch=40, optimizer='rmsprop', learning_rate=0.001, momentum=None, decay=None,
                          rho=0.9, epsilon=1e-06, validation_split=0.3)
                          
    clf2 = models.Keras_NN_Classifier(batch_norm=True, hidden_units=1024,
                          hidden_layers=2, input_dropout=0.5,
                          prelu=True, hidden_dropout=0.5, hidden_activation='tanh', batch_size=200, #256
                          nb_epoch=40, optimizer='adadelta', learning_rate=1, momentum=None, decay=None,
                          rho=0.95, epsilon=1e-06, validation_split=0) 

    clf3 = models.Keras_NN_Classifier(batch_norm=True, hidden_units=512,
                          hidden_layers=3, input_dropout=0.5,
                          prelu=True, hidden_dropout=0.5, hidden_activation='tanh', batch_size=200, #256
                          nb_epoch=40, optimizer='rmsprop', learning_rate=0.001, momentum=None, decay=None,
                          rho=0.9, epsilon=1e-06, validation_split=0.3)   

    clf4 = models.Keras_NN_Classifier(batch_norm=True, hidden_units=512,
                          hidden_layers=3, input_dropout=0.5,
                          prelu=True, hidden_dropout=0.5, hidden_activation='tanh', batch_size=200, #256
                          nb_epoch=40, optimizer='adadelta', learning_rate=1, momentum=None, decay=None,
                          rho=0.95, epsilon=1e-06, validation_split=0)
                          
    clf5 = LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=5, fit_intercept=True,
                          intercept_scaling=1, class_weight=None, random_state=None,
                          solver='lbfgs', max_iter=200, multi_class='multinomial', verbose=0)    
    
    clfs = [clf1, clf2, clf3, clf4, clf5]                          
   
    train_probs, test_probs = utils.StackModels(train, test, y, clfs, n_folds, sd_scaler) 
    
    return train_probs, test_probs    
    
    
def Trees_stacking(train, test, y):
    
    train = sparse.csr_matrix(train)
    test = sparse.csr_matrix(test)
    
    clf1=models.XGBoost_multilabel(nthread=2, eta=0.08 ,gamma=0.1, max_depth=15,
                           min_child_weight=2,
                           max_delta_step=None,
                           subsample=0.7, colsample_bytree=0.3,
                           silent=1, seed=1301,
                           l2_reg=1.8, l1_reg=0.15, num_round=300)
                          
    clf2=RandomForestClassifier(n_estimators=250, criterion='entropy', max_depth=15, min_samples_split=2,
                            min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=0.6,
                            max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=2,
                            random_state=1301, verbose=0)
                          
    clf3=ExtraTreesClassifier(n_estimators=300, criterion='entropy', max_depth=15,
                             min_samples_split=2, min_samples_leaf=1, 
                             min_weight_fraction_leaf=0.0, max_features=0.5,
                             max_leaf_nodes=None, bootstrap=False, oob_score=False,
                             n_jobs=2, random_state=1301, verbose=0)  
                          
    clf4 = AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy', max_depth=15, min_samples_leaf=1, 
                             min_weight_fraction_leaf=0.0, max_features=0.4, random_state=1301),
                             n_estimators=300, learning_rate=0.07, random_state=1301)                           
    
    clfs = [clf1, clf2, clf3, clf4]
        
    train_probs, test_probs = utils.StackModels(train, test, y, clfs, n_folds) 
    
    return train_probs, test_probs
    
def Get_distances(data, class_vector): # This function computes distance from each observation to nearest neighbors
    dist = []                          # of each class in the space of predicted probabilities. These distances
    l = list(np.unique(class_vector))  # are used along with the meta-features in the second level of estimation.
    for i in l:
       c = data[class_vector==i]
       neigh = NearestNeighbors(n_neighbors=5)
       neigh.fit(c)
       z = neigh.kneighbors(data, n_neighbors=5, return_distance=True)
       d1 = z[0][:, 0]                                                    # dist to the nearest neighbor of each class
       d2 = z[0][:, 0] + z[0][:, 1] + z[0][:, 2]                          # sum of dist to the three nearest neighbor of each class
       d3 = z[0][:, 0] + z[0][:, 1] + z[0][:, 2] + z[0][:, 3] + z[0][:, 4] # sum of dist to the five nearest neighbor of each class
       d = np.vstack((d1, d2, d3)).T
       dist.append(d)
    d = dist[0]
    for i in range(1,37):
        d = np.hstack((d, dist[i]))
    return d

train_raw, test_raw, y = Load_Data_raw()
train_log, test_log = Load_Data_log()
train_pca, test_pca, meta_knn_train, meta_knn_test, y = KNN_stacking(train_log, test_log, y) 
meta_nn_train, meta_nn_test = NN_stacking(train_log, test_log, y)
meta_trees_train, meta_trees_test = Trees_stacking(train_raw, test_raw, y)
num_class = len(np.unique(y))
xgb_train = meta_trees_train[:,0:num_class]
xgb_test = meta_trees_test[:,0:num_class]
train_class=np.argmax(xgb_train, axis=1)
test_class=np.argmax(xgb_test, axis=1)
dist_train = Get_distances(train_pca, train_class)  # Compute distances in PCA space
dist_test = Get_distances(test_pca, test_class)
scaler = StandardScaler()
dist_train_nn = scaler.fit_transform(dist_train)   # scale distance matrix to use in NN classifier
dist_test_nn = scaler.transform(dist_test)
train_trees = np.hstack((meta_knn_train, meta_nn_train, meta_trees_train, dist_train)) 
test_trees = np.hstack((meta_knn_test, meta_nn_test, meta_trees_test, dist_test))  
train_nn = np.hstack((meta_knn_train, meta_nn_train, meta_trees_train, dist_train_nn)) 
test_nn = np.hstack((meta_knn_test, meta_nn_test, meta_trees_test, dist_test_nn))  

clf_xgb = models.XGBoost_multilabel(nthread=6, eta=0.016 ,gamma=1, max_depth=9,
                           min_child_weight=11,
                           max_delta_step=None,
                           subsample=1, colsample_bytree=0.75,
                           silent=1, seed=1301,
                           l2_reg=3, l1_reg=0.2, num_round=800)  

clf_nn =  models.Keras_NN_Classifier(batch_norm=True, hidden_units=512,
                          hidden_layers=2, input_dropout=0.4,
                          prelu=True, hidden_dropout=0.4, hidden_activation=None, batch_size=128, #256
                          nb_epoch=10, optimizer='adam', learning_rate=0.001, momentum=None, decay=None,
                          rho=None, epsilon=1e-08, validation_split=0) #110

clf_xgb.fit(train_trees, y)

preds_xgb = clf_xgb.predict_proba(test_trees)

if (Bagging==True):
   preds_nn = utils.Bagging(train_nn, test_nn, y, bagging_size, clf_nn)
else:
    clf_nn.fit(train_nn, y)
    preds_nn = clf_nn.predict_proba(test_nn)
    

# Compute the weighted probability matrix for final submission
preds_subm = (preds_xgb**0.784)*(preds_nn**0.216)
# Save submission  
utils.save_submission(preds_subm)                                                                                                                                           