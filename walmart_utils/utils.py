'''
This script contains procedures for feature engineering, bagging, and stacking algorithms

'''

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cross_validation import StratifiedKFold

from walmart_utils import paths

path_train = paths.DATA_TRAIN_PATH
path_test=paths.DATA_TEST_PATH
path_sample_submission=paths.SAMPLE_SUBMISSION_PATH
path_submission=paths.SUBMISSION_PATH

# Create various visit features
def Generate_Basic_Features(data):                   
    data['Count'] = data['ScanCount']
    data['Count'][data['ScanCount']<0]=0
    data['FinelineNumber'].fillna(value=9999, inplace=True)
    data['Upc'].fillna(value=-1, inplace=True)
    data['DepartmentDescription'] = preprocessing.LabelEncoder().fit_transform(list(data['DepartmentDescription']))
    data['Weekday'] = preprocessing.LabelEncoder().fit_transform(list(data['Weekday']))

    df = data.groupby(['VisitNumber', 'DepartmentDescription'], as_index=False)['Count'].sum()
    df1 = df.groupby(['VisitNumber'], as_index=False)['Count'].min()
    df2 = df.groupby(['VisitNumber'], as_index=False)['Count'].max()
    df3 = df.groupby(['VisitNumber'], as_index=False)['Count'].mean()
    df1.rename(columns={'Count': 'Min'}, inplace=True)
    df2.rename(columns={'Count': 'Max'}, inplace=True)
    df3.rename(columns={'Count': 'Mean'}, inplace=True)
    data = data.merge(df1, how='left', on=['VisitNumber'], copy=True)
    data = data.merge(df2, how='left', on=['VisitNumber'], copy=True)
    data = data.merge(df3, how='left', on=['VisitNumber'], copy=True)         
    data['Range'] = data['Max'] - data['Min'] 

    df = data[data['DepartmentDescription']==67]
    df = df.groupby(['VisitNumber'], as_index=False)['Count'].count()
    df.rename(columns={'Count': 'Null'}, inplace=True) 
    data = data.merge(df, how='left', on=['VisitNumber'], copy=True) 
    data['Null'].fillna(value=0, inplace=True) 
    data['Null'][data['Null']>0] = 1

    df = data[data['ScanCount']<0]
    df = df.groupby(['VisitNumber'], as_index=False)['Count'].count()
    df.rename(columns={'Count': 'Has_Neg'}, inplace=True) 
    data = data.merge(df, how='left', on=['VisitNumber'], copy=True) 
    data['Has_Neg'].fillna(value=0, inplace=True)
    data['Has_Neg'][data['Has_Neg']>0] = 1

    df = data[data['FinelineNumber']==9999]
    df = df.groupby(['VisitNumber'], as_index=False)['Count'].count()
    df.rename(columns={'Count': 'Missing'}, inplace=True) 
    data = data.merge(df, how='left', on=['VisitNumber'], copy=True) 
    data['Missing'].fillna(value=0, inplace=True)
    data['Missing'][data['Missing']>0] = 1

    df = data.groupby(['VisitNumber',  'FinelineNumber'], as_index=False)['Count'].count()
    df = df.groupby(['VisitNumber'], as_index=False)['Count'].count()
    df.rename(columns={'Count': 'N_Fineline'}, inplace=True) 
    data = data.merge(df, how='left', on=['VisitNumber'], copy=True)
         
    df = data.groupby(['VisitNumber',  'Upc'], as_index=False)['Count'].count()
    df = df.groupby(['VisitNumber'], as_index=False)['Count'].count()
    df.rename(columns={'Count': 'N_Upc'}, inplace=True) 
    data = data.merge(df, how='left', on=['VisitNumber'], copy=True) 

    df = data.groupby(['VisitNumber', 'DepartmentDescription'], as_index=False)['Count'].count()
    df = df.groupby(['VisitNumber'], as_index=False)['Count'].count()
    df.rename(columns={'Count': 'N_Dep'}, inplace=True) 
    data = data.merge(df, how='left', on=['VisitNumber'], copy=True)

    df = data.groupby(['VisitNumber'], as_index=False)['Count'].sum()
    df.rename(columns={'Count': 'Sum'}, inplace=True) 
    data = data.merge(df, how='left', on=['VisitNumber'], copy=True)

    data['Ratio_F_D'] = data['N_Fineline']/data['N_Dep']
    data['Ratio_U_D'] = data['N_Upc']/data['N_Dep'] 
    data['mean_to_min'] = data['Mean']/data['Min']
    data['mean_to_min'][data['Min']==0] = 0
    data['max_to_mean'] = data['Max']/data['Mean']
    data['max_to_mean'][data['Mean']==0] = 0    

    return data
      
def Fineline_Count(data, fineline_list):
    
    new_data=data[['VisitNumber', 'Weekday']].drop_duplicates()
    for i in range(len(fineline_list)):
        df=data[data['FinelineNumber']==fineline_list[i]]
        df = df.groupby(['VisitNumber'], as_index=False)['Count'].sum()
        df.rename(columns={'Count': 'F_%s' % (i)}, inplace=True)
        new_data = new_data.merge(df, how='left', on=['VisitNumber'], copy=True)
        new_data['F_%s' % (i)].fillna(value=0, inplace=True)
    new_data.drop('Weekday', axis=1, inplace=True)    
    return new_data 

def Department_Count_Products(data, department_list):
    
    new_data=data[['VisitNumber', 'Weekday', 'Sum']].drop_duplicates()
    for i in range(len(department_list)):
       df=data[data['DepartmentDescription']==department_list[i]]
       df = df.groupby(['VisitNumber'], as_index=False)['Count'].sum()
       df.rename(columns={'Count': 'D_%s' % (i)}, inplace=True)
       new_data = new_data.merge(df, how='left', on=['VisitNumber'], copy=True)
       new_data['D_%s' % (i)].fillna(value=0, inplace=True)
       new_data['Ratio_%s' % (i)] = new_data['D_%s' % (i)]/new_data['Sum']
       new_data['Ratio_%s' % (i)].fillna(value=0, inplace=True)
    new_data.drop(['Sum', 'Weekday'], axis=1, inplace=True)
    return new_data
    
def Department_Counts_Neg_Products(data, department_list):

    new_data=data[['VisitNumber', 'Weekday']].drop_duplicates()
    for i in range(len(department_list)):
       df=data[(data['DepartmentDescription']==department_list[i]) & (data['ScanCount']<0)]
       df = df.groupby(['VisitNumber'], as_index=False)['ScanCount'].sum()
       df['ScanCount'] = df['ScanCount']*(-1)
       df.rename(columns={'ScanCount': 'D1_%s' % (i)}, inplace=True)
       new_data = new_data.merge(df, how='left', on=['VisitNumber'], copy=True)
       new_data['D1_%s' % (i)].fillna(value=0, inplace=True)
    new_data.drop('Weekday', axis=1, inplace=True)   
    return new_data 
    
def Department_Counts_Multiple_Products(data, department_list):

    new_data=data[['VisitNumber', 'Weekday']].drop_duplicates()  
    for i in range(len(department_list)):
       df=data[(data['DepartmentDescription']==department_list[i]) & (data['ScanCount']>1)]
       df = df.groupby(['VisitNumber'], as_index=False)['ScanCount'].count()
       df.rename(columns={'ScanCount': 'D2_%s' % (i)}, inplace=True)
       new_data = new_data.merge(df, how='left', on=['VisitNumber'], copy=True)
       new_data['D2_%s' % (i)].fillna(value=0, inplace=True)
    new_data.drop('Weekday', axis=1, inplace=True)   
    return new_data 
    
def Department_Counts_Multiple_Rows(data, department_list):

    new_data=data[['VisitNumber', 'Weekday']].drop_duplicates()
    for i in range(len(department_list)):
       df=data[data['DepartmentDescription']==department_list[i]]
       df = df.groupby(['VisitNumber', 'FinelineNumber'], as_index=False)['Upc'].count()
       df[df['Upc']==1]=0
       df = df.groupby(['VisitNumber'], as_index=False)['Upc'].sum()
       df.rename(columns={'Upc': 'D3_%s' % (i)}, inplace=True)
       new_data = new_data.merge(df, how='left', on=['VisitNumber'], copy=True)
       new_data['D3_%s' % (i)].fillna(value=0, inplace=True)
    new_data.drop('Weekday', axis=1, inplace=True)   
    return new_data 

def Department_Counts_Rows_Pos_ScanCount(data, department_list):

    new_data=data[['VisitNumber', 'Weekday']].drop_duplicates()
    for i in range(len(department_list)):
       df=data[(data['DepartmentDescription']==department_list[i]) & (data['ScanCount']>0)]
       df = df.groupby(['VisitNumber'], as_index=False)['ScanCount'].count()
       df = df.groupby(['VisitNumber'], as_index=False)['ScanCount'].sum()
       df.rename(columns={'ScanCount': 'D4_%s' % (i)}, inplace=True)
       new_data = new_data.merge(df, how='left', on=['VisitNumber'], copy=True)
       new_data['D4_%s' % (i)].fillna(value=0, inplace=True)
    new_data.drop('Weekday', axis=1, inplace=True)   
    return new_data 

def Department_Counts_Rows_Neg_Scancount(data, department_list):

    new_data=data[['VisitNumber', 'Weekday']].drop_duplicates()
    for i in range(len(department_list)):
       df=data[(data['DepartmentDescription']==department_list[i]) & (data['ScanCount']<0)]
       df = df.groupby(['VisitNumber'], as_index=False)['Upc'].count()
       df = df.groupby(['VisitNumber'], as_index=False)['Upc'].sum()
       df.rename(columns={'Upc': 'D5_%s' % (i)}, inplace=True)
       new_data = new_data.merge(df, how='left', on=['VisitNumber'], copy=True)
       new_data['D5_%s' % (i)].fillna(value=0, inplace=True)
    new_data.drop('Weekday', axis=1, inplace=True)   
    return new_data 

def Department_Dummies(data, department_list, department_dummies_list):

    new_data=data[['VisitNumber', 'Weekday']].drop_duplicates()
    for i in range(len(department_list)):
       df=data[(data['DepartmentDescription']==department_list[i]) & (data['Sum']>0)]
       df = df.groupby(['VisitNumber'], as_index=False)['Upc'].count()
       df['Upc'][df['Upc']>0] = 1
       df.rename(columns={'Upc': 'D6_%s' % (department_list[i])}, inplace=True)
       new_data = new_data.merge(df, how='left', on=['VisitNumber'], copy=True)
       new_data['D6_%s' % (department_list[i])].fillna(value=0, inplace=True)
    new_data.drop('Weekday', axis=1, inplace=True)
    # Two-way dummy interactions for the most frequent departments (q>=0.60)   
    for i in range(len(department_dummies_list)-1):
        for j in range(i+1, len(department_dummies_list)):
            new_data['D_%s_%s' % (department_dummies_list[i], department_dummies_list[j])] = \
            new_data['D6_%s' % (department_dummies_list[i])]*new_data['D6_%s' % (department_dummies_list[j])]
            
    # Three-way dummy interactions for the most frequent departments    
    #for i in range(len(department_dummies_list)-2):
    #    for j in range(i+1, len(department_dummies_list)-1):
    #        for k in range(i+2, len(department_dummies_list)):
    #           new_data['D_%s_%s_%s' % (department_dummies_list[i], department_dummies_list[j], department_dummies_list[k])]= \
    #           new_data['D6_%s' % (department_dummies_list[i])]*new_data['D6_%s' % (department_dummies_list[j])]* \
    #           new_data['D6_%s' % (department_dummies_list[k])]    
    return new_data          

def Load_Train(log_scale=None):
    train=pd.read_csv(path_train)
    train = train[train['DepartmentDescription']!="HEALTH AND BEAUTY AIDS"]
    train =  Generate_Basic_Features(train)
    df = train.groupby('FinelineNumber', as_index=False)['Count'].sum() 
    df = df[df['Count']>df['Count'].quantile(q=0.3)]
    df.rename(columns={'Count': 'Fineline_Indicator'}, inplace=True)
    train = train.merge(df, how='left', on='FinelineNumber', copy=True)
    train['Fineline_Indicator'].fillna(value=-1, inplace=True)
    fineline_list = list(train['FinelineNumber'][train['Fineline_Indicator']!=-1].unique())
    fineline_list.sort()
    department_list = list(train['DepartmentDescription'].unique())
    department_list.sort()
    df = train.groupby('DepartmentDescription', as_index=False)['ScanCount'].count()
    df = df[df['ScanCount']>df['ScanCount'].quantile(q=0.60)] 
    department_dummies_list = list(df['DepartmentDescription'].unique())
    features = ['Min', 'Max', 'Mean', 'Range', 'N_Fineline', 'N_Upc', 'N_Dep', 'Sum', 'Ratio_F_D', 'Ratio_U_D',
                'mean_to_min', 'max_to_mean']
    if log_scale:
        train[features]=np.log(train[features]+1)
        train1 = np.log(Fineline_Count(train, fineline_list)+1)
        train2 = np.log(Department_Count_Products(train, department_list)+1)
        train3 = np.log(Department_Counts_Neg_Products(train, department_list)+1)
        train4 = np.log(Department_Counts_Multiple_Products(train, department_list)+1)
        train5 = np.log(Department_Counts_Multiple_Rows(train, department_list)+1)
        train6 = np.log(Department_Counts_Rows_Pos_ScanCount(train, department_list)+1)
        train7 = np.log(Department_Counts_Rows_Neg_Scancount(train, department_list)+1)
    else:
        train1 = Fineline_Count(train, fineline_list)
        train2 = Department_Count_Products(train, department_list)
        train3 = Department_Counts_Neg_Products(train, department_list)
        train4 = Department_Counts_Multiple_Products(train, department_list)
        train5 = Department_Counts_Multiple_Rows(train, department_list)
        train6 = Department_Counts_Rows_Pos_ScanCount(train, department_list)
        train7 = Department_Counts_Rows_Neg_Scancount(train, department_list)
    train8 = Department_Dummies(train, department_list, department_dummies_list)
    train.drop_duplicates(['VisitNumber'], inplace=True)              
    merge_list = [train1, train2, train3, train4, train5, train6, train7, train8]
    for i in range(len(merge_list)):
        train = train.merge(merge_list[i], how='inner', on=['VisitNumber'], copy=True)
    
    columns_drop = ['Upc', 'ScanCount', 'DepartmentDescription', 'FinelineNumber', 'Count', 'Fineline_Indicator',
                    'D1_14', 'D1_37', 'D1_46', 'D1_47', 'D2_37', 'D2_47', 'D2_58', 'D3_14', 'D3_37', 'D3_46', 'D3_47',
                    'D5_14', 'D5_37', 'D5_46', 'D5_47']    
    train.drop(columns_drop, axis=1, inplace=True)
    if log_scale:
        days = pd.get_dummies(train['Weekday'])
        train = pd.concat([train, days], axis=1)
        train.drop('Weekday', axis=1, inplace=True)
        
    return train, fineline_list, department_list, department_dummies_list
          
def Load_Test(fineline_list, department_list, department_dummies_list, log_scale=None):
    
    test=pd.read_csv(path_test)    
    test =  Generate_Basic_Features(test)
    features = ['Min', 'Max', 'Mean', 'Range', 'N_Fineline', 'N_Upc', 'N_Dep', 'Sum', 'Ratio_F_D', 'Ratio_U_D',
                'mean_to_min', 'max_to_mean']
    if log_scale:
        test[features]=np.log(test[features]+1)  
        test1 = np.log(Fineline_Count(test, fineline_list)+1)
        test2 = np.log(Department_Count_Products(test, department_list)+1)
        test3 = np.log(Department_Counts_Neg_Products(test, department_list)+1)
        test4 = np.log(Department_Counts_Multiple_Products(test, department_list)+1)
        test5 = np.log(Department_Counts_Multiple_Rows(test, department_list)+1)
        test6 = np.log(Department_Counts_Rows_Pos_ScanCount(test, department_list)+1)
        test7 = np.log(Department_Counts_Rows_Neg_Scancount(test, department_list)+1)
    else:
        test1 = Fineline_Count(test, fineline_list)
        test2 = Department_Count_Products(test, department_list)
        test3 = Department_Counts_Neg_Products(test, department_list)
        test4 = Department_Counts_Multiple_Products(test, department_list)
        test5 = Department_Counts_Multiple_Rows(test, department_list)
        test6 = Department_Counts_Rows_Pos_ScanCount(test, department_list)
        test7 = Department_Counts_Rows_Neg_Scancount(test, department_list)
    train8 = Department_Dummies(test, department_list, department_dummies_list)
    test.drop_duplicates(['VisitNumber'], inplace=True)
    merge_list = [test1, test2, test3, test4, test5, test6, test7, test8]
    for i in range(len(merge_list)):
        test = test.merge(merge_list[i], how='inner', on=['VisitNumber'], copy=True)
    
    columns_drop = ['Upc', 'ScanCount', 'DepartmentDescription', 'FinelineNumber', 'Count','D1_14', 'D1_37', 'D1_46', 
                    'D1_47', 'D2_37', 'D2_47', 'D2_58', 'D3_14', 'D3_37', 'D3_46', 'D3_47', 'D5_14', 'D5_37', 'D5_46', 
                    'D5_47']    
    test.drop(columns_drop, axis=1, inplace=True)
    if log_scale:
        days = pd.get_dummies(test['Weekday'])
        test = pd.concat([test, days], axis=1)
        test.drop('Weekday', axis=1, inplace=True)
    
    return test
    
def save_submission(predictions):
    sample_submission = pd.read_csv(path_sample_submission)
    columns = list(sample_submission.columns) 
    columns.remove('VisitNumber')
    sample_submission[columns] = predictions
    sample_submission.to_csv(path_submission, index=False)       
    
def Bagging(train, test, target, bagging_size, clf):
    
# This procedure performs bagging to stabilize predictions of the Neural Net Classifier
    
    rng = np.random.RandomState(1014)   # set random seed for bagging              
    num_train = train.shape[0]
    num_class = len(np.unique(target))
    preds_bagging = np.zeros((test.shape[0], num_class, bagging_size), dtype=float)
    for n in range(bagging_size):
        sampleSize = int(num_train)
        index_base = rng.randint(num_train, size=sampleSize) # get random indices with replacement
        train_boot=train[index_base]
        y_boot=target[index_base]
        clf.fit(train_boot, y_boot)
        preds = clf.predict_proba(test)
        if preds.shape[1]==37:
            preds = np.insert(preds, 8, np.zeros(preds.shape[0]), axis=1)
        preds_bagging[:, :, n] = preds
    preds_bagging = np.mean(preds_bagging, axis=2)  # compute average of bootstrap predictions
    return preds_bagging 
    
def StackModels(train, test, y, clfs, n_folds, scaler=None): # train data (pd data frame), test data (pd date frame), Target data,
                                                # list of models to stack, number of folders, boolean for scaling

# StackModels() performs Stacked Aggregation on data: it uses n different classifiers to get out-of-fold 
# predictions for target data. It uses the whole training dataset to obtain signal predictions for test.
# This procedure adds n meta-features to both train and test data (where n is number of models to stack).

    print("Generating Meta-features")
    num_class = np.unique(y).shape[0]
    skf = list(StratifiedKFold(y, n_folds))
    if scaler:
        scaler = preprocessing.StandardScaler().fit(train)
        train_sc = scaler.transform(train)
        test_sc = scaler.transform(test)
    else:
        train_sc = train
        test_sc = test
    blend_train = np.zeros((train.shape[0], num_class*len(clfs))) # Number of training data x Number of classifiers
    blend_test = np.zeros((test.shape[0], num_class*len(clfs)))   # Number of testing data x Number of classifiers   
    for j, clf in enumerate(clfs):
        print ('Training classifier [%s]' % (j))
        for i, (tr_index, cv_index) in enumerate(skf):
            
            print ('stacking Fold [%s] of train data' % (i))
            
            # This is the training and validation set (train on 2 folders, predict on a 3d folder)
            X_train = train[tr_index]
            Y_train = y[tr_index]
            X_cv = train[cv_index]
            if scaler:
               scaler_cv = preprocessing.StandardScaler().fit(X_train)
               X_train=scaler_cv.transform(X_train)
               X_cv=scaler_cv.transform(X_cv)
            clf.fit(X_train, Y_train)
            pred = clf.predict_proba(X_cv)
            blend_train[cv_index, j*num_class:(j+1)*num_class] = pred
                
        print('stacking test data')        
        clf.fit(train_sc, y)
        pred = clf.predict_proba(test_sc)

        blend_test[:, j*num_class:(j+1)*num_class] = pred
                   
    return blend_train, blend_test    