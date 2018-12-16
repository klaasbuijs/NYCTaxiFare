"""
used this script to play around with the data from the Kaggle NYC Taxi Fare Prediction challenge. The final model
ranked 704/1488 with a root mean squared error of 3.544.
"""
import pandas as pd
import numpy as np
import pickle
import matplotlib.pylab as plt
import xgboost as xgb
import multiprocessing as mp

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV, KFold

def loadDataSet(dataSourceTrain, dataSourceTest):
    '''
    this function loads the train and test data into two pandas dataframes
    :param dataSource: string, csv file name and path
    :return: pandas Dataframe with train or test data
    '''
    trainTypes = {'fare_amount': 'float32',
                  'pickup_datetime': 'str',
                  'pickup_longitude': 'float32',
                  'pickup_latitude': 'float32',
                  'dropoff_longitude': 'float32',
                  'dropoff_latitude': 'float32',
                  'passenger_count': 'uint8'}

    testTypes = {'key': 'str',
                  'pickup_datetime': 'str',
                  'pickup_longitude': 'float32',
                  'pickup_latitude': 'float32',
                  'dropoff_longitude': 'float32',
                  'dropoff_latitude': 'float32',
                  'passenger_count': 'uint8'}

    cols = list(trainTypes.keys())
    train_df = pd.read_csv(dataSourceTrain, usecols=cols, dtype=trainTypes, nrows=20_000_000)
    cols = list(testTypes.keys())
    test_df = pd.read_csv(dataSourceTest, usecols=cols, dtype=testTypes)

    return train_df, test_df

def filterDataSet(train_df):
    """ this function cleans up the train data by removing outliers that were identified by visual inspection of data
    plots and reading of some walk-through kernels on Kaggle """
    train_df = train_df[train_df['fare_amount'] > 0]
    train_df = train_df[train_df['passenger_count'] <= 6]
    train_df = train_df[train_df['pickup_latitude'].between(40, 42)]
    train_df = train_df[train_df['pickup_longitude'].between(-75, -72)]
    train_df = train_df[train_df['dropoff_latitude'].between(40, 42)]
    train_df = train_df[train_df['dropoff_longitude'].between(-75, -72)]
    train_df = train_df.dropna(how='any', axis=0)

    return train_df

def addFeatures(train_df):
    """ add additional features to the train and test set """
    #add additional datetime features and drop pickup_datetime column
    train_df['pickup_datetime'] = pd.to_datetime(train_df['pickup_datetime'])
    train_df = train_df.assign(hour= train_df['pickup_datetime'].dt.hour.astype(np.int32))
    train_df = train_df.assign(day= train_df['pickup_datetime'].dt.day.astype(np.int32))
    train_df = train_df.assign(month= train_df['pickup_datetime'].dt.month.astype(np.int32))
    train_df = train_df.assign(weekday= train_df['pickup_datetime'].dt.dayofweek.astype(np.int32))
    #alternatively, slower: train_df['weekday'] = train_df.apply(lambda x: x['pickup_datetime'].weekday(), axis = 1)
    train_df = train_df.drop('pickup_datetime', axis=1)

    #add additional travel distance features
    R = 6371.
    train_df = train_df.assign(xDrop = R*np.multiply(np.cos(train_df.dropoff_latitude), np.cos(train_df.dropoff_longitude)))
    train_df = train_df.assign(yDrop = R*np.multiply(np.cos(train_df.dropoff_latitude), np.sin(train_df.dropoff_longitude)))
    train_df = train_df.assign(zDrop = R*np.sin(train_df.dropoff_latitude))
    train_df = train_df.assign(xPick = R*np.multiply(np.cos(train_df.pickup_latitude), np.cos(train_df.pickup_longitude)))
    train_df = train_df.assign(yPick = R*np.multiply(np.cos(train_df.pickup_latitude), np.sin(train_df.pickup_longitude)))
    train_df = train_df.assign(zPick = R*np.sin(train_df.pickup_latitude))

    train_df = train_df.assign(euclidianDistance=np.sqrt(np.power(np.subtract(train_df.xDrop,train_df.xPick), 2) + np.power(np.subtract(train_df.yDrop, train_df.yPick), 2) + np.power(np.subtract(train_df.zDrop,train_df.zPick), 2)))

    JFK = [-73.8352, -73.7401, 40.6195, 40.6659]
    EWR = [-74.1925, -74.1531, 40.6700, 40.7081]
    LGU = [-73.8895, -73.8550, 40.7664, 40.7931]

    train_df['toJFK'] = np.where((train_df['dropoff_longitude'] > JFK[0]) & (train_df['dropoff_longitude'] < JFK[1]) & (
                train_df['dropoff_latitude'] > JFK[2]) & (train_df['dropoff_latitude'] < JFK[3]), 1, 0)

    train_df['fromJFK'] = np.where((train_df['pickup_longitude'] > JFK[0]) & (train_df['pickup_longitude'] < JFK[1]) & (
                train_df['pickup_latitude'] > JFK[2]) & (train_df['pickup_latitude'] < JFK[3]), 1, 0)

    train_df['toEWR'] = np.where((train_df['dropoff_longitude'] > EWR[0]) & (train_df['dropoff_longitude'] < EWR[1]) & (
            train_df['dropoff_latitude'] > EWR[2]) & (train_df['dropoff_latitude'] < EWR[3]), 1, 0)

    train_df['fromEWR'] = np.where((train_df['pickup_longitude'] > EWR[0]) & (train_df['pickup_longitude'] < EWR[1]) & (
            train_df['pickup_latitude'] > EWR[2]) & (train_df['pickup_latitude'] < EWR[3]), 1, 0)

    train_df['toLGU'] = np.where((train_df['dropoff_longitude'] > LGU[0]) & (train_df['dropoff_longitude'] < LGU[1]) & (
            train_df['dropoff_latitude'] > LGU[2]) & (train_df['dropoff_latitude'] < LGU[3]), 1, 0)

    train_df['fromLGU'] = np.where((train_df['pickup_longitude'] > LGU[0]) & (train_df['pickup_longitude'] < LGU[1]) & (
            train_df['pickup_latitude'] > LGU[2]) & (train_df['pickup_latitude'] < LGU[3]), 1, 0)

    longCenter = -74.0063889
    latCenter = 40.7141667
    xCenter = R*np.multiply(np.cos(latCenter), np.cos(longCenter))
    yCenter = R*np.multiply(np.cos(latCenter), np.sin(longCenter))
    zCenter = R*np.sin(latCenter)

    train_df = train_df.assign(DistanceDropCenter=np.sqrt(
        np.power(np.subtract(train_df.xDrop, xCenter), 2) + np.power(np.subtract(train_df.yDrop, yCenter),
                                                                            2) + np.power(
            np.subtract(train_df.zDrop, zCenter), 2)))

    train_df = train_df.assign(DistancePickCenter=np.sqrt(
        np.power(np.subtract(train_df.xPick, xCenter), 2) + np.power(np.subtract(train_df.yPick, yCenter),
                                                                     2) + np.power(
            np.subtract(train_df.zPick, zCenter), 2)))

    train_df = train_df.drop('dropoff_latitude', axis=1)
    train_df = train_df.drop('dropoff_longitude', axis=1)
    train_df = train_df.drop('pickup_latitude', axis=1)
    train_df = train_df.drop('pickup_longitude', axis=1)

    return train_df

def clusterCoord(train_df):
    '''
    this function adds clustering information to the dataset, usefulness is questionable, done out of curiosity
    :param train_df: combined dataframe with train and test data
    :return: dataframe with cluster data after KMeans clustering of pickup and dropoff location x, y, z-coordinates
    '''


    train_df = train_df.assign(pickCluster=KMeans(n_clusters=8, random_state=0).fit(train_df[['xPick', 'yPick', 'zPick']].values).labels_)
    train_df = train_df.assign(
        dropOffCluster=KMeans(n_clusters=8, random_state=0).fit(train_df[['xDrop', 'yDrop', 'zDrop']].values).labels_)

    train_df = train_df.drop('xPick', axis=1)
    train_df = train_df.drop('yPick', axis=1)
    train_df = train_df.drop('zPick', axis=1)
    train_df = train_df.drop('xDrop', axis=1)
    train_df = train_df.drop('yDrop', axis=1)
    train_df = train_df.drop('zDrop', axis=1)

    return train_df

def categorizeColumns(train_df):
    '''
    Convert columns with categorical data into one-hot encoded columns
    :param train_df:
    :return:
    '''
    categorical_columns = ['month', 'weekday', 'day', 'hour', 'pickCluster', 'dropOffCluster']

    for cat in categorical_columns:
        le = LabelEncoder()
        train_df[cat] = le.fit_transform(train_df[cat])

        train_df = pd.concat([train_df,
                        pd.get_dummies(train_df[cat]).rename(columns=lambda x: cat + '_' + str(x))], axis=1)
        train_df = train_df.drop(cat, axis=1)
    #train_df = pd.get_dummies(train_df, prefix=categorical_columns, columns=categorical_columns)
    return train_df

def scaleColumns(train_df, test_df):
    """ this function scale the distance columns """
    minMaxScaler = MinMaxScaler()

    train_df[['euclidianDistance']] = minMaxScaler.fit_transform(train_df[['euclidianDistance']].as_matrix())
    test_df[['euclidianDistance']] = minMaxScaler.transform(test_df[['euclidianDistance']].as_matrix())

    minMaxScaler = MinMaxScaler()
    train_df[['DistanceDropCenter']] = minMaxScaler.fit_transform(train_df[['DistanceDropCenter']].as_matrix())
    test_df[['DistanceDropCenter']] = minMaxScaler.transform(test_df[['DistanceDropCenter']].as_matrix())

    minMaxScaler = MinMaxScaler()
    train_df[['DistancePickCenter']] = minMaxScaler.fit_transform(train_df[['DistancePickCenter']].as_matrix())
    test_df[['DistancePickCenter']] = minMaxScaler.transform(test_df[['DistancePickCenter']].as_matrix())

    return train_df, test_df

def trainRegressorRandomSearch(train_df, plot_feature_importances=False):
    """ this function trains an XGBRegressor on the train data. Carries out randomized search cross validation with
    various min_child_weight, gamma, subsample, colsample_bytree, and max depth settings """
    numCores = mp.cpu_count()

    X_train = train_df.loc[:, train_df.columns != 'fare_amount']
    y_train = train_df.fare_amount
    #data_dmatrix = xgb.DMatrix(data=X_train, label=y_train)

    cv_params = {
                'min_child_weight': [1, 5, 10], #model complexity
                'gamma': [0.5, 1, 1.5, 2, 5], #model complexity
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'max_depth': [3, 4, 5, 6, 7] #model complexity
                }

    params = {
                    'objective': 'reg:linear',
                    #'colsample_bytree': 0.8,
                    'learning_rate': 0.1, #0.05-0.3
                    'scale_pos_weight' : 1,
                    #'max_depth': 5, #3-10
                    #'min_child_weight': 1,
                    #'gamma' : 0,
                    'n_estimators': 1000,
                    #'subsample' : 0.8,
                    'silent' : 0,
                    'eta': 0.1, #0.01-0.2
                    'nthread': numCores,
                    'tree_method' : 'exact'
    }

    from xgboost import XGBRegressor

    xgb = XGBRegressor(**params)

    folds = 3
    param_comb = 5

    kf = KFold(n_splits=folds, shuffle = True, random_state = 1001)
    random_search = RandomizedSearchCV(xgb,
                                       param_distributions=cv_params,
                                       n_iter=param_comb,
                                       scoring='neg_mean_squared_error',
                                       n_jobs=1,
                                       cv=kf.split(X_train, y_train),
                                       verbose=3,
                                       random_state=1001,
                                       return_train_score=True,
                                       )
    random_search.fit(X_train, y_train)

    results = pd.DataFrame(random_search.cv_results_)
    results.to_csv('xgb-random-grid-search-results-01.csv', index=False)

    if plot_feature_importances is True:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax = xgb.plot_importance(random_search, ax=ax, height=0.8, max_num_features=20)
        ax.grid("off", axis="y")
        plt.savefig('xgBoost_feature_importances.png', dpi=1200)

    return random_search

def trainRegressorRandomSearch2(train_df):
    """ this function trains an XGBRegressor on the train data. Carries out randomized search cross validation with
    various learning rates and subsample settings """
    '''
    Best estimator:
        XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
        colsample_bytree=1.0, eta=0.1, gamma=1, learning_rate=0.05,
        max_delta_step=0, max_depth=6, min_child_weight=5, missing=None,
        n_estimators=1000, n_jobs=1, nthread=8, objective='reg:linear',
        random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
        seed=None, silent=0, subsample=0.8, tree_method='exact')
    '''
    numCores = mp.cpu_count()

    X_train = train_df.loc[:, train_df.columns != 'fare_amount']
    y_train = train_df.fare_amount
    #data_dmatrix = xgb.DMatrix(data=X_train, label=y_train)

    cv_params = dict(learning_rate=[0.05, 0.1, 0.2, 0.3], subsample=[0.6, 0.7, 0.8, 0.9])

    #exact, gpu_exact
    params = dict(objective='reg:linear', min_child_weight=5, max_depth=6, gamma=1, colsample_bytree=1.0,
                  scale_pos_weight=1, n_estimators=1000, silent=0, eta=0.1, nthread=numCores, tree_method='exact',
                  reg_alpha=0, reg_lambda=1)

    from xgboost import XGBRegressor

    folds = 3
    param_comb = 5

    kf = KFold(n_splits=folds, shuffle = True, random_state = 1001)
    randomSearch = RandomizedSearchCV(XGBRegressor(**params),
                                       param_distributions=cv_params,
                                       n_iter=param_comb,
                                       scoring='neg_mean_squared_error',
                                       n_jobs=1,
                                       cv=kf.split(X_train, y_train),
                                       verbose=3,
                                       random_state=1001,
                                       return_train_score=True
                                       )
    randomSearch.fit(X_train, y_train)

    results = pd.DataFrame(randomSearch.cv_results_)
    results.to_csv(path+'xgb-random-grid-search-results-02.csv', index=False)

    print('\n Grid scores:')
    print(randomSearch.cv_results_)
    print('\n Best estimator:')
    print(randomSearch.best_estimator_)
    print('\n Best hyperparameters:')
    print(randomSearch.best_params_)

    return randomSearch

def trainRegressor(train_df, test_df, keys):
    '''
    used this function to train final XGBoost regressor with settings found through various random search
    cross-validation rounds. Predicts output variables for test set and saves them as csv file.
    :param train_df: train dataframe
    :param test_df: test dataframe
    :param keys: keys for linking predicted values to original data
    :return: model (XGBoost trained model)
    '''
    from xgboost import XGBRegressor
    from sklearn.datasets import dump_svmlight_file

    numCores = mp.cpu_count()

    X_train = train_df.loc[:, train_df.columns != 'fare_amount'].as_matrix()
    X_test = test_df.loc[:, train_df.columns != 'fare_amount'].as_matrix()
    y_train = train_df.fare_amount

    dtestMatrix = xgb.DMatrix(X_test)
    dump_svmlight_file(X_train, y_train, 'dtrain.svm', zero_based=True)
    del X_train
    del y_train
    del X_test

    dtrainMatrix = xgb.DMatrix('dtrain.svm')

    #dtrainMatrix = xgb.DMatrix(X_train.values, y_train.values, feature_names=X_train.columns.values)

    params = dict(objective='reg:linear', learning_rate=0.05, subsample=0.8, min_child_weight=5, max_depth=6, gamma=1,
                  colsample_bytree=1.0, scale_pos_weight=1, n_estimators=1000, silent=0, eta=0.1, nthread=numCores,
                  tree_method='exact', eval_metric='rmse', reg_alpha=0, reg_lambda=1, random_state=1001,
                  n_jobs=numCores)

    #found best num_boost_round: 752
    #model = xgb.cv(params=params, dtrain=dtrainMatrix, num_boost_round=3000, nfold=5,
    #                metrics=['rmse'], early_stopping_rounds=100)

    model = xgb.train(params, dtrainMatrix, num_boost_round=752)
    model.dump_model('dump.raw.txt')
    #model.fit(X_train, y_train)

    # Predict from test set
    prediction = model.predict(dtestMatrix)

    # Create submission file
    submission = pd.DataFrame({"key": keys, "fare_amount": prediction.round(2)})
    submission.to_csv('taxi_fare_submission.csv', index=False)
    return model

if __name__ == "__main__":
    print('loaded libraries.')

    dataSourceTrain = 'train.csv'
    dataSourceTest = 'test.csv'

    try:
        train_df, test_df = loadDataSet(dataSourceTrain, dataSourceTest)
        keys = test_df.key
        test_df = test_df.drop('key', axis=1)
        print('loaded csv file.')
    except:
        print('Failed to load data.')

    try:
        train_df = filterDataSet(train_df)
        print('filtered dataframe.')
    except:
        print('Failed to filter data.')

    # combine train and test set to allow for efficient feature addition and categorization
    numRowsTrain = train_df.shape[0]
    combined_df = pd.concat([train_df, test_df], axis=0)

    del train_df
    del test_df

    try:
        numPartitions = mp.cpu_count() - 1
        dfChunks = np.array_split(combined_df, numPartitions)

        pool = mp.Pool(processes=(mp.cpu_count() - 1))
        dfs = pool.map(addFeatures, dfChunks)
        pool.close()
        pool.join()
        combined_df = pd.concat(dfs)

        del dfs

        print('added travel distance features.')
        print('added datetime features.')
    except:
        print('Failed to add features to data.')

    try:
        combined_df = clusterCoord(combined_df)
        print('added clustering features.')
    except:
        print('Failed add clustering features.')

    try:
        combined_df = categorizeColumns(combined_df)
        print('transformed column into one-hot encoded features.')
    except:
        print('Failed to one-hot encode data.')

    train_df = combined_df[:numRowsTrain]  # Up to the last initial training set row
    test_df = combined_df[numRowsTrain:]

    del combined_df

    try:
        train_df, test_df = scaleColumns(train_df, test_df)
        print('scaled euclidian distance.')
    except:
        print('Failed to scale data.')
    print('Finished loading and processing data.')

    path = "\\"

    train_df.reset_index(drop=True).to_feather(path + 'nyc_taxi_data_train.feather')
    test_df.reset_index(drop=True).to_feather(path + 'nyc_taxi_data_test.feather')

    try:
        xgbModel = trainRegressor(train_df, test_df, keys)

        pickle.dump(xgbModel, open("xgboostmodel.pickle.dat", "wb"))

        from xgboost import plot_importance

        fig, ax = plt.subplots(figsize=(12, 8))
        ax = plot_importance(xgbModel, ax=ax, height=0.8, max_num_features=20)
        ax.grid(False, axis="y")
        plt.savefig(path + 'xgBoost_feature_importances.png', dpi=1200)
    except:
        print('Failed to train model.')
