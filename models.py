import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn import linear_model
import statsmodels.api as sm

def svr_model(df):

    print('svm model...')
    df.dropna(inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :10], df.iloc[:, 11], test_size=0.25,random_state=42)

    model = SVR()
    model.fit(X_train, y_train)

    prediction = model.predict(X_test)
    prediction = np.atleast_2d(prediction).T

    df_prediction = pd.DataFrame(prediction, index=y_test.index, columns=['y_pred'])
    df_prediction['y_obs'] = y_test
    df_prediction['rank_pred'] = ((df_prediction['y_pred'] * (-5.91607)) + 10.5).astype(int) #un-standardize data with stdev and avg of 1-20 sequence to get finish position
    df_prediction['rank_obs'] = ((df_prediction['y_obs'] * (-5.91607)) + 10.5).astype(int)

    print('complete')
    return df_prediction, model

def linear_reg_model(df):
    print('linear model...')
    #df = pd.read_csv(r"df_xy_2023-02-22.csv", index_col=0)
    df.dropna(inplace=True)

    # build/analyze model
    X = df[['pace_lap', 'num_laps', 'num_stints', 'overall_speed', 'soft_fastest_lap', 'medium_fastest_lap',
       'GridPosition_std']]
    y = df['Position_std']

    #X = sm.add_constant(X)
    results = sm.OLS(y, X).fit()
    print(results.summary())

    # return model
    # X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :10], df.iloc[:, 11], test_size=0.25,random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=42)

    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    print(model.coef_)
    prediction = model.predict(X_test)

    prediction = np.atleast_2d(prediction).T

    df_prediction = pd.DataFrame(prediction, index=y_test.index, columns=['y_pred'])
    df_prediction['y_obs'] = y_test
    #plt.scatter(df_prediction['y_obs'], df_prediction['y_pred'])
    df_prediction['rank_pred'] = ((df_prediction['y_pred'] * (-5.91607)) + 10.5).astype(int) #un-standardize data with stdev and avg of 1-20 sequence to get finish position
    df_prediction['rank_obs'] = ((df_prediction['y_obs'] * (-5.91607)) + 10.5).astype(int)

    print('complete')
    return df_prediction, model

def random_forest_model(df):
    print('random forest model')
    #df = pd.read_csv(r"df_xy_2023-02-22.csv", index_col=0)
    df.dropna(inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :10], df.iloc[:, 11], test_size=0.25,random_state=42)

    print('training model...')
    model= RandomForestRegressor(n_estimators=5000, max_features='sqrt', max_depth=50, random_state=18).fit(X_train,y_train)
    prediction = model.predict(X_test)
    mse = mean_squared_error(y_test, prediction)
    print('mse: ', mse)
    rmse = mse ** .5
    print('rmse: ', rmse)

    prediction = np.atleast_2d(prediction).T

    df_prediction = pd.DataFrame(prediction, index=y_test.index, columns=['y_pred'])
    df_prediction['y_obs'] = y_test
    #plt.scatter(df_prediction['y_obs'], df_prediction['y_pred'])
    df_prediction['rank_pred'] = ((df_prediction['y_pred'] * (-5.91607)) + 10.5).astype(int) #un-standardize data with stdev and avg of 1-20 sequence to get finish position
    df_prediction['rank_obs'] = ((df_prediction['y_obs'] * (-5.91607)) + 10.5).astype(int)

    print('complete')
    return df_prediction, model