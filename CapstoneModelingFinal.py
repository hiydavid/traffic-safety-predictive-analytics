### CAPSTONE MODELING FINAL

############################################################ LOAD & PREP
# Load packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

# Load machine learning packages
import xgboost
from xgboost import plot_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from statsmodels.tools.eval_measures import rmse
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

# Set options
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.float_format', '{:,.5f}'.format)

# Set current directory (change depending on which local machine)
path = r"/Users/dhua22/Desktop/MSBA"
path = r"D:/_dhuang/Work/NYU Stern MSBA Work/Capstone/Data/CapstoneModeling"
os.chdir(path)

# Loading data sets
data_1 = pd.read_csv('ny_census.csv')

data_2 = pd.read_csv('ny_census_arcgis.csv')
data_2 = data_2.fillna(data_2.mean())

data_3 = pd.read_csv('ny_la_census.csv')
data_3 = data_3.fillna(data_3.mean())

data_4 = pd.read_csv('ny_dc_census.csv')
data_4 = data_4.fillna(data_4.mean())

data_5 = pd.read_csv('ny_census_perc.csv')

data_6 = pd.read_csv('ny_census_arcgis_perc.csv')
data_6 = data_6.fillna(data_6.mean())

data_7 = pd.read_csv('ny_la_census_perc.csv')
data_7 = data_7.fillna(data_7.mean())

data_8 = pd.read_csv('ny_dc_census_perc.csv')
data_8 = data_8.fillna(data_8.mean())

# Drop variables
drop_X = ['GEOID', 'City', 'Borough', 'Class', 'CasualtiesPerPop', 'PedeCasualtiesCount',
          'CasualtiesPerPopDens', 'TotalInjuries', 'TotalDeaths', 'Collisions',
          'CasualtiesCount', 'pop']

# Target variable
target_y = 'CasualtiesCount'



############################################################ Ranking Functions

### Define ranking function
def panos_ranking(preds, actual):

    # if input data is numpy array convert to series
    if type(preds) == np.ndarray:
        preds = pd.Series(preds)
    else:
        preds = preds
    if type(actual) == np.ndarray:
        actual = pd.Series(actual)
    else:
        actual = actual

    # if input data is series convert input values values to a dataframe
    preds = preds.to_frame()
    actual = actual.to_frame()
    # concatonate the 2 values into a data frame
    rank_df = pd.concat([preds.reset_index(drop=True), actual.reset_index(drop=True)], ignore_index=True, axis=1)
    # rename the columns
    rank_df.columns = ['preds','actual']
    # sort the values by actual rank
    rank_df = rank_df.sort_values(by='actual', ascending=False)
    # rank the actuals
    rank_df['actual_rank'] = rank_df['actual'].rank(ascending=False)
    # rank the preds
    rank_df['pred_rank'] = rank_df['preds'].rank(ascending=False)
    # calculate the perfect score
    rank_df['PerfectScoreTotal'] = rank_df['actual'].cumsum()
    # copy the dataframe
    pred_df = rank_df.copy()
    # sort the values by actual rank
    pred_df = pred_df.sort_values(by='preds', ascending=False)
    # drop the perfectscore column to avoid duplicate columns
    pred_df = pred_df.drop(['PerfectScoreTotal'], axis=1)
    # merge necessary columns together
    final_df = pd.concat([pred_df.reset_index(drop=True),rank_df['PerfectScoreTotal'].reset_index(drop=True)],axis=1)
    # Create PerfectScore_Remaining_Casualties
    final_df['PerfectScore_Remaining_Casualties'] = final_df['actual'].sum()-final_df['PerfectScoreTotal']
    # Create PerfectScore_Random
    final_df['PerfectScore_Random'] = final_df['PerfectScore_Remaining_Casualties']/(final_df['actual'].count()-final_df['pred_rank']+1)
    # Create Perfect_GainOverRandom
    final_df['Perfect_GainOverRandom'] = final_df['PerfectScoreTotal'].diff()-final_df['PerfectScore_Random'].shift(1)
    final_df['Perfect_GainOverRandom'].iloc[0] = final_df['PerfectScoreTotal'].iloc[0]-final_df['PerfectScore_Random'].iloc[0]
    # Create Perfect_TotalGainOverRandom
    final_df['Perfect_TotalGainOverRandom'] = final_df['Perfect_GainOverRandom']
    final_df['Perfect_TotalGainOverRandom'].iloc[1:] = final_df['Perfect_TotalGainOverRandom'].cumsum()
    # Create Total Actual
    final_df['Total_Actual'] = final_df['actual'].cumsum()
    # Create Remaining Casualties
    final_df['Remaining Casualties'] = final_df['actual'].sum()
    final_df['Remaining Casualties'].iloc[1:] = final_df['actual'].sum()-final_df['Total_Actual'].shift(1)
    # Create RandomScore
    final_df['RandomScore'] = final_df['Remaining Casualties']/(final_df['actual'].count()-final_df['pred_rank']+1)
    # Create GainOverRandom
    final_df['GainOverRandom'] = final_df['actual']-final_df['RandomScore']
    # Create TotalGainOverRandom
    final_df['TotalGainOverRandom'] = final_df['GainOverRandom']
    final_df['TotalGainOverRandom'].iloc[1:] = final_df['GainOverRandom'].cumsum()
    # Create Gain_Over_PerfectGain
    final_df['Gain_Over_PerfectGain'] = final_df['TotalGainOverRandom']/final_df['Perfect_TotalGainOverRandom']
    # print the first 5 rows
    return final_df



############################################################ Modeling Functions

### Define workflow function
def run_models(data_i, k, n_trees, depth, max_feat):

    ### Define data input
    if data_i == 'data_1':
        df = data_1
        X = df.drop(drop_X, axis=1)
        y = df[target_y]
        strat = df[['Borough', 'Class']].values
        X_train, X_test, y_train, y_test = train_test_split(
                X, y, stratify = strat, test_size = 0.30, random_state = 1234)
    elif data_i == 'data_2':
        df = data_2
        X = df.drop(drop_X, axis=1)
        y = df[target_y]
        strat = df[['Borough', 'Class']].values
        X_train, X_test, y_train, y_test = train_test_split(
                X, y, stratify = strat, test_size = 0.30, random_state = 1234)
    elif data_i == 'data_3':
        df = data_3
        target_city = df['City'] == 'LA'
        train_city = df[-target_city]
        test_city = df[target_city]
        X_train = train_city.drop(drop_X, axis=1)
        X_test = test_city.drop(drop_X, axis=1)
        y_train = train_city[target_y]
        y_test = test_city[target_y]
    elif data_i == 'data_4':
        df = data_4
        target_city = df['City'] == 'DC'
        train_city = df[-target_city]
        test_city = df[target_city]
        X_train = train_city.drop(drop_X, axis=1)
        X_test = test_city.drop(drop_X, axis=1)
        y_train = train_city[target_y]
        y_test = test_city[target_y]
    elif data_i == 'data_5':
        df = data_5
        X = df.drop(drop_X, axis=1)
        y = df[target_y]
        strat = df[['Borough', 'Class']].values
        X_train, X_test, y_train, y_test = train_test_split(
                X, y, stratify = strat, test_size = 0.30, random_state = 1234)
    elif data_i == 'data_6':
        df = data_6
        X = df.drop(drop_X, axis=1)
        y = df[target_y]
        strat = df[['Borough', 'Class']].values
        X_train, X_test, y_train, y_test = train_test_split(
                X, y, stratify = strat, test_size = 0.30, random_state = 1234)
    elif data_i == 'data_7':
        df = data_7
        target_city = df['City'] == 'LA'
        train_city = df[-target_city]
        test_city = df[target_city]
        X_train = train_city.drop(drop_X, axis=1)
        X_test = test_city.drop(drop_X, axis=1)
        y_train = train_city[target_y]
        y_test = test_city[target_y]
    elif data_i == 'data_8':
        df = data_8
        target_city = df['City'] == 'DC'
        train_city = df[-target_city]
        test_city = df[target_city]
        X_train = train_city.drop(drop_X, axis=1)
        X_test = test_city.drop(drop_X, axis=1)
        y_train = train_city[target_y]
        y_test = test_city[target_y]

    ### Train and test negative binomial model
    negbinom = sm.GLM(
            y_train,
            X_train,
            family = sm.families.NegativeBinomial()
            ).fit()
    negbinom_pred = negbinom.predict(X_test)
    negbinom_rmse = rmse(y_test, negbinom_pred)
    negbinom_mae = MAE(y_test, negbinom_pred)
    negbinom_ranking = panos_ranking(negbinom_pred, y_test)

    ### Train, validate, and test k-nearest neighbors model
    ss = StandardScaler()
    X_train_scaled = ss.fit_transform(X_train)
    X_test_scaled = ss.fit_transform(X_test)
    knn = KNeighborsRegressor(n_neighbors = 6)
    knn_cv = - cross_val_score(
            knn,
            X_train_scaled,
            y_train,
            cv = k,
            scoring = 'neg_mean_squared_error'
            ).mean()
    knn.fit(X_train_scaled, y_train)
    knn_pred = knn.predict(X_test_scaled)
    knn_rmse = np.sqrt(MSE(y_test, knn_pred))
    knn_mae = MAE(y_test, knn_pred)
    knn_ranking = panos_ranking(knn_pred, y_test)

    ### Train, validate, and test random forest model
    rf = RandomForestRegressor(
        n_estimators = n_trees,
        max_depth = depth,
        max_features = max_feat
    )
    rf_cv = - cross_val_score(
            rf,
            X_train,
            y_train,
            cv = k,
            scoring = 'neg_mean_squared_error'
            ).mean()
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_rmse = np.sqrt(MSE(y_test, rf_pred))
    rf_mae = MAE(y_test, rf_pred)
    rf_ranking = panos_ranking(rf_pred, y_test)

    ### Train and test xgboost model
    xgb = xgboost.XGBRegressor(
        n_estimators = n_trees,
        learning_rate = 0.08,
        gamma = 0,
        colsample_bytree = max_feat,
        max_depth = depth
    )
    kfold = KFold(n_splits = k)
    xgb_cv = - cross_val_score(
        xgb,
        X_train,
        y_train,
        cv = kfold,
        scoring = 'neg_mean_squared_error'
        ).mean()
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    xgb_rmse = np.sqrt(MSE(y_test, xgb_pred))
    xgb_mae = MAE(y_test, xgb_pred)
    xgb_ranking = panos_ranking(xgb_pred, y_test)

    # Print iteration titles
    print(" ")
    print("======================================================================")
    if data_i == 'data_1':
        print("ITERATION:")
        print("Train & Test on NYC with Census Features")
        print("Target Variable:", target_y)
        print(" ")
    elif data_i == 'data_2':
        print("ITERATION:")
        print("Train & Test on NYC with Census & Road Features")
        print("Target Variable:", target_y)
        print(" ")
    elif data_i == 'data_3':
        print("ITERATION:")
        print("Train on NYC & Test on LA with Census Features")
        print("Target Variable:", target_y)
        print(" ")
    elif data_i == 'data_4':
        print("ITERATION:")
        print("Train on NYC & Test on DC with Census Features")
        print("Target Variable:", target_y)
        print(" ")
    elif data_i == 'data_5':
        print("ITERATION:")
        print("Train & Test on NYC with Proportioned Census Features")
        print("Target Variable:", target_y)
        print(" ")
    elif data_i == 'data_6':
        print("ITERATION:")
        print("Train & Test on NYC with Proportioned Census & Road Features")
        print("Target Variable:", target_y)
        print(" ")
    elif data_i == 'data_7':
        print("ITERATION:")
        print("Train on NYC & Test on LA with Proportioned Census")
        print("Target Variable:", target_y)
        print(" ")
    elif data_i == 'data_8':
        print("ITERATION:")
        print("Train on NYC & Test on DC with Proportioned Census & Road Features")
        print("Target Variable:", target_y)
        print(" ")
    
    # Print model scores
    print("======================================================================")
    print("MODEL PERFORMANCE")
    print(" ")
    print("TARGET BASELINE")
    print("  Target Stdev: {:.2f}".format(np.std(y_test)))
    print("  Target MAE: {:.2f}".format(abs(y_test - np.mean(y_test)).mean()))
    print(" ")  
    results_dict = {
            'Model' : ['Negative Binomial', 'K-Nearest Neighbors', 'Random Forest', 'XGBoost'],
            'CV RMSE' : ['n/a', np.sqrt(knn_cv), np.sqrt(rf_cv), np.sqrt(xgb_cv)],
            'Prediction RMSE' : [negbinom_rmse, knn_rmse, rf_rmse, xgb_rmse],
            'Prediction MAE' : [negbinom_mae, knn_mae, rf_mae, xgb_mae]
            }
    results_df = pd.DataFrame.from_dict(results_dict)
    print(results_df)

    # Print variable importance from Random Forest & XGBoost
    print("======================================================================")
    print("FEATURE IMPORTANCE")
    # Random Forest Feature Importance Chart
    importances = pd.Series(
            data = rf.feature_importances_,
            index = X_train.columns
            )
    importances_sorted = importances.sort_values()[-9:]
    importances_sorted.plot(kind = 'barh', color = 'lightblue', figsize = (6.5, 3))
    plt.ylabel('Features')
    plt.title('Random Forest Top 10 Features')
    plt.show()
    
    # XGBoost Feature Importance Chart
    fig, ax = plt.subplots(figsize = (6.5, 3))
    plot_importance(
            xgb,
            max_num_features = 10,
            importance_type = "weight",
            title = 'XGBoost Top 10 Features',
            ax = ax)
    plt.show()
    print(" ")

    # Print ranking performance chart
    print("======================================================================")
    print("RANKING PERFORMANCE")
    plt.plot(
        negbinom_ranking['pred_rank'],
        negbinom_ranking['Perfect_TotalGainOverRandom'],
        color ='blue',
        label = 'Perfect Ranking')
    plt.plot(
        negbinom_ranking['pred_rank'],
        negbinom_ranking['TotalGainOverRandom'],
        color ='red',
        label = 'Neg Binom Ranking')
    plt.plot(
        negbinom_ranking['pred_rank'],
        knn_ranking['TotalGainOverRandom'],
        color ='green',
        label = 'K-NN Ranking')
    plt.plot(
        negbinom_ranking['pred_rank'],
        rf_ranking['TotalGainOverRandom'],
        color ='orange',
        label = 'Random Forest Ranking')
    plt.plot(
        negbinom_ranking['pred_rank'],
        xgb_ranking['TotalGainOverRandom'],
        color = 'gray',
        label = 'XGBoost Ranking')
    plt.xlabel('Predicted Rank Position')
    plt.ylabel('Cumulative Gain Score')
    plt.legend(loc = 'upper left')
    if data_i == 'data_1':
        plt.title('Train & Test on NYC with Census Features')
    elif data_i == 'data_2':
        plt.title('Train & Test on NYC with Census & Road Features')
    elif data_i == 'data_3':
        plt.title('Train on NYC & Test on LA with Census Features')
    elif data_i == 'data_4':
        plt.title('Train on NYC & Test on DC with Census Features')
    elif data_i == 'data_5':
        plt.title('Train & Test on NYC with Proportioned Census Features')
    elif data_i == 'data_6':
        plt.title('Train & Test on NYC with Proportioned Census & Road Features')
    elif data_i == 'data_7':
        plt.title('Train on NYC & Test on LA with Proportioned Census')
    elif data_i == 'data_8':
        plt.title('Train on NYC & Test on DC with Proportioned Census')
    plt.figure(figsize = (6.5, 4))
    plt.show()



############################################################ Run Modeling & Anlayze Results

# Train NYC Test NYC with Census Only
run_models(
        data_i = 'data_1',
        k = 5,
        n_trees = 500,
        depth = 5,
        max_feat = 0.50
        )

# Train NYC Test NYC with Census & Road Condition Data
run_models(
        data_i = 'data_2',
        k = 5,
        n_trees = 500,
        depth = 5,
        max_feat = 0.50
        )

# Train NYC Test LA with Census Only
run_models(
        data_i = 'data_3',
        k = 5,
        n_trees = 500,
        depth = 5,
        max_feat = 0.50
        )

# Train NYC Test DC with Census Only
run_models(
        data_i = 'data_4',
        k = 5,
        n_trees = 50,
        depth = 5,
        max_feat = 0.50
        )

# Train & Test on NYC with Proportioned Census Features
run_models(
        data_i = 'data_5',
        k = 5,
        n_trees = 50,
        depth = 5,
        max_feat = 0.50
        )

# Train NYC Test NYC with Proportioned Census & Road Condition Data
run_models(
        data_i = 'data_6',
        k = 5,
        n_trees = 500,
        depth = 5,
        max_feat = 0.50
        )

# Train NYC Test LA with Proportioned Census Only
run_models(
        data_i = 'data_7',
        k = 5,
        n_trees = 500,
        depth = 5,
        max_feat = 0.50
        )

# Train NYC Test DC with Proportioned Census Only
run_models(
        data_i = 'data_8',
        k = 5,
        n_trees = 50,
        depth = 5,
        max_feat = 0.50
        )
