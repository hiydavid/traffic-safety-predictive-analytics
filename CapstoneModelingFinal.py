### CAPSTONE MODELING FINAL

############################################################ LOAD & PREP
# Load packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Load machine learning packages
import xgboost
from xgboost import plot_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import KFold
from sklearn import cross_validation
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from matplotlib import pyplot
from sklearn.model_selection import train_test_split

# Set options
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.float_format', '{:,.5f}'.format)

# Set current directory (change to your own director)
path = r"D:\_dhuang\Work\NYU Stern MSBA Work\Capstone\Data\CapstoneModeling"
os.chdir(path)

# Loading Data
data_1 = pd.read_csv('ny_census.csv')
data_2 = pd.read_csv('ny_census_arcgis.csv')
data_2 = data_2.fillna(data_2.mean())
data_3 = pd.read_csv('ny_la_census.csv')
data_3 = data_3.fillna(data_3.mean())


# Drop variables
drop_X = ['GEOID', 'City', 'Borough', 'Class', 'CasualtiesPerPop', 
          'TotalInjuries', 'TotalDeaths', 'Collisions', 'CasualtiesCount', 
          'pop_dens', 'pop']
target_y = 'CasualtiesPerPop'



############################################################ Modeling Functions

# Define workflow function
def run_models(data_i, n_trees, depth, max_feat):
    
    # Define data input
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
        y = y_test
    
    # Train and test random forest model
    rf = RandomForestRegressor(
        n_estimators = n_trees, 
        max_depth = depth, 
        max_features = max_feat
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_rmse = np.sqrt(MSE(y_test, rf_pred))
    rf_mae = MAE(y_test, rf_pred)
    
    # Train and test xgboost model
    xgb = xgboost.XGBRegressor(
        n_estimators = n_trees, 
        learning_rate = 0.08, 
        gamma = 0,
        subsample = max_feat,
        colsample_bytree = 1, 
        max_depth = depth
    )
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    xgb_rmse = np.sqrt(MSE(y_test, xgb_pred))
    xgb_mae = MAE(y_test, xgb_pred)
    
    # Print model scores
    print("======================================================================")
    print("Random Forest Results:")
    print("Model RMSE: {:.4f}".format(rf_rmse))
    print("Model MAE: {:.4f}".format(rf_mae))
    print("Target Stdev: {:.4f}".format(np.std(y)))
    importances = pd.Series(
    data = rf.feature_importances_,
    index = X_train.columns)
    importances_sorted = importances.sort_values()
    importances_sorted.plot(kind = 'barh', color = 'lightblue', figsize = (6, 4))
    plt.title('Features Importances')
    plt.show()

    print("======================================================================")
    print("XGBoost Results:")
    print("Model RMSE: {:.4f}".format(xgb_rmse))
    print("Model MAE: {:.4f}".format(xgb_mae))
    print("Target Stdev: {:.4f}".format(np.std(y)))
    plot_importance(xgb)
    pyplot.show()
    


############################################################ Run Modeling & Anlayze Results

run_models(
        data_i = 'data_1', 
        n_trees = 1000, 
        depth = 5, 
        max_feat = 0.75
        )

run_models(
        data_i = 'data_2', 
        n_trees = 1000, 
        depth = 5, 
        max_feat = 0.75
        )

run_models(
        data_i = 'data_3', 
        n_trees = 1000, 
        depth = 5, 
        max_feat = 0.75
        )