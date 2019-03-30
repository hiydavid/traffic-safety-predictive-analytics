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

# Drop variables
drop_X = ['GEOID', 'City', 'Borough', 'Class', 'CasualtiesPerPop',
          'TotalInjuries', 'TotalDeaths', 'Collisions', 'CasualtiesCount',
          'pop_dens', 'pop']

# Target variable
target_y = 'CasualtiesPerPop'



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

    ### Train and test negative binomial model
    negbinom = sm.GLM(
            y_train,
            X_train,
            family = sm.families.NegativeBinomial()
            ).fit()
    negbinom_pred = negbinom.predict(X_test)
    negbinom_rmse = rmse(y_test, negbinom_pred)
    negbinom_mae = MAE(y_test, negbinom_pred)

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

    # Print model scores
    print(" ")
    print("======================================================================")
    if data_i == 'data_1':
        print("ITERATION:")
        print("Train & Test on NYC with Census Features")
        print("Target Variable:", target_y)
        print(" ")
    elif data_i == 'data_2':
        print("ITERATION:")
        print("Train & Test on NYC with Census & Raoad Features")
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
    print("======================================================================")
    print("MODEL PERFORMANCE")
    print(" ")
    print("TARGET BASELINE")
    print("  Target Stdev: {:.2f}".format(np.std(y_test)))
    print("  Target MAE: {:.2f}".format(abs(y_test - np.mean(y_test)).mean()))
    print(" ")
    print("NEGATIVE BINOMIAL")
    print("  Model RMSE: {:.2f}".format(negbinom_rmse))
    print("  Model MAE: {:.2f}".format(negbinom_mae))
    print(" ")
    print("K-NEAREST NEIGHBOR")
    print("  CV RMSE: {:.2f}".format(np.sqrt(knn_cv)))
    print("  Model RMSE: {:.2f}".format(knn_rmse))
    print("  Model MAE: {:.2f}".format(knn_mae))
    print(" ")
    print("RANDOM FOREST")
    print("  CV RMSE: {:.2f}".format(np.sqrt(rf_cv)))
    print("  Model RMSE: {:.2f}".format(rf_rmse))
    print("  Model MAE: {:.2f}".format(rf_mae))
    print(" ")
    print("EXTREME GRADIENT BOOST")
    print("  CV RMSE: {:.2f}".format(np.sqrt(xgb_cv)))
    print("  Model RMSE: {:.2f}".format(xgb_rmse))
    print("  Model MAE: {:.2f}".format(xgb_mae))
    print(" ")

    # Print variable importance from Random Forest & XGBoost
    print("======================================================================")
    print("FEATURE IMPORTANCE")
    importances = pd.Series(
            data = rf.feature_importances_,
            index = X_train.columns
            )
    importances_sorted = importances.sort_values()[-9:]
    importances_sorted.plot(kind = 'barh', color = 'lightblue', figsize = (6, 4))
    plt.title('Random Forest Feature Importance')
    plt.show()
    print(" ")
    plot_importance(
            xgb, 
            max_num_features = 10, 
            importance_type = "weight", 
            title = 'XGBoost Feature Importance')
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