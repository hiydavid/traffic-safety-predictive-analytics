### Owner:      David Huang
### Model:      Random Forest
### City:       New York City
### Date:       2019-02-04

############################################################ CHANGE LOG
# 2019-01-21    Created file
# 2019-01-23    Updated to run different features in one workflow
# 2019-01-28    Added with features_3 data and excluded death-related targets
# 2019-02-02    Added casualties, ability to change director, added new variables
# 2019-02-04    Updated with features_5 and imputation via fillna()

############################################################ LOAD & PREP

# Load packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Set current directory (change to your own director)
path = r"D:\_dhuang\Work\NYU Stern MSBA Work\Capstone\Data\CapstoneModeling"
os.chdir(path)

# Set options
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.float_format', '{:,.5f}'.format)

# Load datasets
f1 = pd.read_csv('df_features_1.csv')       # Count data
f2 = pd.read_csv('df_features_2.csv')       # Normalized by population
f3 = pd.read_csv('df_features_3.csv')       # Normalized by land
f4 = pd.read_csv('df_features_4.csv')       # Ron's new variables
f5 = pd.read_csv('df_features_5.csv')       # Features 1 and 4 combined

# Setting random state
SEED = 1234

############################################################ WORKFLOW FUNCTIONS

# Load packages
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as MSE

# Create model workflow that prep, train, and test random forest model
def rando(df, city, y_var, n_trees, depth, max_feat):
    
    # Split data into training and test sets
    # geo = df[df['City'] == city].dropna(axis = 'rows')    # drop rows with NAs
    geo = df[df['City'] == city].fillna(df.mean())          # fill NAs with average
    X = geo.drop(['GEOID', 'City', 'Collisions', 'Casualties',
                   'PedeInjuries', 'PedeDeaths', 'TotalInjuries', 
                   'TotalDeaths'], axis = 1)
    y = geo[y_var]
    
    # Split data into
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size = 0.4, random_state = SEED)
    
    # Model fitting
    rf = RandomForestRegressor(
        n_estimators = n_trees, 
        max_depth = depth,
        max_features = max_feat,
        random_state = SEED)
    rf.fit(X_train, y_train)
    
    # Predict class probability using fitted model 
    y_pred = rf.predict(X_test)
    
    # Score calculation
    train_score = rf.score(X_train, y_train)
    test_score = rf.score(X_test, y_test)
    rmse_score = np.sqrt(MSE(y_test, y_pred))
    
    # Prin restuls
    print("MODEL OUTPUT ****************************************")
    print("")
    print("Target City: {}".format(geo['City'].unique()))
    print("Target Variable: {}".format(y_var))
    print("Train Score: {:.2f}".format(train_score))
    print("Test Score: {:.2f}".format(test_score))
    print("Model RMSE: {:.2f}".format(rmse_score))
    print("Target Mean: {:.2f}".format(np.mean(y)))
    print("Target Stdev: {:.2f}".format(np.std(y)))
    
    # Visualizing features importances
    importances = pd.Series(
    	data = rf.feature_importances_,
    	index = X_train.columns)
    importances_sorted = importances.sort_values()
    importances_sorted.plot(kind = 'barh', color = 'lightgreen', figsize = (7, 10))
    plt.title('Features Importances')
    plt.show()
    
    # Spaceholder
    print("")
    print("END OF OUTPUT ***************************************")
    print("")

############################################################ NYC, OVERALL CASUALTIES
    
# Run random forest model on NYC, Total Injuries, Feature Set 1
# Train Score 0.57 / Test Score 0.24 / RMSE 81.05
# Top 5: pop_dens, race_minority, trav_trans, age_millenial, trav_walk
# Bottom 5: trav_motorcycle, trav_bike, edu_genz, age_grad, widowed
rando(df = f1, 
      city = "NYC", 
      y_var = "Casualties", 
      n_trees = 500, 
      depth = 5,
      max_feat = 0.50
      )

# Run random forest model on NYC, Total Injuries, Feature Set 2
# Train Score 0.51 / Test Score 0.25 / RMSE 80.31
# Top 5: pop_dens, perc_minority, perc_walk, perc_bs, perc_hsged
# Bottom 5: perc_mbike, perc_bike, perc_genx, perc_widowed, perc_retiree
rando(df = f2,
      city = "NYC", 
      y_var = "Casualties", 
      n_trees = 500, 
      depth = 5,
      max_feat = 0.50
      )

# Run random forest model on NYC, Total Injuries, Feature Set 3
# Train Score 0.52 / Test Score 0.20 / RMSE 82.99
# Top 5: cars_dens, bs_dens, boomber_dens, median_earnings, retiree_dens
# Bottom 5: mbike_dens, bike_dens, female_dens, trans_dens, millenial_dens
rando(df = f3,
      city = "NYC", 
      y_var = "Casualties", 
      n_trees = 500, 
      depth = 5,
      max_feat = 0.5
      )

# Run random forest model on NYC, Total Injuries, Feature Set 4 (Ron's new features)
# Train Score 0.74 / Test Score 0.45 / RMSE 89.04
# Top 5: road_sumlength, sqml, crime_idx, road_aadt, road_maxlanes
# Bottom 5: biz_alc, biz_health, biz_lib, fastfood_6mo, fastfood_freq
rando(df = f4,
      city = "NYC",
      y_var = "Casualties",
      n_trees = 500,
      depth = 5,
      max_feat = 0.50
      )

# Run random forest model on NYC, Total Injuries, Feature Set 5 (f1 and f4 combination)
# Train Score 0.71 / Test Score 0.54 / RMSE 75.08
# Top 5: road_sumlength, road_maxlanes, crime_idx, biz_retail, sqmi
# Bottom 5: trav_motorcycle, biz_alc, trav_bike, pop, age_retiree
rando(df = f5,
      city = "NYC",
      y_var = "Casualties",
      n_trees = 500,
      depth = 5,
      max_feat = 0.50
      )