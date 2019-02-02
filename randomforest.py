### Owner:      David Huang
### Model:      Random Forest
### Date:       2019-02-02

############################################################ CHANGE LOG
# 2019-01-21    Created file
# 2019-01-23    Updated to run different features in one workflow
# 2019-01-28    Added with feature_3 data
#               Excluded target variable related to deaths
# 2019-02-02    Added casualties target and removed other targets
#               Added ability to update and change working directory
#               Tryout Ron's new features on NYC only

############################################################ LOAD & PREP

# Load packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Set current directory (change to your own director)
os.chdir('D:\_dhuang\Work\NYU Stern MSBA Work\Capstone\Data\CapstoneModeling')

# Set options
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.float_format', '{:,.5f}'.format)

# Load datasets
f1 = pd.read_csv('df_features_1.csv')       # Count data
f2 = pd.read_csv('df_features_2.csv')       # Normalized by population
f3 = pd.read_csv('df_features_3.csv')       # Normalized by land

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
    geo = df[df['City'] == city].dropna(axis = 'rows')
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
    importances_sorted.plot(kind = 'barh', color = 'lightgreen')
    plt.title('Features Importances')
    plt.show()
    
    # Spaceholder
    print("")
    print("END OF OUTPUT ***************************************")
    print("")

############################################################ NYC, OVERALL CASUALTIES

# Run random forest model on NYC, Total Injuries, Feature Set 1
# Fit 0.57 / RMSE 80.84
# Top 5: pop_dens, race_minority, trav_trans, age_millenial, trav_walk
# Bottom 5: trav_motorcycle, trav_bike, edu_grad, age_genz, widowed
rando(df = f1, 
      city = "NYC", 
      y_var = "Casualties", 
      n_trees = 500, 
      depth = 5,
      max_feat = 0.50
      )

# Run random forest model on NYC, Total Injuries, Feature Set 2
# Fit 0.51 / RMSE 79.91
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
# Fit 0.52 / RMSE 82.75
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
# Fit XXXX / RMSE XXXXX
# Top 5: 
# Bottom 5: 





############################################################ LA, OVERALL CASUALTIES

# Run random forest model on LA, Total Injuries, Feature Set 1
rando(df = f1, 
      city = "LA", 
      y_var = "Casualties", 
      n_trees = 500, 
      depth = 5,
      max_feat = 0.5
      )

# Run random forest model on LA, Total Injuries, Feature Set 2
rando(df = f2, 
      city = "LA", 
      y_var = "Casualties", 
      n_trees = 500, 
      depth = 5,
      max_feat = 0.5
      )

# Run random forest model on LA, Total Injuries, Feature Set 3
rando(df = f3,
      city = "LA", 
      y_var = "Casualties", 
      n_trees = 500, 
      depth = 5,
      max_feat = 0.5
      )
