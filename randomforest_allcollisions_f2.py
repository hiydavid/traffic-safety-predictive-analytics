### Owner:      David Huang
### Model:      Random Forest
### Version:    df_features_2
### Target:     Total collisions
### Date:       2019-01-22

############################################################ LOAD & PREP

# Load packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set options
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.float_format', '{:,.5f}'.format)

# Load datasets
data = pd.read_csv('df_features_2.csv')

# Setting random state
SEED = 1234

############################################################ WORKFLOW FUNCTIONS

# Load packages
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as MSE

# Review city dataframes
# LA = data[data['City'] == "LA"].dropna(axis = 'rows')
# NY = data[data['City'] == "NYC"].dropna(axis = 'rows')
# DC = data[data['City'] == "DC"].dropna(axis = 'rows')

# Create model workflow that prep, train, and test random forest model
def rando(df, train_city, y_var, n_trees, depth, max_feat):
    
    # Split data into training and test sets
    train = data[data['City'] == train_city].dropna(axis = 'rows')
    test = data[data['City'] == "DC"].dropna(axis = 'rows')
    
    # Split data into 
    y_train = train[y_var]
    y_test = test[y_var]
    X_train = train.drop(['GEOID', 'City', 'Collisions', 'TotalInjuries', 'TotalDeaths'], axis = 1)
    X_test = test.drop(['GEOID', 'City', 'Collisions', 'TotalInjuries', 'TotalDeaths'], axis = 1)
    
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
    fit_score = rf.score(X_train, y_train)
    rmse_score = np.sqrt(MSE(y_test, y_pred))
    print("Model Fit: {:.6f}".format(fit_score))
    print("Model RMSE: {:.2f}".format(rmse_score))
    print("y-Test Mean: {:.2f}".format(np.mean(y_test)))
    print("y-Test Stdev: {:.2f}".format(np.std(y_test)))
    
    # Visualizing features importances
    importances = pd.Series(
    	data = rf.feature_importances_,
    	index = X_train.columns)
    importances_sorted = importances.sort_values()
    importances_sorted.plot(kind = 'barh', color = 'lightgreen')
    plt.title('Features Importances')
    plt.show()

############################################################ RUN MODELS & REVIEW OUTPUTS

# Run random forest model on NYC, Total Injuries
rando(df = data, 
      train_city = "NYC", 
      y_var = "TotalInjuries", 
      n_trees = 500, 
      depth = 5,
      max_feat = 0.5
      )

# Run random forest model on NYC, Total Deaths
rando(df = data, 
      train_city = "NYC", 
      y_var = "TotalDeaths", 
      n_trees = 500, 
      depth = 5,
      max_feat = 0.5
      )

# Run random forest model on LA, Total Injuries
rando(df = data, 
      train_city = "LA", 
      y_var = "TotalInjuries", 
      n_trees = 500, 
      depth = 5,
      max_feat = 0.5
      )

# Run random forest model on LA, Total Deaths
rando(df = data, 
      train_city = "LA", 
      y_var = "TotalDeaths", 
      n_trees = 500, 
      depth = 5,
      max_feat = 0.5
      )
