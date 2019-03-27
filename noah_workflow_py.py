import pandas as pd
import numpy as np
import xgboost as xgb
from statsmodels.tools.eval_measures import rmse
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from matplotlib import pyplot as plt

################################################## DATA IMPORT ##################################################
# load the data
file = 'df_features_1.csv' # THIS SHOULD BE REPLACED BY THE TRAINING DATA FOR THE CITY
path = 'C:/Users/nbranham/Documents/NYU_MSBA/Capstone_Scripts/'
data = pd.read_csv(path+file)
data = data.dropna()

# filter the data for NYC
data = data.loc[data['City']=='NYC']
data = data.drop(['TotalDeaths','TotalInjuries','PedeDeaths','PedeInjuries','Collisions','GEOID','City'], axis=1)

################################################## SPLIT DATA ###################################################
# split the data into features and target variable
X = data.drop('Casualties', axis=1)
y = data['Casualties']

# split the data into train test split using 30% for test data
X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.3)

#################################################### XGBOOST ####################################################
# Convert the training and testing sets into DMatrixes: DM_train, DM_test
DM_train = xgb.DMatrix(X_train, y_train)
DM_test =  xgb.DMatrix(X_test, y_test)

params = {"objective":"reg:linear",
          "max_depth":5, # effects how each tree is allow to grow during any given boosting round (must be positive)
          "n_estimators":500,
          "colsample_bytree":.5} # the fraction of features you can select from (must be between 0 and 1)

# Train the model: xg_reg
xg_reg = xgb.train(params = params, dtrain=DM_train, num_boost_round=5)

# Now let's compute RMSE using 5-fold Cross-Validation
cv_results = xgb.cv(dtrain=DM_train, params=params, nfold=5, num_boost_round=5, metrics="rmse", as_pandas=True, seed=123)
rmse_5cv = (cv_results['test-rmse-mean']).tail(1)

method_name = 'XGBoost'
print('Method: %s' %method_name)
print('RMSE on 5-fold CV: %.4f' %rmse_5cv)

# plot feature importance
xgb.plot_importance(xg_reg)
plt.show()
################################################### NEG BINOM ###################################################
# run the model
nyc_negbinom = sm.GLM(y_train, X_train,family=sm.families.NegativeBinomial()).fit()

# print model results on the training data
print(nyc_negbinom.summary())

############################################## K NEAREST NEIGHBORS ##############################################
# Import StandardScaler from scikit-learn
from sklearn.preprocessing import StandardScaler
from math import sqrt
# Create arrays for the features and the response variable
y = data['Casualties'].values
X = data.drop('Casualties', axis=1).values

# Create the scaler
ss = StandardScaler()

# Apply the scaling method to the dataset used for modeling.
X_scaled = ss.fit_transform(X)

# Now we can split the data into train and test splits and run the KNN for each value from 1 to 20 Neighbors
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_scaled, y, test_size = 0.2, random_state=42)

# Import train_test_split, KNeighborsRegressor and mean_squared_error from scikit-learn
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# output the predictions for the K-value
model = KNeighborsRegressor(n_neighbors = 6)
#fit the model
model.fit(X_train2, y_train2)

# For deciding the value of k, plotting the elbow curve every time is be a cumbersome and tedious process.
# You can simply use gridsearch to find the best value.
from sklearn.model_selection import GridSearchCV
params = {'n_neighbors':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}

knn = KNeighborsRegressor()
knn_model = GridSearchCV(knn, params, cv=5)
knn_model.fit(X_train2,y_train2)
print(knn_model.best_params_)
############################################## TEST EACH MODEL ON HOLDOUT ##############################################
# THE DATA FOR THE HOLDOUT WILL NEED TO BE LOADED AT THIS STEP

# XGB
pred_xgb = xg_reg.predict(DM_test)
xgb_rmse = rmse(y_test, pred_xgb)
print("Root Mean Squared Error for XGBoost is: {}".format(xgb_rmse))

# NEG BINOMIAL
preds = nyc_negbinom.predict(X_test)
negbinom_rmse = rmse(y_test, preds)
print("Root Mean Squared Error for Negative Binomial Regression is: {}".format(negbinom_rmse))

# KNN
pred_knn = knn_model.predict(X_test2)
knn_rmse = sqrt(mean_squared_error(y_test2,pred_knn))
print('Root Mean Squared Error for KNN is: {}'.format(knn_rmse))
