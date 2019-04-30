import pandas as pd
import numpy as np
import xgboost as xgb
from statsmodels.tools.eval_measures import rmse
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from matplotlib import pyplot as plt

# load the data
path = 'C:/Users/nbranham/Documents/NYU_MSBA/Capstone_Scripts/df_features_1.csv'
data = pd.read_csv(path)
data = data.dropna()

# filter the data for NYC
data = data.loc[data['City']=='NYC']
data = data.drop(['TotalDeaths','TotalInjuries','PedeDeaths','PedeInjuries','Collisions','GEOID','City'], axis=1)

# split the data into features and target variable
X = data.drop('Casualties', axis=1)
y = data['Casualties']

# split the data into train test split using 30% for test data
X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.3)

# Convert the training and testing sets into DMatrixes: DM_train, DM_test
DM_train = xgb.DMatrix(X_train, y_train)
DM_test =  xgb.DMatrix(X_test, y_test)

# Create the parameter dictionary: params
# noahparams = {"objective":"reg:linear",
#           "max_depth":5, # effects how each tree is allow to grow during any given boosting round (must be positive)
#           "n_estimators":500,
#           "learning_rate":0.08, # effects how quickly the model fites the residual error using addtional base learners
#           "gamma":0, # min loss reduction to create a new tree split
#           "subsample":0.75, # the total set that can be used for any given boosting round (must be between 0 and 1)
#           "colsample_bytree":1} # the fraction of features you can select from (must be between 0 and 1)

params = {"objective":"reg:linear",
          "max_depth":5, # effects how each tree is allow to grow during any given boosting round (must be positive)
          "n_estimators":500,
          "colsample_bytree":.5} # the fraction of features you can select from (must be between 0 and 1)

# Train the model: xg_reg
xg_reg = xgb.train(params = params, dtrain=DM_train, num_boost_round=5)

# predict using the test data
predictions = xg_reg.predict(DM_test)
results_rmse = rmse(y_test, predictions)
# output the RMSE
print("The test Standard Deviation is: {}".format(np.std(y_test)))
print("Root Mean Squared Error: {}".format(results_rmse))

# Now let's compute RMSE using 5-fold Cross-Validation
cv_results = xgb.cv(dtrain=DM_train, params=params, nfold=5, num_boost_round=5, metrics="rmse", as_pandas=True, seed=123)
rmse_5cv = (cv_results['test-rmse-mean']).tail(1)

method_name = 'XGBoost'
print('Method: %s' %method_name)
print('RMSE on training: %.4f' %results_rmse)
print('RMSE on 5-fold CV: %.4f' %rmse_5cv)

# plot feature importance
xgb.plot_importance(xg_reg)
plt.show()
