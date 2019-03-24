import pandas as pd
import numpy as np
import xgboost
from statsmodels.tools.eval_measures import rmse
from sklearn import cross_validation
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import KFold
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot

# load the data
path = '/Users/ivyhou/Desktop/NYU_Stern_MSBA/Capstone/documents/'
data = pd.read_csv(path+'df_features_1.csv')
data = data.dropna()

# filter the data for NYC
data = data.loc[data['City']=='NYC']
data = data.drop(['TotalDeaths','TotalInjuries','PedeDeaths','PedeInjuries','Collisions','GEOID','City'], axis=1)

# split the data into features and target variable
X = data.drop('Casualties', axis=1).values
y = data['Casualties'].values

# split the data into train test split using 30% for test data
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y ,test_size=0.3)

# Let's try XGboost algorithm to see if we can get better results
xgb = xgboost.XGBRegressor(n_estimators=500, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=5)

# fit the XGboost model on the training data
xgb.fit(X_train,y_train)

# predict using the test data
predictions = xgb.predict(X_test)
results_rmse = rmse(predictions,y_test)
# output the RMSE
print("The test Standard Deviation is: {}".format(np.std(y_test)))
print("Root Mean Squared Error: {}".format(results_rmse))

# Now let's compute RMSE using 10-fold Cross-Validation
kf = KFold(len(X), n_folds=10)
xval_err = 0
for train, test in kf:
    xgb.fit(X[train], y[train])
    p = xgb.predict(X[test])
    e = p-y[test]
    xval_err += np.dot(e,e)

rmse_10cv = np.sqrt(xval_err/len(X))
method_name = 'XGBoost'
print('Method: %s' %method_name)
print('RMSE on training: %.4f' %results_rmse)
print('RMSE on 10-fold CV: %.4f' %rmse_10cv)

# plot feature importance
xgb.plot_importance()
