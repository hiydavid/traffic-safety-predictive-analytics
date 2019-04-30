# import the required packages
import pandas as pd
from sklearn.model_selection import train_test_split

# set the path to the folder with the files
path = '/Users/ivyhou/Desktop/NYU_Stern_MSBA/Capstone/documents/'
data = pd.read_csv(path+'strat_fold_2.csv')

# split the target variable and the features
X = data.drop('Borough', axis=1)
y = data['Borough']

# train test split using 30% for testing data and stratifying on class
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y,
                                                    test_size=0.30)

# convert the target variable to a DF and
y_train = y_train.to_frame()
train = X_train.join(y_train)
print(train.head())
train.to_csv('training_nyc_normalized.csv')

y_test = y_test.to_frame()
test = X_test.join(y_test)
print(test.head())
test.to_csv('testing_nyc_normalized.csv')
