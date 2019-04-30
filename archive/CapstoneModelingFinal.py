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
from statsmodels.discrete.discrete_model import NegativeBinomialResults
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

# Set options
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.float_format', '{:,.5f}'.format)

# Set current directory (change depending on which local machine)
path = r"/Users/dhua22/Documents/Jupyter/FileInput"
os.chdir(path)

# Loading data sets for base model
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

# Loading data sets for negative binomial model
nb_nyc = pd.read_csv('negbinom_nyc.csv')
nb_nyc = nb_nyc.fillna(nb_nyc.mean())
nb_la = pd.read_csv('negbinom_la.csv')
nb_la = nb_la.fillna(nb_la.mean())
nb_dc = pd.read_csv('negbinom_dc.csv')
nb_dc = nb_dc.fillna(nb_dc.mean())

# Drop variables
drop_X = ['GEOID', 'City', 'Borough', 'Class', 'CasualtiesPerPop', 'PedeCasualtiesCount',
          'CasualtiesPerPopDens', 'TotalInjuries', 'TotalDeaths', 'Collisions',
          'CasualtiesCount', 'pop']



############################################################ Ranking Functions
# Define ranking function
def panos_ranking(preds, actual):

    # Conditionally convert to series
    if type(preds) == np.ndarray:
        preds = pd.Series(preds)
    else:
        preds = preds
    if type(actual) == np.ndarray:
        actual = pd.Series(actual)
    else:
        actual = actual

    # Create ranking dataframe
    preds = preds.to_frame()
    actual = actual.to_frame()
    rank_df = pd.concat([preds.reset_index(drop = True), actual.reset_index(drop = True)], ignore_index = True, axis = 1)
    rank_df.columns = ['preds','actual']
    rank_df = rank_df.sort_values(by = 'actual', ascending = False)
    rank_df['actual_rank'] = rank_df['actual'].rank(ascending = False)
    rank_df['pred_rank'] = rank_df['preds'].rank(ascending = False)
    rank_df['PerfectScoreTotal'] = rank_df['actual'].cumsum()
    pred_df = rank_df.copy()
    pred_df = pred_df.sort_values(by = 'preds', ascending = False)
    pred_df = pred_df.drop(['PerfectScoreTotal'], axis = 1)
    final_df = pd.concat([pred_df.reset_index(drop = True), rank_df['PerfectScoreTotal'].reset_index(drop = True)], axis = 1)
    final_df['PerfectScore_Remaining_Casualties'] = final_df['actual'].sum() - final_df['PerfectScoreTotal']
    final_df['PerfectScore_Random'] = final_df['PerfectScore_Remaining_Casualties'] / (final_df['actual'].count() - final_df['pred_rank'] + 1)
    final_df['Perfect_GainOverRandom'] = final_df['PerfectScoreTotal'].diff() - final_df['PerfectScore_Random'].shift(1)
    final_df['Perfect_GainOverRandom'].iloc[0] = final_df['PerfectScoreTotal'].iloc[0] - final_df['PerfectScore_Random'].iloc[0]
    final_df['Perfect_TotalGainOverRandom'] = final_df['Perfect_GainOverRandom']
    final_df['Perfect_TotalGainOverRandom'].iloc[1:] = final_df['Perfect_TotalGainOverRandom'].cumsum()
    final_df['Total_Actual'] = final_df['actual'].cumsum()
    final_df['Remaining Casualties'] = final_df['actual'].sum()
    final_df['Remaining Casualties'].iloc[1:] = final_df['actual'].sum() - final_df['Total_Actual'].shift(1)
    final_df['RandomScore'] = final_df['Remaining Casualties'] / (final_df['actual'].count() - final_df['pred_rank']+1)
    final_df['GainOverRandom'] = final_df['actual'] - final_df['RandomScore']
    final_df['TotalGainOverRandom'] = final_df['GainOverRandom']
    final_df['TotalGainOverRandom'].iloc[1:] = final_df['GainOverRandom'].cumsum()
    final_df['Gain_Over_PerfectGain'] = final_df['TotalGainOverRandom'] / final_df['Perfect_TotalGainOverRandom']
    return final_df



############################################################ Ranking Model Function
# Define workflow function
def run_models(city, target_y, k):

    # Define data input
    if city == 'NYCc' and target_y == 'CasualtiesCount':
        df = data_1[data_1['pop'] >= 200]
        X = df.drop(drop_X, axis = 1)
        y = df[target_y]
        strat = df[['Borough', 'Class']].values
        X_train, X_test, y_train, y_test = train_test_split(
                X, y, stratify = strat, test_size = 0.30, random_state = 1234)
    elif city == 'NYCr' and target_y == 'CasualtiesCount':
        df = data_2[data_1['pop'] >= 200]
        X = df.drop(drop_X, axis = 1)
        y = df[target_y]
        strat = df[['Borough', 'Class']].values
        X_train, X_test, y_train, y_test = train_test_split(
                X, y, stratify = strat, test_size = 0.30, random_state = 1234)
    elif city == 'LA' and target_y == 'CasualtiesCount':
        df = data_3
        target_city = df['City'] == city
        train_city = df[-target_city]
        train_city = train_city[train_city['pop'] >= 200]
        test_city = df[target_city]
        test_city = test_city[test_city['pop'] >= 200]
        X_train = train_city.drop(drop_X, axis=1)
        X_test = test_city.drop(drop_X, axis=1)
        y_train = train_city[target_y]
        y_test = test_city[target_y]
    elif city == 'DC' and target_y == 'CasualtiesCount':
        df = data_4
        target_city = df['City'] == city
        train_city = df[-target_city]
        train_city = train_city[train_city['pop'] >= 200]
        test_city = df[target_city]
        X_train = train_city.drop(drop_X, axis=1)
        X_test = test_city.drop(drop_X, axis=1)
        y_train = train_city[target_y]
        y_test = test_city[target_y]
    elif city == 'NYCc' and target_y == 'CasualtiesPerPop':
        df = data_5[data_1['pop'] >= 200]
        X = df.drop(drop_X, axis=1)
        y = df[target_y]
        strat = df[['Borough', 'Class']].values
        X_train, X_test, y_train, y_test = train_test_split(
                X, y, stratify = strat, test_size = 0.30, random_state = 1234)
    elif city == 'NYCr' and target_y == 'CasualtiesPerPop':
        df = data_6[data_1['pop'] >= 200]
        X = df.drop(drop_X, axis=1)
        y = df[target_y]
        strat = df[['Borough', 'Class']].values
        X_train, X_test, y_train, y_test = train_test_split(
                X, y, stratify = strat, test_size = 0.30, random_state = 1234)
    elif city == 'LA' and target_y == 'CasualtiesPerPop':
        df = data_7
        target_city = df['City'] == city
        train_city = df[-target_city]
        train_city = train_city[train_city['pop'] >= 200]
        test_city = df[target_city]
        test_city = test_city[test_city['pop'] >= 200]
        X_train = train_city.drop(drop_X, axis=1)
        X_test = test_city.drop(drop_X, axis=1)
        y_train = train_city[target_y]
        y_test = test_city[target_y]
    elif city == 'DC' and target_y == 'CasualtiesPerPop':
        df = data_8
        target_city = df['City'] == city
        train_city = df[-target_city]
        train_city = train_city[train_city['pop'] >= 200]
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
        n_estimators = 500,
        max_depth = 5,
        max_features = 0.5
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
        n_estimators = 500,
        learning_rate = 0.08,
        gamma = 0,
        colsample_bytree = 0.5,
        max_depth = 5
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
    if city == 'NYCc' and target_y == 'CasualtiesCount':
        print("ITERATION:")
        print("Train & Test on NYC with Census Features")
        print("Target Variable:", target_y)
        print(" ")
    elif city == 'NYCr' and target_y == 'CasualtiesCount':
        print("ITERATION:")
        print("Train & Test on NYC with Census & Road Features")
        print("Target Variable:", target_y)
        print(" ")
    elif city == 'LA' and target_y == 'CasualtiesCount':
        print("ITERATION:")
        print("Train on NYC & Test on LA with Census Features")
        print("Target Variable:", target_y)
        print(" ")
    elif city == 'DC' and target_y == 'CasualtiesCount':
        print("ITERATION:")
        print("Train on NYC & Test on DC with Census Features")
        print("Target Variable:", target_y)
        print(" ")
    elif city == 'NYCc' and target_y == 'CasualtiesPerPop':
        print("ITERATION:")
        print("Train & Test on NYC with Proportioned Census Features")
        print("Target Variable:", target_y)
        print(" ")
    elif city == 'NYCr' and target_y == 'CasualtiesPerPop':
        print("ITERATION:")
        print("Train & Test on NYC with Proportioned Census & Road Features")
        print("Target Variable:", target_y)
        print(" ")
    elif city == 'LA' and target_y == 'CasualtiesPerPop':
        print("ITERATION:")
        print("Train on NYC & Test on LA with Proportioned Census")
        print("Target Variable:", target_y)
        print(" ")
    elif city == 'DC' and target_y == 'CasualtiesPerPop':
        print("ITERATION:")
        print("Train on NYC & Test on DC with Proportioned Census & Road Features")
        print("Target Variable:", target_y)
        print(" ")

    # Print model scores
    print("======================================================================")
    print("MODEL PERFORMANCE")
    print(" ")
    print("TARGET BASELINE")
    print("  Test Set Stdev: {:.2f}".format(np.std(y_test)))
    print("  Test Set MAE: {:.2f}".format(abs(y_test - np.mean(y_test)).mean()))
    print(" ")
    results_dict = {
            'Model' : ['Negative Binomial', 'K-Nearest Neighbors', 'Random Forest', 'XGBoost'],
            'CV RMSE' : ['n/a', np.sqrt(knn_cv), np.sqrt(rf_cv), np.sqrt(xgb_cv)],
            'Prediction RMSE' : [negbinom_rmse, knn_rmse, rf_rmse, xgb_rmse],
            'Prediction MAE' : [negbinom_mae, knn_mae, rf_mae, xgb_mae]
            }
    results_df = pd.DataFrame.from_dict(results_dict)
    print(results_df)
    print(" ")

    # Print variable importance from Random Forest & XGBoost
    print("======================================================================")
    print("FEATURE IMPORTANCE")
    # Random Forest Feature Importance Chart
    importances = pd.Series(
            data = rf.feature_importances_,
            index = X_train.columns
            )
    importances_sorted = importances.sort_values()[-10:]
    importances_sorted.plot(kind = 'barh', color = 'lightblue', figsize = (5, 3))
    plt.ylabel('Features')
    plt.title('Random Forest Top 10 Features')
    plt.show()

    # XGBoost Feature Importance Chart
    fig, ax = plt.subplots(figsize = (5, 3))
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
    plt.legend()
    if city == 'NYCc' and target_y == 'CasualtiesCount':
        plt.title('Train & Test on NYC with Census Features')
    elif city == 'NYCr' and target_y == 'CasualtiesCount':
        plt.title('Train & Test on NYC with Census & Road Features')
    elif city == 'LA' and target_y == 'CasualtiesCount':
        plt.title('Train on NYC & Test on LA with Census Features')
    elif city == 'DC' and target_y == 'CasualtiesCount':
        plt.title('Train on NYC & Test on DC with Census Features')
    elif city == 'NYCc' and target_y == 'CasualtiesPerPop':
        plt.title('Train & Test on NYC with Proportioned Census Features')
    elif city == 'NYCr' and target_y == 'CasualtiesPerPop':
        plt.title('Train & Test on NYC with Proportioned Census & Road Features')
    elif city == 'LA' and target_y == 'CasualtiesPerPop':
        plt.title('Train on NYC & Test on LA with Proportioned Census')
    elif city == 'DC' and target_y == 'CasualtiesPerPop':
        plt.title('Train on NYC & Test on DC with Proportioned Census')
    plt.figure(figsize = (5, 4))
    plt.show()



############################################################ Negative Binomial Model Function
# Define negative binomial model function that produces summary for analysis
def run_negbinom(area, target_y, features):

    # Define features to include
    if features == 'transpo':
        keep = ['pop_dens', 'trav_cars', 'trav_trans', 'trav_motorcycle', 'trav_bike',
                'trav_walk', 'trav_home']
    elif features == 'demog':
        keep = ['pop_dens', 'race_white', 'race_minority', 'female', 'age_genz',
                'age_millenial', 'age_genx', 'age_boomer', 'age_retiree', 'divsep',
                'widowed', 'median_age', 'not_us_citizen', 'median_earnings',
                'edu_lowedu', 'edu_hsged', 'edu_bs', 'edu_grad', 'unemp', 'below_pov']
    elif features == 'road':
        keep = ['road_maxspeed', 'road_meanspeed', 'road_maxlength', 'road_minlength',
                'road_meanlength', 'road_totlanes', 'road_maxlanes', 'road_iri', 'road_bumps',
                'road_aadt', 'road_sumlength', 'road_pci', 'road_pavewidth', 'road_vc', 'road_q']
    elif features == 'census':
        keep = ['pop_dens', 'race_white', 'race_minority', 'female', 'age_genz',
                'age_millenial', 'age_genx', 'age_boomer', 'age_retiree', 'divsep',
                'widowed', 'median_age', 'not_us_citizen', 'median_earnings', 'trav_cars',
                'trav_trans', 'trav_motorcycle', 'trav_bike', 'trav_walk', 'trav_home',
                'edu_lowedu', 'edu_hsged', 'edu_bs', 'edu_grad', 'unemp', 'below_pov']
    elif features == 'hypoth':
        keep = ['pop_dens', 'median_age', 'median_earnings', 'trav_cars', 'trav_trans',
                'trav_bike', 'trav_walk', 'below_pov', 'race_minority']
    elif features == 'custom':
        keep = [custom]

    # Define city input
    if area == 'NYC':
        df = nb_nyc[nb_nyc['pop'] >= 200]
    elif area == 'Bronx':
        df = nb_nyc[nb_nyc['Borough'] == 'Bronx']
        df = df[df['pop'] >= 200]
    elif area == 'Brooklyn':
        df = nb_nyc[nb_nyc['Borough'] == 'Brooklyn']
        df = df[df['pop'] >= 200]
    elif area == 'Manhattan':
        df = nb_nyc[nb_nyc['Borough'] == 'Manhattan']
        df = df[df['pop'] >= 200]
    elif area == 'Queens':
        df = nb_nyc[nb_nyc['Borough'] == 'Queens']
        df = df[df['pop'] >= 200]
    elif area == 'Staten Island':
        df = nb_nyc[nb_nyc['Borough'] == 'Staten Island']
        df = df[df['pop'] >= 200]
    elif area == 'LA':
        df = nb_la
        df = nb_la[nb_la['pop'] >= 200]
    elif area == 'DC':
        df = nb_dc

    # Define data input
    if (target_y == 'CasualtiesCount' or target_y == 'PedeCasualtiesCount' or
        target_y == 'CyclCasualtiesCount' or  target_y == 'MotrCasualtiesCount'):
        df = df[df['Type'] == 'Count']
        X = df[keep]
        y = df[target_y]
    elif (target_y == 'CasualtiesPerPop' or target_y == 'PedeCasualtiesPerPop' or
          target_y == 'CyclCasualtiesPerPop' or target_y == 'MotrCasualtiesPerPop'):
        df = df[df['Type'] == 'Percentage']
        X = df[keep]
        y = df[target_y]
    elif target_y == 'Collisions':
        df = df[df['Type'] == 'Count']
        X = df[keep]
        y = df[target_y]

    ### Train and test negative binomial model
    Xa = sm.add_constant(X)
    negbinom = sm.GLM(y, Xa, family = sm.families.NegativeBinomial()).fit()
    negbinom_pred = negbinom.predict(Xa)
    negbinom_ranking = panos_ranking(negbinom_pred, y)

    ### Print all outputs
    print("==============================================================================")
    print("AIC:", NegativeBinomialResults.aic(negbinom))
    print("==============================================================================")
    print(negbinom.summary())
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
    plt.xlabel('Predicted Rank Position')
    plt.ylabel('Cumulative Gain Score')
    plt.legend()
    plt.figure(figsize = (5, 4))
    plt.show()



############################################################ Run Models
# run_models() instructions:
# city: select 'NYCc' (census only), 'NYCr' (census & road features), 'LA', or 'DC'
# target_y: select 'CasualtiesCount' or 'CasualtiesPerPop'
# k: select number of folds for cross validation

# Run model workflow function
run_models(
    city = 'NYCc',
    target_y = 'CasualtiesPerPop',
    k = 5
)


# run_negbinom() instructions:
# city: select 'NYC', 'Brooklyn', 'Bronx', 'Queens', 'Manhattan', 'Staten Island', 'LA', or 'DC'
# target_y: select 'CasualtiesCount', 'CasualtiesPerPop'
# features: select 'transpo', 'demog', 'census', or 'hypoth'

# Run negative binomial model function
run_negbinom(
    area = 'NYC',
    target_y = 'CasualtiesPerPop',
    features = 'hypoth'
)
