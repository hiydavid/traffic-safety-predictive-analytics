def bigfatgreek_ranking(preds, actual):
    
    # if input data is numpy array convert to series
    if type(preds) == np.ndarray:
        preds = pd.Series(preds)
    else:
        preds = preds
    if type(actual) == np.ndarray:
        actual = pd.Series(actual)
    else:
        actual = actual
    # if input data is series convert input values values to a dataframe
    preds = preds.to_frame()
    actual = actual.to_frame()
    # concatonate the 2 values into a data frame
    rank_df = pd.concat([preds.reset_index(drop=True), actual.reset_index(drop=True)], ignore_index=True, axis=1)
    # rename the columns
    rank_df.columns = ['preds','actual']
    # sort the values by actual rank
    rank_df = rank_df.sort_values(by='actual', ascending=False)
    # rank the actuals
    rank_df['actual_rank'] = rank_df['actual'].rank(ascending=False)
    # rank the preds
    rank_df['pred_rank'] = rank_df['preds'].rank(ascending=False)
    # calculate the perfect score
    rank_df['PerfectScoreTotal'] = rank_df['actual'].cumsum()
    # copy the dataframe
    pred_df = rank_df.copy()
    # sort the values by actual rank
    pred_df = pred_df.sort_values(by='preds', ascending=False)
    # drop the perfectscore column to avoid duplicate columns
    pred_df = pred_df.drop(['PerfectScoreTotal'], axis=1)
    # merge necessary columns together
    final_df = pd.concat([pred_df.reset_index(drop=True),rank_df['PerfectScoreTotal'].reset_index(drop=True)],axis=1)
    # Create PerfectScore_Remaining_Casualties
    final_df['PerfectScore_Remaining_Casualties'] = final_df['actual'].sum()-final_df['PerfectScoreTotal']
    # Create PerfectScore_Random
    final_df['PerfectScore_Random'] = final_df['PerfectScore_Remaining_Casualties']/(final_df['actual'].count()-final_df['pred_rank']+1)
    # Create Perfect_GainOverRandom
    final_df['Perfect_GainOverRandom'] = final_df['PerfectScoreTotal'].diff()-final_df['PerfectScore_Random'].shift(1)
    final_df['Perfect_GainOverRandom'].iloc[0] = final_df['PerfectScoreTotal'].iloc[0]-final_df['PerfectScore_Random'].iloc[0]
    # Create Perfect_TotalGainOverRandom
    final_df['Perfect_TotalGainOverRandom'] = final_df['Perfect_GainOverRandom']
    final_df['Perfect_TotalGainOverRandom'].iloc[1:] = final_df['Perfect_TotalGainOverRandom'].cumsum()
    # Create Total Actual
    final_df['Total_Actual'] = final_df['actual'].cumsum()
    # Create Remaining Casualties
    final_df['Remaining Casualties'] = final_df['actual'].sum()
    final_df['Remaining Casualties'].iloc[1:] = final_df['actual'].sum()-final_df['Total_Actual'].shift(1)
    # Create RandomScore
    final_df['RandomScore'] = final_df['Remaining Casualties']/(final_df['actual'].count()-final_df['pred_rank']+1)
    # Create GainOverRandom
    final_df['GainOverRandom'] = final_df['actual']-final_df['RandomScore']
    # Create TotalGainOverRandom
    final_df['TotalGainOverRandom'] = final_df['GainOverRandom']
    final_df['TotalGainOverRandom'].iloc[1:] = final_df['GainOverRandom'].cumsum()
    # Create Gain_Over_PerfectGain
    final_df['Gain_Over_PerfectGain'] = final_df['TotalGainOverRandom']/final_df['Perfect_TotalGainOverRandom']
    # print the first 5 rows
    return final_df
