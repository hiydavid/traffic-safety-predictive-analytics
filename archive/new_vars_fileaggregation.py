# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 15:43:56 2019

@author: n0284436
"""
# load the required packages
import pandas as pd
import glob 
import os

# Load in the master data file and get the de-duped census tracts and store them in the base dataframe
filepath = "C:/Users/n0284436/Documents/NYU_Stern_MSBA/Capstone/Master.1.20.2019.csv"
master = pd.read_csv(filepath)

# create an empty list 
# change this to character
GEOID = pd.DataFrame(master[['GEOID','City']].drop_duplicates(keep='first'))
GEOID['GEOID'] = GEOID['GEOID'].astype(str)
GEOID['City'] = GEOID['City'].astype(str)
NYC = GEOID['City'] == 'NYC'
GEOID_NYC = GEOID[NYC]
GEOID_NYC = GEOID_NYC.loc[:,['GEOID','City']]
LA = GEOID['City'] == 'LA'
GEOID_LA = GEOID[LA]
GEOID_LA = GEOID_LA.loc[:,['GEOID','City']]

# create path to folder where all the files to import reside
path = r"C:\Users\n0284436\Documents\NYU_Stern_MSBA\Capstone\New_Vars"

# look for all of the xlsx files within that folder
allFiles = glob.glob(path + "\*.xlsx")

# create an empty list
data = []

# for loop to look at files within a directory
for i in allFiles: # for file in all of the files
    data.append( # append the file
        {
            # subset just the file name
            'File_Name': i.split(os.sep)[7] 
         }
    )

# convert the list of paths and file name to a data frame
df = pd.DataFrame(data) 

# label the files by City
# split the file name up by the "_" - we are essentially splitting up the file name into pieces
file_df = df['File_Name'].apply(lambda x: pd.Series(x.split('_')))    
# rename the columns to be more understandable 
file_df.columns = ['City','Variable1','Variable2'] 
# add in the full file name again
file_df['File_Name'] = df['File_Name'] 

# create file dataframes for each city
file_la = file_df[file_df['City']=='LA']
file_nyc = file_df[file_df['City']=='NYC']

# change working directory
os.chdir(path)
# for each file in the file dataframe
for filename in file_nyc['File_Name']: 
    # if the file is not null then
    if pd.notnull(filename):
        # Load spreadsheet
        xl = pd.ExcelFile(filename)
        # Load a sheet into a DataFrame by name: df1
        df1 = xl.parse('Exported Data')
        # multiply the tract by 100 to remove decimals and to make GEOID 11 characters
        df1.loc[:,'GEOID'] *= 100
        # convert x1 geoid to character
        df1['GEOID'] = df1['GEOID'].astype(str)
        # append it to the NYC list
        GEOID_NYC = pd.merge(GEOID_NYC, df1, how='left', on='GEOID')

# for each file in the file dataframe
for filename in file_la['File_Name']: 
    # if the file is not null then
    if pd.notnull(filename):
        # Load spreadsheet
        xl = pd.ExcelFile(filename)
        # Load a sheet into a DataFrame by name: df1
        df2 = xl.parse('Exported Data')
        # multiply the tract by 100 to remove decimals and to make GEOID 11 characters
        df2.loc[:,'GEOID'] *= 100
        # convert x1 geoid to character
        df2['GEOID'] = df2['GEOID'].astype(str)
        # append it to the LA list
        GEOID_LA = pd.merge(GEOID_LA, df2, how='left', on='GEOID')

# look for all of the csv files within that folder
allFiles = glob.glob(path + "\*.csv")

# create an empty list
data = []

# for loop to look at files within a directory
for i in allFiles: # for file in all of the files
    data.append( # append the file
        {
            # subset just the file name
            'File_Name': i.split(os.sep)[7] 
         }
    )

# convert the list of paths and file name to a data frame
df = pd.DataFrame(data) 

# label the files by City
# split the file name up by the "_" - we are essentially splitting up the file name into pieces
file_df = df['File_Name'].apply(lambda x: pd.Series(x.split('_')))    
# rename the columns to be more understandable 
file_df.columns = ['City','Variable1','Variable2'] 
# add in the full file name again
file_df['File_Name'] = df['File_Name'] 

# create file dataframes for each city
file_la = file_df[file_df['City']=='LA']
file_nyc = file_df[file_df['City']=='NYC']

# for each file in the file dataframe
for filename in file_nyc['File_Name']: 
    # if the file is not null then
    if pd.notnull(filename):
        # Load spreadsheet
        df3 = pd.read_csv(filename)
        # convert x1 geoid to character
        df3['GEOID'] = df3['GEOID'].astype(str)
        # add .0 to the end of the GEOID for join
        df3['GEOID']=df3['GEOID']+'.0'
        # append it to the NYC list
        GEOID_NYC = pd.merge(GEOID_NYC, df3, how='left', on='GEOID')

# for each file in the file dataframe
for filename in file_la['File_Name']: 
    # if the file is not null then
    if pd.notnull(filename):
        # Load spreadsheet
        df4 = pd.read_csv(filename)
        # convert x1 geoid to character
        df4['GEOID'] = df4['GEOID'].astype(str)
        # add .0 to the end of the GEOID for join
        df4['GEOID']=df4['GEOID']+'.0'
        # append it to the LA list
        GEOID_LA = pd.merge(GEOID_LA, df4, how='left', on='GEOID')

# Append the two cities together
NewVars = GEOID_NYC.append(GEOID_LA, ignore_index=True)
# store the column names in a list
cols = NewVars.dtypes
cols = pd.DataFrame(cols) 
# get the index of each column
cols.insert(0, 'Index', range(0, 0 + len(cols)))
# Re order the columns
NewVarsExport = NewVars.iloc[:, [11,7,37,0,1,2,3,4,5,8,9,10,12,13,14,15,16,17,18,19,20,21,
                                 22,23,24,25,26,27,28,29,30,31,33,34,35,39,40,31,42,43]]

# change working directory
path2 = "C:/Users/n0284436/Documents/NYU_Stern_MSBA/Capstone"
os.chdir(path2)
# export data to csv
NewVarsExport.to_csv('new_vars.csv')
