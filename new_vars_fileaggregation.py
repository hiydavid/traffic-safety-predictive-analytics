# load the required packages
import pandas as pd
import glob 
import os

# Load in the master data file and get the de-duped census tracts and store them in the base dataframe
master = pd.read_csv("C:/Users/n0284436/Documents/NYU_Stern_MSBA/Capstone/Master.1.20.2019.csv")

# create an empty list 
# change this to character
GEOID = pd.DataFrame(master['GEOID'].drop_duplicates(keep='first'))
GEOID['GEOID'] = GEOID['GEOID'].astype(str)

# Change directory 
path = r"C:\Users\n0284436\Documents\NYU_Stern_MSBA\Capstone\New_Vars"

# look for all of the csv files within that folder
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

# for each file in the file dataframe
for filename in file_df['File_Name']: 
    # if the file is not null then
    if pd.notnull(filename): 
        # Load spreadsheet
        xl = pd.ExcelFile(filename)
        # Load a sheet into a DataFrame by name: df1
        df1 = xl.parse('Exported Data')
        df1.loc[:,'GEOID'] *= 100
        # convert x1 geoid to character
        df1['GEOID'] = df1['GEOID'].astype(str)
        # append it to the PL list
        GEOID = pd.merge(GEOID, df1, how='left', on='GEOID')

# change working directory
os.chdir("C:/Users/n0284436/Documents/NYU_Stern_MSBA/Capstone")
# export data to csv
GEOID.to_csv('new_vars.csv')
