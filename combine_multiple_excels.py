import os
import numpy as np
path='./'

file_names = []
'''
for dirpath, dirs, files in os.walk(path): # recurse through the current directory
    for dirname in dirs:
        dname = os.path.join(dirpath, dirname)
        #print(dirname)
        #if 'data' not in dname and 'fibrosis' not in dname:  # get the full direcotry name/path:
        onlyfiles = [f for f in os.listdir(dname) if os.path.isfile(os.path.join(dname, f)) and f.endswith('xlsx')]    # check files in a particular directory
        for i in range(len(onlyfiles)):
            file1 = onlyfiles[i]
            #if file.endswith('xlsx'):
            print(file1)
            file_names.append(os.path.join(dname, file1))
'''
# this is when the files are under current directory, the xlsx files are not getting appended to the array
if len(file_names) == 0:
    onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith('xlsx')]    # check files in a particular directory
    for i in range(len(onlyfiles)):
        file1 = onlyfiles[i]
        #if file.endswith('xlsx'):
        print(file1)
        file_names.append(os.path.join(path, file1))

print(file_names)
import pandas as pd
import xlsxwriter
## Method 2 gets all sheets of a given file
columns =  ['A', 'B', 'C', 'D', 'E', 'F', 'G']
#df = pd.DataFrame(columns=columns)
df_total = pd.DataFrame(columns=columns)
writer = pd.ExcelWriter('final_output.xlsx', engine='xlsxwriter')
run=1
for file1 in file_names:                         # loop through Excel files
    if file1.endswith('.xlsx'):
        excel_file = pd.ExcelFile(file1)
        sheets = excel_file.sheet_names
        print(sheets)
        for sheet in sheets:               # loop through sheets inside an Excel file
            df = excel_file.parse(sheet_name = sheet)
            if run != 1:
                a = 'Run {:d}'.format(run)
                new_row = pd.DataFrame({'A':[a], 'B':np.nan, 'C':np.nan, 'D':np.nan, 'E':np.nan, 'F':np.nan, 'G':np.nan})
                df = pd.concat([new_row, df.iloc[:]]).reset_index(drop = True)
            #print('df_total is: \n', df_total)
            #print('df is: \n', df)
            df_total = pd.concat([df_total, df])
            df_total = df_total.append(pd.Series(), ignore_index=True)
            #df_total.append(df)
            #print('df_total is: \n', df_total)
            #df_total.to_excel(writer, index=False, sheet_name=sheet)
    run += 1
#writer.save()
#print('df_total is: \n', df_total)
# create the mean and average
arr = []
for i in range(10):
    k = 4 + i*7 # not 6 + i*7 as seen from final_output.xlsx because the df indexing is different from excel indexing
    print('k {}, cell: {}'.format(k, df_total.iloc[k]['C']))
    #print('k {}, cell: {}'.format(k, df_total.loc[k]['C'])) # did not work
    #print('k {}, cell: {}'.format(k, df_total.at[k, 2])) # on some systems the iloc indexing in giving error. This did not work
    # create an array to hold the average AUCs from all the runs
    arr.append(df_total.iloc[k]['C'])
    #arr.append(df_total.loc[k]['C'])
    #arr.append(df_total.at[k, 2]) # on some systems the iloc is giving error

df = pd.DataFrame(columns=columns)
# create two blank rows
df_total = df_total.append(pd.Series(), ignore_index=True)
df_total = df_total.append(pd.Series(), ignore_index=True)
np_arr = np.asarray(arr)
arr_mean = np.round(np.mean(np_arr), 2)
arr_std = np.round(np.std(np_arr), 2)
# create a new row to calculate the average of average AUCs
new_row = pd.DataFrame({'A':'avg', 'B':[arr_mean], 'C':np.nan, 'D':np.nan, 'E':np.nan, 'F':np.nan, 'G':np.nan})
df = pd.concat([new_row, df.iloc[:]]).reset_index(drop = True)
# create a new row to calculate the std of average AUCs
new_row = pd.DataFrame({'A':'std', 'B':[arr_std], 'C':np.nan, 'D':np.nan, 'E':np.nan, 'F':np.nan, 'G':np.nan})
df = pd.concat([df.iloc[:], new_row]).reset_index(drop = True)
df_total = pd.concat([df_total, df])
df_total = df_total.append(pd.Series(), ignore_index=True)
df_total.to_excel(writer, index=False, sheet_name=sheet)
writer.save()

