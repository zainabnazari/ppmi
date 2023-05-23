import pandas as pd
import numpy as np
from pathlib import Path
import functools
####################################################################
# This code is written for the extrcction of last column of                               
# the each patients for the base visit.
# Written by Zainab Nazari, last modify: 18.05.2023 15:46
####################################################################


# We put the main path to the folder where 'counts' files are there.
path1=Path("/scratch/znazari/PPMI_ver_sep2022/RNA_Seq_data/star_ir2/counts/")

# We read the file: in the bl.txt is written the name of all files.
bl_files=pd.read_csv(path1/"bl.txt", header=None )

# We define a function where we can take the second phrase seperated by dot. The second phrase 
# is the patient ID. So with this functin we want to get the patient IDs from their file's name
def function_names(fname):
    tokens=fname.split('.')
    return tokens[1]

# here
def merge(a,b):
    a.iloc[:,-1:].reset_index(drop=False).merge(b.iloc[:,-1:].reset_index(drop=True), left_index=True, right_index=True)
    return a
# here    
def merge_df(a,b):
    return a.reset_index(drop=True).merge(b.iloc[:,-1:].reset_index(drop=True),left_index=True, right_index=True)

# here we go through all file names and extract their patient IDs and put into a list
# So here we have the list of patient IDs for those whom they have put the data for their base visit
bl_list=[]
for i in range(len(bl_files)):
    bl_list=bl_list+[function_names(bl_files.iloc[i][0])]


# here we read all the files with with base visit from the count folder where we have all the files
# for all the patients and all the visit.
# here we have a list of all files with base visit. 
list_bl_files=[]
for i in range(len(bl_files)):
    list_bl_files=list_bl_files+[pd.read_csv(path1/bl_files.iloc[i][0],skiprows=1,delimiter='\t', index_col=[0])]


# here  we take the last column of each file? 
for i in range(len(bl_list)):
    list_bl_files[i].columns = list(list_bl_files[i].columns[:-1])+[bl_list[i]]

# first two columns we merege 
first_two_elements_merge_bl=list_bl_files[0].iloc[:,-1:].reset_index(drop=False).merge(list_bl_files[1].iloc[:,-1:].reset_index(drop=True), left_index=True, right_index=True)



list_bl_files_conv=[first_two_elements_merge_bl]+list_bl_files[2:]

del list_bl_files


merged_bl=functools.reduce(merge_df,list_bl_files_conv)


del list_bl_files_conv

merged_bl.to_csv("bl_matrix_counts_ir2.txt",  index=False)
