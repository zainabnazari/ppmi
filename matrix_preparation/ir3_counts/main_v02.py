import pandas as pd
import numpy as np
from pathlib import Path
import functools

path1=Path("/scratch/znazari/PPMI_ver_sep2022/RNA_Seq_data/star_ir3/counts/")

v02_files=pd.read_csv(path1/"v02.txt", header=None )


def function_names(fname):
    tokens=fname.split('.')
    return tokens[1]


def merge(a,b):
    a.iloc[:,-1:].reset_index(drop=False).merge(b.iloc[:,-1:].reset_index(drop=True), left_index=True, right_index=True)
    return a
    
def merge_df(a,b):
    return a.reset_index(drop=True).merge(b.iloc[:,-1:].reset_index(drop=True),left_index=True, right_index=True)



v02_list=[]
for i in range(len(v02_files)):
    v02_list=v02_list+[function_names(v02_files.iloc[i][0])]
   
    

list_v02_files=[]
for i in range(len(v02_files)):
    list_v02_files=list_v02_files+[pd.read_csv(path1/v02_files.iloc[i][0],skiprows=1,delimiter='\t', index_col=[0])]


for i in range(len(v02_list)):
    list_v02_files[i].columns = list(list_v02_files[i].columns[:-1])+[v02_list[i]]


first_two_elements_merge_v02=list_v02_files[0].iloc[:,-1:].reset_index(drop=False).merge(list_v02_files[1].iloc[:,-1:].reset_index(drop=True), left_index=True, right_index=True)


list_v02_files_conv=[first_two_elements_merge_v02]+list_v02_files[2:]

del list_v02_files
merged_v02=functools.reduce(merge_df,list_v02_files_conv)

del list_v02_files_conv

merged_v02.to_csv("v02_matrix_counts_ir3.txt",  index=False)




