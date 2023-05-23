import pandas as pd
import numpy as np
from pathlib import Path
import functools

path1=Path("/scratch/znazari/PPMI_ver_sep2022/RNA_Seq_data/star_ir2/counts/")

v08_files=pd.read_csv(path1/"v08.txt", header=None )


def function_names(fname):
    tokens=fname.split('.')
    return tokens[1]


def merge(a,b):
    a.iloc[:,-1:].reset_index(drop=False).merge(b.iloc[:,-1:].reset_index(drop=True), left_index=True, right_index=True)
    return a
    
def merge_df(a,b):
    return a.reset_index(drop=True).merge(b.iloc[:,-1:].reset_index(drop=True),left_index=True, right_index=True)





v08_list=[]
for i in range(len(v08_files)):
    v08_list=v08_list+[function_names(v08_files.iloc[i][0])]

    
list_v08_files=[]
for i in range(len(v08_files)):
    list_v08_files=list_v08_files+[pd.read_csv(path1/v08_files.iloc[i][0],skiprows=1,delimiter='\t', index_col=[0])]

    
for i in range(len(v08_list)):
    list_v08_files[i].columns = list(list_v08_files[i].columns[:-1])+[v08_list[i]]


first_two_elements_merge_v08=list_v08_files[0].iloc[:,-1:].reset_index(drop=False).merge(list_v08_files[1].iloc[:,-1:].reset_index(drop=True), left_index=True, right_index=True)




list_v08_files_conv=[first_two_elements_merge_v08]+list_v08_files[2:]



merged_v08=functools.reduce(merge_df,list_v08_files_conv)



merged_v08.to_csv("v08_matrix_counts_ir2.txt",  index=False)




