import pandas as pd
import numpy as np
from pathlib import Path
import functools

path1=Path("/scratch/znazari/PPMI_ver_sep2022/RNA_Seq_data/star_ir3/counts/")

v06_files=pd.read_csv(path1/"v06.txt", header=None )


def function_names(fname):
    tokens=fname.split('.')
    return tokens[1]


def merge(a,b):
    a.iloc[:,-1:].reset_index(drop=False).merge(b.iloc[:,-1:].reset_index(drop=True), left_index=True, right_index=True)
    return a
    
def merge_df(a,b):
    return a.reset_index(drop=True).merge(b.iloc[:,-1:].reset_index(drop=True),left_index=True, right_index=True)






v06_list=[]
for i in range(len(v06_files)):
    v06_list=v06_list+[function_names(v06_files.iloc[i][0])]


    
v06_list=[]
for i in range(len(v06_files)):
    v06_list=v06_list+[function_names(v06_files.iloc[i][0])]





list_v06_files=[]
for i in range(len(v06_files)):
    list_v06_files=list_v06_files+[pd.read_csv(path1/v06_files.iloc[i][0],skiprows=1,delimiter='\t', index_col=[0])]
    
for i in range(len(v06_list)):
    list_v06_files[i].columns = list(list_v06_files[i].columns[:-1])+[v06_list[i]]
    




first_two_elements_merge_v06=list_v06_files[0].iloc[:,-1:].reset_index(drop=False).merge(list_v06_files[1].iloc[:,-1:].reset_index(drop=True), left_index=True, right_index=True)





list_v06_files_conv=[first_two_elements_merge_v06]+list_v06_files[2:]




merged_v06=functools.reduce(merge_df,list_v06_files_conv)






merged_v06.to_csv("v06_matrix_counts_ir3.txt",  index=False)





