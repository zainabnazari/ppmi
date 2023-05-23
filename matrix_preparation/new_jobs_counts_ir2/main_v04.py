import pandas as pd
import numpy as np
from pathlib import Path
import functools

path1=Path("/scratch/znazari/PPMI_ver_sep2022/RNA_Seq_data/star_ir2/counts/")

bl_files=pd.read_csv(path1/"bl.txt", header=None )
v02_files=pd.read_csv(path1/"v02.txt", header=None )
v04_files=pd.read_csv(path1/"v04.txt", header=None )
v06_files=pd.read_csv(path1/"v06.txt", header=None )
v08_files=pd.read_csv(path1/"v08.txt", header=None )


def function_names(fname):
    tokens=fname.split('.')
    return tokens[1]


def merge(a,b):
    a.iloc[:,-1:].reset_index(drop=False).merge(b.iloc[:,-1:].reset_index(drop=True), left_index=True, right_index=True)
    return a
    
def merge_df(a,b):
    return a.reset_index(drop=True).merge(b.iloc[:,-1:].reset_index(drop=True),left_index=True, right_index=True)






   
v04_list=[]
for i in range(len(v04_files)):
    v02_list=v04_list+[function_names(v04_files.iloc[i][0])]
    

list_v04_files=[]
for i in range(len(v04_files)):
    list_v04_files=list_v04_files+[pd.read_csv(path1/v04_files.iloc[i][0],skiprows=1,delimiter='\t', index_col=[0])]


for i in range(len(v04_list)):
    list_v04_files[i].columns = list(list_v04_files[i].columns[:-1])+[v04_list[i]]




first_two_elements_merge_v04=list_v04_files[0].iloc[:,-1:].reset_index(drop=False).merge(list_v04_files[1].iloc[:,-1:].reset_index(drop=True), left_index=True, right_index=True)





list_v04_files_conv=[first_two_elements_merge_v04]+list_v04_files[2:]
merged_v04=functools.reduce(merge_df,list_v04_files_conv)

merged_v04.to_csv("v04_matrix_counts_ir2.txt",  index=False)


