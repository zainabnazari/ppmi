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






bl_list=[]
for i in range(len(bl_files)):
    bl_list=bl_list+[function_names(bl_files.iloc[i][0])]


v02_list=[]
for i in range(len(v02_files)):
    v02_list=v02_list+[function_names(v02_files.iloc[i][0])]
   
v04_list=[]
for i in range(len(v04_files)):
    v02_list=v04_list+[function_names(v04_files.iloc[i][0])]
    
v06_list=[]
for i in range(len(v06_files)):
    v02_list=v06_list+[function_names(v06_files.iloc[i][0])]

v08_list=[]
for i in range(len(v08_files)):
    v02_list=v08_list+[function_names(v08_files.iloc[i][0])]

list_bl_files=[]
for i in range(len(bl_files)):
    list_bl_files=list_bl_files+[pd.read_csv(path1/bl_files.iloc[i][0],skiprows=1,delimiter='\t', index_col=[0])]

list_v02_files=[]
for i in range(len(v02_files)):
    list_v02_files=list_v02_files+[pd.read_csv(path1/v02_files.iloc[i][0],skiprows=1,delimiter='\t', index_col=[0])]

list_v04_files=[]
for i in range(len(v04_files)):
    list_v04_files=list_v04_files+[pd.read_csv(path1/v04_files.iloc[i][0],skiprows=1,delimiter='\t', index_col=[0])]

list_v06_files=[]
for i in range(len(v06_files)):
    list_v06_files=list_v06_files+[pd.read_csv(path1/v06_files.iloc[i][0],skiprows=1,delimiter='\t', index_col=[0])]
    
list_v08_files=[]
for i in range(len(v08_files)):
    list_v08_files=list_v08_files+[pd.read_csv(path1/v08_files.iloc[i][0],skiprows=1,delimiter='\t', index_col=[0])]


for i in range(len(bl_list)):
    list_bl_files[i].columns = list(list_bl_files[i].columns[:-1])+[bl_list[i]]

for i in range(len(v02_list)):
    list_v02_files[i].columns = list(list_v02_files[i].columns[:-1])+[v02_list[i]]

for i in range(len(v04_list)):
    list_v04_files[i].columns = list(list_v04_files[i].columns[:-1])+[v04_list[i]]

for i in range(len(v06_list)):
    list_v06_files[i].columns = list(list_v06_files[i].columns[:-1])+[v06_list[i]]
    
for i in range(len(v08_list)):
    list_v8_files[i].columns = list(list_v08_files[i].columns[:-1])+[v08_list[i]]


first_two_elements_merge_bl=list_bl_files[0].iloc[:,-1:].reset_index(drop=False).merge(list_bl_files[1].iloc[:,-1:].reset_index(drop=True), left_index=True, right_index=True)

first_two_elements_merge_v02=list_v02_files[0].iloc[:,-1:].reset_index(drop=False).merge(list_v02_files[1].iloc[:,-1:].reset_index(drop=True), left_index=True, right_index=True)

first_two_elements_merge_v04=list_v04_files[0].iloc[:,-1:].reset_index(drop=False).merge(list_v04_files[1].iloc[:,-1:].reset_index(drop=True), left_index=True, right_index=True)

first_two_elements_merge_v06=list_v06_files[0].iloc[:,-1:].reset_index(drop=False).merge(list_v06_files[1].iloc[:,-1:].reset_index(drop=True), left_index=True, right_index=True)

first_two_elements_merge_v08=list_v08_files[0].iloc[:,-1:].reset_index(drop=False).merge(list_v08_files[1].iloc[:,-1:].reset_index(drop=True), left_index=True, right_index=True)



list_bl_files_conv=[first_two_elements_merge_bl]+list_bl_files[2:]

list_v02_files_conv=[first_two_elements_merge_v02]+list_v02_files[2:]

list_v04_files_conv=[first_two_elements_merge_v4]+list_v04_files[2:]

list_v06_files_conv=[first_two_elements_merge_v06]+list_v06_files[2:]

list_v08_files_conv=[first_two_elements_merge_v08]+list_v08_files[2:]


merged_bl=functools.reduce(merge_df,list_bl_files_conv)

merged_v02=functools.reduce(merge_df,list_v02_files_conv)

merged_v04=functools.reduce(merge_df,list_v04_files_conv)

merged_v06=functools.reduce(merge_df,list_v06_files_conv)

merged_v08=functools.reduce(merge_df,list_v08_files_conv)




merged_bl.to_csv("bl_matrix_counts_ir2.txt",  index=False)

merged_v02.to_csv("v02_matrix_counts_ir2.txt",  index=False)

merged_v04.to_csv("v04_matrix_counts_ir2.txt",  index=False)

merged_v06.to_csv("v06_matrix_counts_ir2.txt",  index=False)

merged_v08.to_csv("v08_matrix_counts_ir2.txt",  index=False)




