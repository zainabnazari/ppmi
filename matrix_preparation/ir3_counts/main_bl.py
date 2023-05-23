import pandas as pd
import numpy as np
from pathlib import Path
import functools

path1=Path("/scratch/znazari/PPMI_ver_sep2022/RNA_Seq_data/star_ir3/counts/")

bl_files=pd.read_csv(path1/"bl.txt", header=None )



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




list_bl_files=[]
for i in range(len(bl_files)):
    list_bl_files=list_bl_files+[pd.read_csv(path1/bl_files.iloc[i][0],skiprows=1,delimiter='\t', index_col=[0])]




for i in range(len(bl_list)):
    list_bl_files[i].columns = list(list_bl_files[i].columns[:-1])+[bl_list[i]]




first_two_elements_merge_bl=list_bl_files[0].iloc[:,-1:].reset_index(drop=False).merge(list_bl_files[1].iloc[:,-1:].reset_index(drop=True), left_index=True, right_index=True)





list_bl_files_conv=[first_two_elements_merge_bl]+list_bl_files[2:]

del list_bl_files


merged_bl=functools.reduce(merge_df,list_bl_files_conv)


del list_bl_files_conv



merged_bl.to_csv("bl_matrix_counts_ir3.txt",  index=False)






