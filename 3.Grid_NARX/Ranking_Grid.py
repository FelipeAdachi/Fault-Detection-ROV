#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 17:06:39 2019

@author: felipeadachi
"""


import pandas as pd
import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go
import json

#---------------------#---------------------#---------------------#---------------------#---------------------#

def write_json_content(filename, content):
    with open(filename, 'w') as f:
        f.write(json.dumps(content,indent=4))

#---------------------#---------------------#---------------------#---------------------#---------------------#

targets=['GYROX','GYROY','GYROZ']

plt_index=['Todos','CFS','RF','SW']
subfolders=['ALL_MUTINFO','CFS_MUTINFO','RF_MUTINFO','SW_MUTINFO']
df_all=pd.DataFrame(index=subfolders)

for target in targets:
    path='/home/felipeadachi/Dados/serverlab/Models_GridSearch/'+str(target)+'/'
    df=pd.DataFrame(index=subfolders)
    df_mae=pd.DataFrame(index=subfolders)

    df_std=pd.DataFrame(index=subfolders)
    df_mae_std=pd.DataFrame(index=subfolders)

    df_rank=pd.DataFrame(index=subfolders)
    df_rank_mean=pd.DataFrame(index=subfolders)


    for folder in subfolders:
        i=0
        mse=[]
        np.asarray(mse, dtype=np.float32)
        #print(path+folder+'/log_narx.txt')
        with open(path+folder+'/log_narx.txt','r') as f:
            s=f.read()
        lines=s.splitlines()
        
        
        for line in lines:
            if line.startswith('Metricas'):
                #print(lines[i+1])
                st=''
                j=0
                for param in lines[i+1][1:-1].split(','):
                    if j==1:
                        st=st+str(param[2])
                    if j==2:
                        st=st+str(param[2:])
                    if j==0:
                        st=st+str(param)
                    j+=1
                
                #print(st)
                #df.insert(st)
                #print(lines[i+3])
                mse_line=[]
                mae_line=[]
                
                for mse_f in lines[i+3][1:-1].split(','):
                    mse_line.append(float(mse_f))
                for mae_f in lines[i+5][1:-1].split(','):
                    mae_line.append(float(mae_f))
                
                #print(mse)
                #print(mse_line)
                #print("mean>",np.nanmean(mse_line))
                if st not in df:
                    df[st]=st
                    df_std[st]=st
                    df_rank[st]=st
                if st not in df_mae:
                    df_mae[st]=st
                    df_mae_std[st]=st
                
                
                df[st][folder]=float(np.nanmean(mse_line))
                df_std[st][folder]=float(np.nanstd(mse_line))
                
                df_mae[st][folder]=float(np.nanmean(mae_line))
                df_mae_std[st][folder]=float(np.nanstd(mae_line))
                

                
                #df[folder,st]=np.nanmean(mse_line)
                #df.insert(st)
                #print(df)
        
            i+=1
        #with pd.option_context('display.max_columns',None):
        #    print(df)
    #print(df)    
    #print(df.idxmin(axis=1))
    #print(df.min(axis=0))

    df = df[df.columns.drop(list(df.filter(regex='.3')))]


    with pd.option_context('display.max_columns',None):
        print(df)


    for column in df:
        df_rank[column]=df[column].rank(ascending=1,)

    with pd.option_context('display.max_columns',None):
        print(df_rank)

    df_rank = df_rank[df_rank.columns.drop(list(df_rank.filter(regex='.3')))]

    df_rank_mean['mean']=df_rank.mean(axis=1)
    df_rank_mean['std']=df_rank.std(axis=1)
    df_rank_mean=df_rank_mean.round(2)
    df_rank_mean = df_rank_mean.applymap(str)
    df_all[target]=df_rank_mean['mean']+"$\pm$"+df_rank_mean['std']
    
    df_rank_mean.to_csv('rank_metrics.csv',index=True)
    df_rank_mean.to_latex('rank_metrics.tex',index=True)

df_all.to_latex('all_ranks.tex',escape=False)
