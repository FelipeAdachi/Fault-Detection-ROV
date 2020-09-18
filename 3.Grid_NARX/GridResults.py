#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 13:12:37 2019

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


path='/home/felipeadachi/Dados/serverlab/Models_GridSearch/GYROZ/'
plt_index=['ALL','CFS','RF','SW']
subfolders=['ALL_MUTINFO','CFS_MUTINFO','RF_MUTINFO','SW_MUTINFO']

path_out='/home/felipeadachi/Dados/Scripts/Final_models'
df=pd.DataFrame(index=subfolders)
df_mae=pd.DataFrame(index=subfolders)

df_std=pd.DataFrame(index=subfolders)
df_mae_std=pd.DataFrame(index=subfolders)


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
print(df.min(axis=1))

df = df[df.columns.drop(list(df.filter(regex='.3')))]
df_std = df_std[df_std.columns.drop(list(df_std.filter(regex='.3')))]
    
with pd.option_context('display.max_columns',None):
    print(df)

#js_mse=df.to_dict('index',into=OrderedDict)
#write_json_content(path + '/MSE.json',js_mse)

#js_mae=df_mae.to_dict('index',into=OrderedDict)
#write_json_content(path + '/MAE.json',js_mae)


minindex=(df.min(axis=1)).idxmin()
#print(minindex)
minvalue=(df.min(axis=1)).min()
index=[]
for i in df:
    #print("iiiii:",i)
    index.append(i)
    if df[i].min()==minvalue:
        mincolumn=i


print("Best>",minindex,",",mincolumn)
print("MSE of>",minvalue)
best=dict()
    
for j in range(len(subfolders)):

    series=df.iloc[j]
    print(subfolders[j],series)
    print("min>>>",series.min())
    best[subfolders[j]]={}
                    
    for i in range(len(series)):
        #if df.iloc[0][i]==df.iloc[0].min():
        #    print("min index=",i)
        if series[i]==series.min():
            print(series.index[i])
            print(series[i])
            best[subfolders[j]]['combination']=series.index[i]
            best[subfolders[j]]['mse']=series[i]
            with open(path_out+'/best_params.json','w+') as f:
                #print(subfolders[j],file=f)
                #print(series.index[i],series[i],file=f)
                #print("\n",file=f)
                json.dump(best,f)
    
df = df.apply(pd.to_numeric, errors='coerce')

discret= [
        # Let first 10% (0.1) of the values have color rgb(0, 0, 0)
        [0, 'rgb(0, 0, 0)'],
        [0.1, 'rgb(0, 0, 0)'],

        # Let values between 10-20% of the min and max of z
        # have color rgb(20, 20, 20)
        [0.1, 'rgb(20, 20, 20)'],
        [0.2, 'rgb(20, 20, 20)'],

        # Values between 20-30% of the min and max of z
        # have color rgb(40, 40, 40)
        [0.2, 'rgb(40, 40, 40)'],
        [0.3, 'rgb(40, 40, 40)'],

        [0.3, 'rgb(60, 60, 60)'],
        [0.4, 'rgb(60, 60, 60)'],

        [0.4, 'rgb(80, 80, 80)'],
        [0.5, 'rgb(80, 80, 80)'],

        [0.5, 'rgb(100, 100, 100)'],
        [0.6, 'rgb(100, 100, 100)'],

        [0.6, 'rgb(120, 120, 120)'],
        [0.7, 'rgb(120, 120, 120)'],

        [0.7, 'rgb(140, 140, 140)'],
        [0.8, 'rgb(140, 140, 140)'],

        [0.8, 'rgb(160, 160, 160)'],
        [0.9, 'rgb(160, 160, 160)'],

        [0.9, 'rgb(180, 180, 180)'],
        [1.0, 'rgb(180, 180, 180)']
        ]

log_alt = [
        [0, 'rgb(116,173,209)'],        #0
        [1./10000, 'rgb(171,217,233)'], #10
        [1./1000, 'rgb(224,243,248)'],  #100
        [1./100, 'rgb(254,224,144)'],   #1000
        [1./10, 'rgb(253,174,97)'],       #10000
        [1., 'rgb(244,109,67)'],             #100000

    ]


log = [
        [0, 'rgb(211, 235, 244)'],        #0
        [1./10000, 'rgb(224,243,248)'], #10
        [1./1000, 'rgb(254,224,144)'],  #100
        [1./100, 'rgb(253,174,97)'],   #1000
        [1./10, 'rgb(244,109,67)'],       #10000
        [1., 'rgb(215,48,39)'],             #100000

    ]

cbar= {
        'tick0': 0,
        'tickmode': 'array',
        'tickvals': [0, 1000, 10000, 100000]
    }

    
x=index
y=plt_index

z = df.values
print(z.shape)
std=df_std.values
print(z)
print("stdddd>",std)
std=np.around((10000*std).astype(np.double),1)
z=np.around(10000*z,1)
#print("z>",z)
annotations = go.Annotations()
for n, row in enumerate(z):
    for m, val in enumerate(row):
        annotations.append(go.Annotation(text=str(z[n][m])+'\n'+chr(177)+str(std[n][m]), x=x[m], y=y[n],
                                         xref='x1', yref='y1', showarrow=False))

data = [go.Heatmap(
  z = z,
  y = y,
  x = x,
  colorbar = cbar,
  showscale = False,
  colorscale = log
)]

layout = go.Layout(
  title = '',
  annotations=annotations,
  width=1150,
  height=400,
  xaxis = dict(
    tickmode = 'linear'
  )
)

fig = go.Figure(data=data, layout=layout)

py.plot(fig, filename='heatmap-without-padding')
