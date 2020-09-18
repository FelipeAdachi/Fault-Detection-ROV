#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 15:45:23 2019

@author: felipeadachi
"""
import numpy as np
import pandas as pd
from sklearn import preprocessing
from statsmodels.graphics.tsaplots import plot_pacf
from matplotlib import pyplot as plt
import json
import os
import collections

def main():

    for k in range(1,6):

        filename='/home/felipeadachi/Dados/Kfolds_cpu/Fold_'+str(k)+'/Train_'+str(k)
        path_out='/home/felipeadachi/Dados/Kfolds_cpu/Fold_'+str(k)+'/pacf_analysis/GYROX'
        
        df=pd.DataFrame()
        aux=[]
        

        js=read_json_content(filename)
        print("js>",js[1])
        which_tokeep=['GYROX']
        #print(which_tokeep)
        
        df=df.append(pd.DataFrame(js))


        for col in df.columns:
                if col not in which_tokeep:
                    df=df.drop(columns=col)
        
        df=df.dropna()
        print("df>",df)
        df = df.apply(pd.to_numeric, errors='coerce')
        
        #df['current_t1']=df['SC1I_t1']+df['SC3I_t1']+df['BT1I_t1']+df['BT2I_t1']
        
        
        
        #df=df.drop(columns='SC1I_t1')
        #df=df.drop(columns='SC3I_t1')
        #df=df.drop(columns='BT1I_t1')
        #df=df.drop(columns='BT2I_t1')
        
        y=df['GYROX'].values
        print(y)
        #print(y.shape)
        y=y.reshape(-1,1)
        
        min_max_scaler_y = preprocessing.MinMaxScaler()
        y = min_max_scaler_y.fit_transform(y)
        y=y.ravel()
        #print("x",len(x))
        #print("xshape>",x.shape)    
        #print("leny",y.shape)
        #print(y.shape)
        #print(np.ones([96883,1]).shape)
        
        dfy = pd.DataFrame(y)
        dfy.columns=['GYROX']
        #df3=df3.assign(timestamp=df['timestamp'])
            
        #print("dft>",dft)
        
        series=dfy['GYROX']
        
        
        #define function for ADF test
        
        #apply adf test on the series
        #adf_test(series)
        #print(series)
        acorr=[]
        #acorr=pacf(series,nlags=50)
        #print(acorr)
        #print("conf",conf)
        plot_pacf(series,lags=50)
        plotfile=path_out+"/PACF_F"+str(k)+".png"
        plt.savefig(plotfile)
        plt.show()
        #series.plot()
        #pyplot.show()
    
    
def adf_test(timeseries):
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)

    
    
    
#---------------------#---------------------#---------------------#---------------------#---------------------#

def read_json_content(filename):
    with open(filename, 'r') as f:
        return json.JSONDecoder(object_pairs_hook=collections.OrderedDict).decode(f.read(os.stat(filename).st_size))

#---------------------#---------------------#---------------------#---------------------#---------------------#









if (__name__=='__main__'):
    main();