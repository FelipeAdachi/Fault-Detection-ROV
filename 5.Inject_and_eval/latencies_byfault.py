#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 17:10:59 2019

@author: felipeadachi
"""

import json
import collections
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

which_fs={
        'rf':'RF_MUTINFO',
        'sw':'SW_MUTINFO',
        'cfs':'CFS_MUTINFO',
        'all':'ALL_MUTINFO'
        }


fault_types={
        'BIAS':{'config':{
                'GYROZ':['scenario_constantbias_GZ.json','sc_bias_t100_GZ.json','sc_bias_t10_GZ.json'],
                'GYROX':['scenario_constantbias_GX.json','sc_bias_t10_100_GX.json'],
                'GYROY':['scenario_constantbias_GY.json','sc_bias_t10_100_GY.json']},
                'coeff':'constant',
                'name':'constant_bias'
                },
        'GAIN':{'config':{
                'GYROZ':['scenario_constantgain_GZ.json','sc_gain_t100_GZ.json','sc_gain_t10_GZ.json'],
                'GYROX':['scenario_constantgain_GX.json','sc_gain_t10_100_GX.json'],
                'GYROY':['scenario_constantgain_GY.json','sc_gain_t10_100_GY.json']},
                'coeff':'coefficient',
                'name':'constant_gain'
                },
        'DRIFT':{'config':{
                'GYROZ':['scenario_drift_GZ.json','sc_drift_t100_GZ.json','sc_drift_t10_GZ.json'],
                'GYROX':['scenario_drift_GX.json','sc_drift_t10_100_GX.json'],
                'GYROY':['scenario_drift_GY.json','sc_drift_t10_100_GY.json']},
                'coeff':'slope',
                'name':'drift'
                },
        'STUCKAT':{'config':{
                'GYROZ':['scenario_stuckat_GZ.json','sc_stuck_t100_GZ.json','sc_stuck_t10_GZ.json'],
                'GYROX':['scenario_stuckat_GX.json','sc_stuck_t10_100_GX.json'],
                'GYROY':['scenario_stuckat_GY.json','sc_stuck_t10_100_GY.json']},
                'coeff':'constant',
                'name':'stuck_at'
                },

            }


target='GYROZ'

fs='rf'

threshold=0.07

sliding_window=13

paths={
       '/home/felipeadachi/Dados/serverlab/Detection_Results-23.07.19':[7,8,9,10,11,12],
       '/home/felipeadachi/Dados/serverlab/Detection_Results_t100':[13,14,15],
       '/home/felipeadachi/Dados/serverlab/Detection_Results_t10':[16,17,18],
       
       }


table=pd.DataFrame()



#table.index=subsets
#print(table)

base_save_path='/home/felipeadachi/Dados/serverlab/Detection_Results-23.07.19'
def main():
    k=0
    #fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    #plt.subplots_adjust(hspace=0.3)
    #axes=[ax1,ax2,ax3,ax4]
    #titles=['(a)','(b)','(c)','(d)']
    index=[10,25,50,100]
        
    for fault_type in fault_types:
        df=pd.DataFrame(index=index)
        df_std=pd.DataFrame(index=index)
        
        j=0
        for path in paths:
            total_save_path=path+'/'+target+'/'+which_fs[fs]+'/'+fault_type
            print(total_save_path)
            
            with open(total_save_path+'/'+fault_types[fault_type]['config'][target][j], 'r') as config_file:
                    scenarios = json.load(config_file)
            
            for i in range(1,19):
                if i in paths[path]:
                    for element in scenarios['scenarios']:
                        if element['scenario_number']==i:
                            #print(i)
                            duration=element['duration_fault']
                            duration=duration[0]
                            print("duration>",duration)
                            fault_info=element[fault_types[fault_type]['name']]
                            coeff=fault_info[fault_types[fault_type]['coeff']]
                            coeff=coeff[0]
                            print("coeff>",coeff)
                            filename=str(i)+'.latencies_mean.json'
                            file_std=str(i)+'.latencies_std.json'
                            js=read_json_content(total_save_path+'/'+filename)
                            js_std=read_json_content(total_save_path+'/'+file_std)
                            
                            key = 'mres_%d_thr_%.2f' % (sliding_window, threshold)
                            print("latency>",js[key])
                            print("std",js_std[key])
                            df.loc[duration,coeff]=js[key]
                            df_std.loc[duration,coeff]=js_std[key]
            j+=1
        #df=df.fillna(0)
        print("dffff>",df)
        if fault_type == 'DRIFT':
            print("to remove>>>>",df.loc[50,0.01])
            df.loc[50,0.01]=np.nan
        #df.plot(style='.--',ylim=(-5,100),xlim=(8,103),ax=axes[k],title=titles[k])
        

        #Start Block
        styles = ['bo--','mo-.','go:']
        fig, ax = plt.subplots()
        for col, style in zip(df.columns, styles):
            if fault_type == 'GAIN' and col==1.1:
                df[col].plot(fmt=style,yerr=df_std[col],capsize=4,linewidth=3,xticks=[10,25,50,100],ylim=(-5,110),xlim=(8,103),label=str(col)+"(ND)", ax=ax)
            else:    
                df[col].plot(fmt=style,yerr=df_std[col],capsize=4,linewidth=3,xticks=[10,25,50,100],ylim=(-5,110),xlim=(8,103),label=col, ax=ax)
            plt.xticks([10, 25, 50, 100],
                       ["10\n(2s)", "25\n(5s)", "50\n(10s)", "100\n(20s)"])
         
            plt.yticks([0, 20, 40, 60, 80, 100],
                       ["0 \n(0s)", "20 \n(4s)", "40 \n(8s)","60 \n(12s)","80 \n(16s)", "100 \n(20s)"])
            plt.legend(loc='best',fontsize=20)
            plt.tick_params(labelsize=23)
            
 
        plt.show()
        #End Block        
# =============================================================================
#         df.plot(fmt="s--",capsize=4,yerr=df_std,xticks=[10,25,50,100],ylim=(-5,100),xlim=(8,103))
#         plt.style='.--'
#         plt.xticks([10, 25, 50, 100],
#            ["10\n(2s)", "25\n(5s)", "50\n(10s)", "100\n(20s)"])
#         
#         plt.yticks([0, 20, 40, 60, 80, 100],
#            ["0 ", "20 \n(4s)", "40 \n(8s)","60 \n(12s)","80 \n(16s)", "100 \n(20s)"])
# 
#         plt.show()
#         
# =============================================================================
        k+=1
        
# =============================================================================
#     axes[0].set_ylabel('f-score')
#     axes[2].set_ylabel('f-score')
#     axes[2].set_xlabel('fault duration')
#     axes[3].set_xlabel('fault duration')
#     
# =============================================================================
    
#---------------------#---------------------#---------------------#---------------------#---------------------#    


def read_json_content(filename):
    with open(filename, 'r') as f:
        return json.JSONDecoder(object_pairs_hook=collections.OrderedDict).decode(f.read(os.stat(filename).st_size))

#---------------------#---------------------#---------------------#---------------------#---------------------#

if (__name__=='__main__'):
    main();           
