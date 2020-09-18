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

which_fs={
        'rf':'RF_MUTINFO',
        'sw':'SW_MUTINFO',
        'cfs':'CFS_MUTINFO',
        'all':'ALL_MUTINFO'
        }


fault_types={
        'BIAS':{'config':{
                'GYROZ':['scenario_constantbias_GZ.json','sc_bias_t100_GZ.json','sc_bias_t10_GZ.json'],
                'GYROX':'scenario_constantbias_GX.json',
                'GYROY':'scenario_constantbias_GY.json'},
                'coeff':'constant',
                'name':'constant_bias'
                },
        'GAIN':{'config':{
                'GYROZ':['scenario_constantgain_GZ.json','sc_gain_t100_GZ.json','sc_gain_t10_GZ.json'],
                'GYROX':'scenario_constantgain_GX.json',
                'GYROY':'scenario_constantgain_GY.json'},
                'coeff':'coefficient',
                'name':'constant_gain'
                },
        'DRIFT':{'config':{
                'GYROZ':['scenario_drift_GZ.json','sc_drift_t100_GZ.json','sc_drift_t10_GZ.json'],
                'GYROX':'scenario_drift_GX.json',
                'GYROY':'scenario_drift_GY.json'},
                'coeff':'slope',
                'name':'drift'
                },
        'STUCKAT':{'config':{
                'GYROZ':['scenario_stuckat_GZ.json','sc_stuck_t100_GZ.json','sc_stuck_t10_GZ.json'],
                'GYROX':'scenario_stuckat_GX.json',
                'GYROY':'scenario_stuckat_GY.json'},
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




#table.index=subsets
#print(table)

base_save_path='/home/felipeadachi/Dados/serverlab/Detection_Results-23.07.19'
def main():
    cols=['positives','negatives','true_positives','false_positives']
    index=[10,25,50,100]
    df=pd.DataFrame(index=index,columns=cols)
    df=df.fillna(0)
    print("df pre>",df)    
    for fault_type in fault_types:
        j=0
        for path in paths:
            total_save_path=path+'/'+target+'/'+which_fs[fs]+'/'+fault_type
            print(total_save_path)
            
            with open(total_save_path+'/'+fault_types[fault_type]['config'][target][j], 'r') as config_file:
                    scenarios = json.load(config_file)
            
            for i in range(1,19):
                if i in paths[path]:
                    for element in scenarios['scenarios']:
                        print("Element>",element)
                        if element['scenario_number']==i:
                            #print(i)
                            duration=element['duration_fault']
                            duration=duration[0]
                            print("duration>",duration)
                            fault_info=element[fault_types[fault_type]['name']]
                            coeff=fault_info[fault_types[fault_type]['coeff']]
                            coeff=coeff[0]
                            for col in cols:
                                filename=str(i)+'.'+str(col)+'.json'
                                js=read_json_content(total_save_path+'/'+filename)
                                
                                if col=='positives':
                                    #print("inif>",type(js))
                                    df.loc[duration,col]=df.loc[duration,col]+int(js)
                                else:
                                    #print("coeff>",coeff)
                                    key = 'mres_%d_thr_%.2f' % (sliding_window, threshold)
                                    df.loc[duration,col]=df.loc[duration,col]+js[key]
            j+=1        
    df['recall']=df['true_positives']/df['positives']
    df['precision']=df['true_positives']/(df['true_positives']+df['false_positives'])
    df['f_score']=2*(df['precision']*df['recall'])/(df['precision']+df['recall'])
    print("df pos>",df)
    df.to_csv("byduration.csv")
    ax=df['f_score'].plot.bar()
    #ax.set_xlabel("fault duration")
    #ax.set_ylabel("f-score")
    ax.tick_params(labelsize='large')
    plt.xticks(rotation='horizontal')
    plt.show()
        #df.plot(style='.--',xticks=[1,5,25,50],xlim=(0,53))
        
        #plt.show()
#---------------------#---------------------#---------------------#---------------------#---------------------#    


def read_json_content(filename):
    with open(filename, 'r') as f:
        return json.JSONDecoder(object_pairs_hook=collections.OrderedDict).decode(f.read(os.stat(filename).st_size))

#---------------------#---------------------#---------------------#---------------------#---------------------#

if (__name__=='__main__'):
    main();           
