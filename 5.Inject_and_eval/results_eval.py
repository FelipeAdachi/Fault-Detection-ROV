import json
import collections
import os
import pandas as pd
which_fs={
        'rf':'RF_MUTINFO',
        'sw':'SW_MUTINFO',
        'cfs':'CFS_MUTINFO',
        'all':'ALL_MUTINFO'
        }


target='GYROZ'

subsets=['cfs','rf','sw','all']

thresholds=[0.06,0.07,0.08,0.09]

arrays=[[0.06,0.06,0.06,0.07,0.07,0.07,0.08,0.08,0.08,0.09,0.09,0.09],[7,10,13,7,10,13,7,10,13,7,10,13]]

#tuples=list(zip(*arrays))


#with pd.option_context('display.max_columns',None):
#    print(df)

sliding_window=[7,10,13]


table=pd.DataFrame()



#table.index=subsets
#print(table)

base_save_path='/home/felipeadachi/Dados/serverlab/Detection_Results'
def main():
    df = pd.DataFrame(index=subsets, columns=arrays)

    for fs in subsets:
        total_save_path=base_save_path+'/'+target+'/'+which_fs[fs]+'/ALL_RESULTS/whole_recall.json'
        #print(total_save_path)
        f_score=read_json_content(total_save_path)
        for key in f_score:
            x=key.split("_")
            #print(x[1],x[3])
            df.loc[fs,(str(x[3]),str(x[1]))]=round(f_score[key],2)
    df=df.dropna(axis=1)
    with pd.option_context('display.max_columns',None):
        print("df>",df)
                
    #df.to_csv('teste.csv')
    #df.to_latex('teste.tex')

#---------------------#---------------------#---------------------#---------------------#---------------------#    


def read_json_content(filename):
    with open(filename, 'r') as f:
        return json.JSONDecoder(object_pairs_hook=collections.OrderedDict).decode(f.read(os.stat(filename).st_size))

#---------------------#---------------------#---------------------#---------------------#---------------------#

if (__name__=='__main__'):
    main();           