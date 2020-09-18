import json
import os
import collections
import pandas as pd
import numpy as np
from sklearn import preprocessing
import pickle
from keras.models import model_from_json
from keras.optimizers import SGD
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
import fault_inserter
import itertools
import statistics
from feature_subsets import which_topast


#original='1.s1.v2.08:36.json'
#faulty='1.s1.v2.08:36.json'
original='22.s2.v2.controle.dpfreq.19:46.json'
#original='5.s4.v2.controle.dpfreq.09:47.json'
#original='1.s4.v3.controle.11:25.json'
#original='3.s2.v3.choque.09.43.json'
#original='14.s1.v3.controle.dpfreq.15:11.json'
#original='0.s4.v3.controle.dpfreq.09:05.json'
#original='9.s3.v3.22:28.json'
#original='17.s2.5hz.v3.br.15:37.json'
#faulty='22.s2.v2.controle.dpfreq.19:46.json'
#
#faulty='22.s2.v2.controle.dpfreq.19:46.faulty_changed.json'

which_fs={
        'rf':'RF_MUTINFO',
        'sw':'SW_MUTINFO',
        'cfs':'CFS_MUTINFO',
        'all':'ALL_MUTINFO'
        }

path='/home/felipeadachi/Dados/serverlab/Whole_train/Final_models_teste'
target='GYROZ'
#subsets=['all','rf','sw','cfs']
#configs_file=['scenario_stuckat.json']
fs='rf'
total_path=path+"/"+target+"/"+which_fs[fs]
print(total_path)
thresholds=[0.06,0.08,0.09]
sliding_window=[7,15]

autoregressive={
        'GYROX':8,
        'GYROY':7,
        'GYROZ':7
        
        }


def main():

    config_file='sc_gain.json'
    
    with open(config_file, 'r') as config_file:
        scenarios = json.load(config_file)
    
    source_series=read_json_content(original)
    source_series=source_series['0']
    source_series=source_series['data']
    source_series=has_elapsed(source_series)
    for scenario in scenarios['scenarios']:
        #print(scenario)
        
        altered_series=fault_inserter.inject_fault(source_series, scenario)
        df_results=pd.DataFrame()
        results_cols=['fault','fault_type','fault_value','fault_duration','timestamp']
        
        
        df_alt=pd.DataFrame(altered_series)
        
        for col in results_cols:
            df_results[col]=df_alt[col].values
        
        
        
        for col in results_cols:
            df_alt=df_alt.drop(columns=col)
        
        
        #with pd.option_context('display.max_rows',None):
        #    print("df results>",df_results)
        #pyplot.show()


        df_train=pd.DataFrame()
        df_val=pd.DataFrame()
        
        aux=[]
    
        #js1=read_json_content(altered_series)
        #js2=read_json_content(source_series)
        
        #js_train=js1['0']['data']
        #js_val=js2['0']['data']
        
        js_train=altered_series
        js_val=source_series
            #---------Gera os atributos passados de acordo com which_topast    
        
        for att in which_topast[fs]:
            js_train=past_commands(js_train,att)
            js_val=past_commands(js_val,att)
            att2=att[0]+'_t'+str(att[1])
            aux.append(att2)
        
        which_tokeep=[target,'timestamp']
        which_tokeep=which_tokeep+aux
        #print(which_tokeep)
    
        #---------Gera os dataframes a partir dos json    
        
        df_train=df_train.append(pd.DataFrame(js_train))
        df_val=df_val.append(pd.DataFrame(js_val))
        
        #---------Mantem no dataframe apenas os atributos alvo e independentes
        
        for col in df_train.columns:
                if col not in which_tokeep:
                    df_train=df_train.drop(columns=col)
        for col in df_val.columns:
            if col not in which_tokeep:
                df_val=df_val.drop(columns=col)
        #print("df_train_pre>",df_train.head)
        #with pd.option_context('display.max_columns',None):
        #    print(df_train.head())
        
        df_train = df_train.apply(pd.to_numeric, errors='coerce')
        df_train=df_train.dropna()
        df_val = df_val.apply(pd.to_numeric, errors='coerce')
        df_val=df_val.dropna()
        
        
        
        #print("df_train_pos>",df_train.head)
        
        df_timestamp=pd.DataFrame()
        df_timestamp=df_train['timestamp']
        df_timestamp = df_timestamp.to_frame()
        df_train=df_train.drop(columns='timestamp')
        df_val=df_val.drop(columns='timestamp')
        
        #print(len(df_train))
        #print(len(df_val))
        
        #---------gera numpy para x e y, para train e validation
        
        
        y_train=df_train[target].values
        y_val=df_val[target].values
        
        #print(y.shape)
        #print("y train>",y_train)
        
        y_train=y_train.reshape(-1,1)
        y_val=y_val.reshape(-1,1)
        with open(total_path+'/'+fs+'_y.pickle', 'rb') as f:
            scaler_y_train=pickle.load(f)
            
        y_train=scaler_y_train.transform(y_train)       
        
        #scaler_y_train = preprocessing.MinMaxScaler()
        scaler_y_val = preprocessing.MinMaxScaler()
        
        #y_train = scaler_y_train.fit_transform(y_train)
        y_val = scaler_y_val.fit_transform(y_val)
        #scaler_y=scaler_y_train.fit(y_train)
        #file = open('scaler_final_GYROZ_y', 'wb')
        #pickle.dump(scaler_y,file)
        #file.close()
    
        y_train=y_train.ravel()
        y_val=y_val.ravel()
        #print(y_train)
        #print(y_val)
        
        
        x_train=df_train.drop(columns='GYROZ').values
        x_val=df_val.drop(columns='GYROZ').values
        #print("x train>",x_train)
        #print("y_train norm>",y_train)
        
        with open(total_path+'/'+fs+'_x.pickle', 'rb') as f:
            scaler_x_train=pickle.load(f)
    
        x_train=scaler_x_train.transform(x_train)       
        
        
        scaler_x_val = preprocessing.MinMaxScaler()
        #scaler_x=scaler_x_train.fit(x_train)
        x_val = scaler_x_val.fit_transform(x_val)
        
        #print("x_train norm>",x_train)
        
        
        with open(total_path+'/'+fs+'.json') as f:
            val=f.read()
            model = model_from_json(val)
            model.load_weights(total_path+'/'+fs+'.h5')
        
        opt=SGD(lr=0.2, momentum=0.9)
        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_squared_error'])
        #print(x_train.shape)    
        py = model.predict(x_train)
        #print(py.shape)
        
        predicted_prefault=py
        original_prefault=y_train
        
        predicted_postfault=py
        original_postfault=y_train
        
        
        #print("shape",predicted_postfault.shape)
        
        mean_squared_error(predicted_postfault,original_postfault)
        
        print("MSE pré falha:",mean_squared_error(predicted_prefault,original_prefault))
        print("MSE pós falha:",mean_squared_error(predicted_postfault,original_postfault))
        
        df_1_2 = df_results.merge(df_timestamp, on="timestamp", how="left", indicator=True)

        df_results = df_1_2[df_1_2["_merge"] == "both"].drop(columns=["_merge"])
        print("len py",len(py))
        print("len y_train",len(y_train))
        print("len df_results",len(df_results))
        
        for index,row in df_results.iterrows():
            if row['fault']==0:
                print(row['timestamp'])
        df_results=df_results.reset_index()
        df_results=df_results.drop(columns='index')
        #pyplot.plot(py[1000:], 'blue')
        #pyplot.plot(y_train[1000:], 'orange')
        pyplot.plot(py,linestyle='dashed',color='orange')
        pyplot.plot(y_train,'blue')
        pyplot.plot(df_results['fault'],'ro')
        pyplot.show()
        
        #with open('predicted.pickle', 'wb') as f:
        #    pickle.dump(py, f)       
        
        #with open('original.pickle', 'wb') as f:
        #    pickle.dump(y_train, f)       
        
        py=py.ravel()
        predicted=py
        actual=y_train
        
        #---Create moving_residuals features for each sliding window
        df_results=generate_residuals(df_results,predicted,actual)
        
        generate_statistics(df_results,predicted,actual,saveplot=False)
        #df_results.to_csv('csv_teste.csv')
        #df_results=check_threshold(df_results)
                
        #print("min",min_latency)
        #print("max",max_latency)
                    
        #print("positives>",positives)
        #df_results=pd.concat([df_results,df_moving], ignore_index=True, axis=0)
        #df_results['moving_resid']=moving_average(resid,10)
        #print("df results post moving>",df_results.tail())
        #print(moving_average(resid,10))
        #df_results['moving_resid']=moving_average(resid,10)
        
        pyplot.plot(df_results['fault'],'ro')
        pyplot.plot(df_results['moving_resid_7'])
        
        pyplot.show()
        
#---------------------#---------------------#---------------------#---------------------#---------------------#    


def read_json_content(filename):
    with open(filename, 'r') as f:
        return json.JSONDecoder(object_pairs_hook=collections.OrderedDict).decode(f.read(os.stat(filename).st_size))

#---------------------#---------------------#---------------------#---------------------#---------------------#

def generate_residuals(df_results,predicted,actual):
    resid=np.subtract(predicted,actual)
    resid=resid**2
    resid=np.sqrt(resid)
    df_results['resid']=resid
    for window in sliding_window:
        col_name='moving_resid_'+str(window)
        mov=moving_average(resid,window)
        i=0
        print("window>",window)
                    
        for index, row in df_results.iterrows():
            #print("index>",index)
            if int(index) + len(mov)>= len(df_results):
                #print("inside if,index>",i)
                col_name='moving_resid_'+str(window)
                current_mov=mov[index-window+1]
                df_results.loc[index,col_name]=current_mov
                for thresh in thresholds:
                    col_name='mres_'+str(window)+"_thr_"+str(thresh)
                    if current_mov<thresh:
                        df_results.loc[index,col_name]=0
                    else:
                        df_results.loc[index,col_name]=1
                
            i+=1
    return df_results


#---------------------#---------------------#---------------------#---------------------#---------------------#
def generate_statistics(df_results,predicted,actual,saveplot=False):
    
    nfault=0
    positives = 0
    negatives={}
    true_positives = {}
    false_positives={}
    false_negatives={}
    latencies = {}
    latencies_mean = {}
    latencies_std = {}
    precision={}
    recall={}
    f_score={}

    max_latency=max(sliding_window)+autoregressive[target]

    last_time_was_error = False
    
    for threshold in thresholds:
        for window_size in sliding_window:
            column = 'mres_%d_thr_%.2f' % (window_size, threshold)
            negatives[column]=df_results[column].count()
            true_positives[column] = 0
            latencies[column] = []
            value_count=df_results[column].value_counts()
            value_count=value_count.to_frame()
            try:
                false_positives[column]=value_count.loc[1.0,column]
            except KeyError:
                false_positives[column]=0
            for l in [list(group) for key, group in itertools.groupby(df_results[column].values.tolist())]:
                if l[0] == 1:
                    print(l[0], len(l))
            print('#---#')
    
    for index,row in df_results.iterrows():
        
        if last_time_was_error is True:
            last_time_was_error = ( row['fault'] == 0 )
        elif row['fault'] == 0:
            nfault+=1
            #print("index>",index)
            error_end_index = df_results[index:]['fault'].ne(0).idxmax()
            error_duration = error_end_index - index
            positives+=1
            print(index, error_end_index, error_duration)
    
    #----This excerpt plots the fault ocurrences, along with the classification results of different windows, but with a specified threshold.
    # =============================================================================
            if saveplot:
                
                pyplot.plot(predicted[index-20:index+max_latency+error_duration+20],'--')
                pyplot.plot(actual[index-20:index+max_latency+error_duration+20],'black')
                pyplot.plot((df_results.loc[index-20:index+max_latency+error_duration+20,'fault']).values-0.001,'rs',label='fault')
                offset=0.035
                for window in sliding_window:
                    column = 'mres_%d_thr_0.04' % (window)
                    pyplot.plot((df_results.loc[index-20:index+max_latency+error_duration+20,column]).values-offset-1,'s',label='window='+str(window))
                    offset+=0.035
                pyplot.ylim(bottom=-0.15)
                pyplot.legend(loc='lower right')
                pyplot.savefig('fault_'+str(nfault)+'.png')
                pyplot.clf()
                #pyplot.show()
    # =============================================================================
            
            for threshold in thresholds:
    
                for window_size in sliding_window:
                    latency_tolerance=window_size+int(autoregressive[target])
                    #print(df_results.loc[index-10: index+15])
                    column = 'mres_%d_thr_%.2f' % (window_size, threshold)
                    negatives[column]=negatives[column]-(error_duration+window_size)
                    slice_ = df_results[column][index: index + latency_tolerance + error_duration]
                    number_positives=slice_.value_counts()
                    number_positives=number_positives.to_frame()
                    #print("npos>",number_positives)
                    try:
                        npos=number_positives.loc[1.0,column]
                        false_positives[column]=false_positives[column]-npos
                    except KeyError:
                        pass
                    #print("slice,",slice_)
                    #print(first)
                    error_detected = 1. in slice_.values
                    if error_detected is True:
                        true_positives[column] = true_positives[column] + 1
                        first = slice_.tolist().index(1.)
                        
                        latencies[column].append(first)
                    last_time_was_error = True
    
    
    for key in true_positives:
        try:
            recall[key]=true_positives[key]/positives
        except ZeroDivisionError:
            recall[key]=np.nan
        try:
            precision[key]=true_positives[key]/(true_positives[key]+false_positives[key])
        except ZeroDivisionError:
            precision[key]=np.nan
        try:
            f_score[key]=2*(precision[key]*recall[key])/(precision[key]+recall[key])
        except ZeroDivisionError:
            f_score[key]=np.nan
        false_negatives[key]=positives-true_positives[key]
        
        print(key)
    
    for key in latencies:
        print("key latencies>",latencies[key])
        try:
            latencies_mean[key]=statistics.mean(latencies[key])
            #print("mean latencies>",statistics.mean(latencies[key]))
            
        except statistics.StatisticsError:
            latencies_mean[key]=np.nan
        try:
            latencies_std[key]=statistics.pstdev(latencies[key])
            #print("std latencies>",latencies_std[key])
        except statistics.StatisticsError:
            latencies_std[key]=np.nan
    print("Positives>",positives)
    print("True Positives>",true_positives)
    print("False Negatives>",false_negatives)
    print("False Positives>",false_positives)
    print("Latencies>",latencies)
    print("Latencies mean>",latencies_mean)
    print("Latencies std>",latencies_std)
    print("Precision",precision)
    print("Recall",recall)
    print("f_score",f_score)
    print("negatives",negatives)    
#---------------------#---------------------#---------------------#---------------------#---------------------#

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

#---------------------#---------------------#---------------------#---------------------#---------------------#

def has_elapsed(content):
    
    data_list = content
    
    i=0
    
    while (i<len(data_list)):
        
        data=data_list[i]
        if i==0:
            data['has_elapsed']=1
        if i >= 1:
            if (data['timestamp']-data_list[i-1]['timestamp'])>=500:
                data['has_elapsed']=1
            else:
                data['has_elapsed']=0
        i+=1
    return content

#---------------------#---------------------#---------------------#---------------------#---------------------#

def past_commands(content,att):
    
    data_list=content
    i=0
    
        
    while (i<len(data_list)):
        
        #get_token retorna o número de vezes permitido para pegar os comandos passados, com base nos flags "has_elapsed" de cada timestep
        token=get_token(i,data_list,att)
        #print("token for timestep:",i,"is:",token)
        data=data_list[i]
        tkn={}
        
        tkn[att[0]]=token
        
        
        if i<att[1]:
            data[att[0]+'_t'+str(att[1])]=np.nan
        else:
            try:
                if token>=att[1]:
                    data[att[0]+'_t'+str(att[1])]=data_list[i-att[1]][att[0]]
                else:
                    data[att[0]+'_t'+str(att[1])]=np.nan
            except KeyError:
                data[att[0]+'_t'+str(att[1])]=np.nan                    
        i+=1
    return content
#---------------------#---------------------#---------------------#---------------------#---------------------#
def get_token(i,data_list,att):
    j=0
    token=0
    while j<att[1]:
        if data_list[i-j]['has_elapsed'] == 0:
            token+=1
        else:
            return token
        
        j+=1
    return token



if (__name__=='__main__'):
    main();