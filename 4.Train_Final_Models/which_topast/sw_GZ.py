model = Sequential()
    model.add(Dense(28, input_dim=28, kernel_initializer='uniform', activation='tanh'))
    model.add(Dense(15, kernel_initializer='uniform', activation='tanh'))
    model.add(Dense(1))    

    # Compile model
    #model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
    opt=SGD(lr=0.2, momentum=0.9)

['mtarg1_t2', 'mtarg3_t2', 'roll_t1', 'pitch_t1', 'LACCX_t1', 'LACCY_t1', 'LACCZ_t1', 'GYROX_t2', 'GYROY_t1', 'SC1I_t2', 'SC2I_t2', 'BT1I_t2', 'vout_t1', 'iout_t1', 'cpuUsage_t1']

which_topast = [
    ["mtarg1",1],
    ["mtarg3",1],
        

    ["mtarg1",2],
    ["mtarg3",2],

    ["roll",1],

    ["pitch",1],
    #["deapth",0],
    ["LACCX",1],
    
    ["LACCY",1],
    
    ["LACCZ",1],
    
    ["GYROX",1],
    ["GYROX",2],
    
    ["GYROY",1],
    
    ["SC1I",1],
    ["SC1I",2],
    
    ["SC2I",1],
    ["SC2I",2],
    
    ["BT1I",1],
    ["BT1I",2],
        
    
    ["vout",1],
    
    ["iout",1],
    
    ["cpuUsage",1],
    
    ["GYROZ",1],
    ["GYROZ",2],
    ["GYROZ",3],
    ["GYROZ",4],
    ["GYROZ",5],
    ["GYROZ",6],
    ["GYROZ",7]
    
    #["temp",0]
    
    ]
