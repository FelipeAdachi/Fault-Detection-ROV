model = Sequential()
    model.add(Dense(34, input_dim=34, kernel_initializer='uniform', activation='tanh'))
    model.add(Dense(50, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(1))    

    # Compile model
    #model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
    opt=SGD(lr=0.01, momentum=0.9)

which_topast = [
    ["mtarg1",1],
    ["mtarg3",1],
    
    ["mtarg2",1],
    ["mtarg2",2],
    

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
    
    ["SC3I",1],
    ["SC3I",2],
    
    
    ["BT1I",1],
    ["BT1I",2],
    
    ["BT2I",1],
    ["BT2I",2],
    
    
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
