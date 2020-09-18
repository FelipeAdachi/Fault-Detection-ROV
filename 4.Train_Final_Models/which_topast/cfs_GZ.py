model = Sequential()
    model.add(Dense(13, input_dim=13, kernel_initializer='uniform', activation='tanh'))
    model.add(Dense(30, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(1))    

    # Compile model
    #model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
    opt=SGD(lr=0.2, momentum=0.9)

which_topast = [
    ["mtarg1",1],
    ["mtarg3",1],
        

    ["mtarg1",2],
    ["mtarg3",2],

 
    ["LACCX",1],
    
    ["LACCY",1],
    
    ["GYROZ",1],
    ["GYROZ",2],
    ["GYROZ",3],
    ["GYROZ",4],
    ["GYROZ",5],
    ["GYROZ",6],
    ["GYROZ",7]
    
    #["temp",0]
    
    ]
