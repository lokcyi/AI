from BaseClass.MLModelBase import MLModelBase
import tensorflow as tf
import matplotlib.pyplot as plt
class NN(MLModelBase):
    def training(self,X,y):
            #1.建立模型(Model)
            #將Layer放入Model中
            # Activation Functions
            # A.softmax：值介於 [0,1] 之間，且機率總和等於 1，適合多分類使用。
            # B.sigmoid：值介於 [0,1] 之間，且分布兩極化，大部分不是 0，就是 1，適合二分法。
            # C.Relu (Rectified Linear Units)：忽略負值，介於 [0,∞] 之間。
            # D.tanh：與sigmoid類似，但值介於[-1,1]之間，即傳導有負值。
        model = tf.keras.Sequential([
                tf.keras.layers.Dense(units=16,input_shape=[X.shape[1]]), # units：输出维度
                tf.keras.layers.Dense(units=8, kernel_initializer='random_uniform', activation='relu'),
                tf.keras.layers.Dense(units=1, kernel_initializer='random_uniform',activation='relu')      
                ])
            #model.summary()
            #======================================================================================
            # 定義 tensorboard callback
            #tensorboard_callback = [tf.keras.callbacks.TensorBoard(log_dir='./logs2')]
            
            #========================================================================
            # SGD
            #sgd = tf.keras.optimizers.SGD(lr=0.20, momentum=0.0, decay=0.0, nesterov=False)
            # 随机梯度下降优化器。
            # 包含扩展功能的支持： - 动量（momentum）优化, - 学习率衰减（每次参数更新后） - Nestrov 动量 (NAG) 优化
            # 参数
            # lr: float >= 0. 学习率。
            # momentum: float >= 0. 参数，用于加速 SGD 在相关方向上前进，并抑制震荡。
            # decay: float >= 0. 每次参数更新后学习率衰减值。
            # nesterov: boolean. 是否使用 Nesterov 动量。
            #========================================================================
        
        model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(lr=0.01) #, optimizer=tf.keras.optimizers.SGD(lr=0.2)
                                , metrics = [ 'mae', 'mape'])   
            #需要tunning 時開啟
            #model = tuningNN(model,X_dropped,Y_dropped)                                            
            #======================================================================================
            #3.訓練 fit：以compile函數進行訓練，指定訓練的樣本資料(x, y)，並撥一部分資料作驗證，還有要訓練幾個週期、訓練資料的抽樣方式。
        train_history = model.fit(x=X, y=y,
                        validation_split=0.10, epochs=100, batch_size=100, verbose=1) #,shuffle=True validation_split=0.1, 用最後的10%資料驗證 batch_size=200: 每一批次200筆資料
            

        #dict_keys(['loss', 'accuracy', 'mse', 'mae', 'mape', 'val_loss', 'val_accuracy', 'val_mse', 'val_mae', 'val_mape'])
        
        # 當RMSE收斂至接近0.02，且MAPE接近10%，即完成模型之訓練
        figure, axis_1 = plt.subplots()
        #plt.title(df['TOOLG_ID'].iloc[0]) # title
        plt.xlabel('Epoch Number')
        plt.ylabel("Loss Magnitude")

        loss = axis_1.plot(train_history.history['loss'], label = 'loss')

        axis_2 = axis_1.twinx()
        # mse = axis_2.plot(train_history.history['mse'], label = 'mse',color='red' ) 
        mse = axis_2.plot(train_history.history['mape'], label = 'mape',color='red' )      
        # mape = axis_2.plot(train_history.history['mape'], label = 'mape' )# 準確度 接近10%
                
        axis_1.legend(loc='upper left',fontsize='large')
        axis_2.legend(loc='upper right',fontsize='large')
        #plt.show()   
        plt.savefig("./report/NN_Model.svg")

        return model