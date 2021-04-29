from BaseClass.MLModelBase import MLModelBase
import tensorflow as tf
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
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
                tf.keras.layers.Dense(units=16, kernel_initializer='random_uniform', activation='relu'),
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
        #model = self.tuningNN(model,X,y)
        # model = self.tuningNN2(model,X,y)
            #======================================================================================
            #3.訓練 fit：以compile函數進行訓練，指定訓練的樣本資料(x, y)，並撥一部分資料作驗證，還有要訓練幾個週期、訓練資料的抽樣方式。
        train_history = model.fit(x=X, y=y,
                        validation_split=0.01, epochs=160, batch_size=32, verbose=1) #,shuffle=True validation_split=0.1, 用最後的10%資料驗證 batch_size=200: 每一批次200筆資料


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
    def tuningNN(self,load_model,x,y):
        #tf.keras.wrappers.scikit_learn.KerasClassifier
        from sklearn.model_selection import GridSearchCV
        from sklearn.model_selection import RandomizedSearchCV
        from sklearn.metrics import make_scorer
        from sklearn.metrics import accuracy_score, precision_score, recall_score,r2_score
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
        # randcv = RandomizedSearchCV(estimator=MyNN(lr=0.005,nfirst=10,nhidden1=10,nhidden2=0,dropout=0.2,output_bias=0.1,batch_size=100,epochs=1),\
        #                             param_distributions=dict( epochs=[ 50,100,200], batch_size=[ 10,100],nhidden1=[2,5,10],nfirst=[10,20],dropout=[0.2],output_bias=[0.1,0.9],scale_pos_weight=[1,10]),\
        #                             n_iter=30, scoring='f1', n_jobs=1, cv=cv, verbose=1).fit(dftrain[xs], dftrain['y'])

        # pd.DataFrame(randcv.cv_results_).sort_values(by='mean_test_score',ascending=False)
        build_model = lambda: load_model
        Kmodel = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=build_model, verbose=1)
        scorers = {
            # 'precision_score': make_scorer(precision_score),
            # 'recall_score': make_scorer(recall_score),
            # 'accuracy_score': make_scorer(accuracy_score)
            'r2_score':make_scorer(r2_score)
            }
            #layers=[[8,16,20],  [45, 30, 15]],


        distributions = dict(batch_size = [ 16,32,64,75], epochs = [50, 75,100] #  , optimizer=['rmsprop', 'adam']#hidden_layers=[[64], [32]]
        )
        #     batch_size = [10, 20, 40, 60, 80, 100]
    #     epochs = [10, 50, 100]
        # activations = ['relu'],
        # param_grid = dict(layers=layers, activation=activations, batch_size = [128, 256], epochs=[30])
        # grid = GridSearchCV(estimator=Kmodel, param_grid=param_grid, cv=5)


        clf = RandomizedSearchCV(Kmodel, distributions, scoring=scorers,random_state=0,n_iter = 5, cv = 2, verbose=0,refit='r2_score')

        from joblib import Parallel, delayed, parallel_backend
        with parallel_backend('threading',n_jobs=12):
            search = clf.fit(x, y)

        print(f"最佳準確率: {clf.best_score_}，最佳參數組合：{clf.best_params_}")

        return clf.best_estimator_.model

    # def tuningNN2(self, load_model, x, y):
    #     # build_model = lambda: load_model
    #     keras_pipeline = Pipeline([("scaler", StandardScaler()),
    #                            ("clf", keras.wrappers.scikit_learn.KerasClassifier(
    #                                build_fn=make_model))
    #     ])

    #     param_grid = {'clf__network_layers': [(32, 32), (64, 64), (128, 128, 128)],
    #     'clf__batch_size': [64, 128, 256],
    #     'clf__epochs': [5, 10, 15],
    #     'clf__dropout_rate': [0.1],
    #     # 'clf__optimizer': ['Nadam'],
    #     # 'clf__activation': ['selu'],
    #     # 'clf__k_initializer': ['lecun_normal'],
    #     'clf__verbose': [0]
    #     }

    #     rs_keras = RandomizedSearchCV(keras_pipeline,
    #                                 param_distributions=param_grid,
    #                                 cv=5, refit=True,
    #                                 verbose=0,
    #                                 n_iter=50,
    #                                 scoring="accuracy")

    #     rs_keras.fit(np.array(x),
    #                 np.array(y))

    #     print('Best score obtained: {0}'.format(rs_keras.best_score_))
    #     print('Parameters:')
    #     for param, value in rs_keras.best_params_.items():
    #         print('\t{}: {}'.format(param, value))

    # def make_model(network_layers=[(32, 32)],
    #             dropout_rate=0,
    #             optimizer=tf.keras.optimizers.Adam,
    #             activation='relu',
    #             k_initializer='random_uniform',#'lecun_normal',
    #             n_input=92,
    #             n_class=1):

    #     model = keras.models.Sequential()

    #     for index, layers in enumerate(network_layers):
    #         if not index:
    #             model.add(keras.layers.Dense(layers,
    #                                         input_dim=n_input,
    #                                         activation=activation,
    #                                         kernel_initializer=k_initializer))
    #         else:
    #             model.add(keras.layers.Dense(layers,
    #                                         kernel_initializer=k_initializer,
    #                                         activation=activation))
    #         if dropout_rate and index:
    #             model.add(keras.layers.AlphaDropout(dropout_rate))

    #     model.add(keras.layers.Dense(n_class,
    #                             activation="relu"))
    #                                # activation="sigmoid"))

    #     model.compile(loss='mean_absolute_error',#loss='binary_crossentropy',
    #                 optimizer=optimizer,
    #                 metrics=['accuracy'])
    #     return model