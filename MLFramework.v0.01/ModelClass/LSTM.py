from BaseClass.MLModelBase import MLModelBase
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

class CAT(MLModelBase):
    def __init__(self):
        MLModelBase.__init__(self)
    def training(self,X,y):
        '''
        Documenntation says: this LSTM implementation defaultly has
        activation="tanh",
        recurrent_activation="sigmoid",
        '''
        #Initilizing RNN
        LSTMmodel = Sequential()
        # # LSTMmodel.add(LSTM(units = 50,  input_shape = (x_train.shape[1], 1)))
        #Adding first LSTM Layer and Dropout Regularization
        LSTMmodel.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))

        LSTMmodel.add(Dropout(rate = 0.2))

        #Adding Second LSTM Layer and Dropout Regularization
        LSTMmodel.add(LSTM(units = 50, return_sequences = True))
        LSTMmodel.add(Dropout(rate = 0.2))

        #Adding Third LSTM Layer and Dropout Regularization
        LSTMmodel.add(LSTM(units = 50, return_sequences = True))
        LSTMmodel.add(Dropout(rate = 0.2))

        #Adding fourth LSTM Layer and Dropout Regularization
        LSTMmodel.add(LSTM(units = 50))
        LSTMmodel.add(Dropout(rate = 0.2))

        #Adding the output Layer
        # LSTMmodel.add(Dense(units = 1))
        LSTMmodel.add(Dense(1, activation = "tanh"))

        #Compiling the RNN
        # LSTMmodel.compile(optimizer = 'adam', loss = 'mean_squared_error'   , metrics = ['mse', 'mae', 'mape'])

        LSTMmodel.compile(optimizer = tf.keras.optimizers.Adam(0.001),  loss = 'mean_squared_error'   , metrics = ['mse', 'mae', 'mape'])
        '''
        #Fitting the RNN to the Training Set
        # early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
        '''
        history_lstm_model = LSTMmodel.fit(x, y, epochs = 50, batch_size = 1,verbose = 0 ,shuffle=True)
        return LSTMmodel