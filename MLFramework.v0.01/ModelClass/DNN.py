from BaseClass.MLModelBase import MLModelBase
import tensorflow as tf

class DNN(MLModelBase):
    def training(self,X,y):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=32,activation = 'relu',input_shape=[X.shape[1]]),
            tf.keras.layers.Dense(units=16,activation = 'relu'),
            tf.keras.layers.Dense(units=4,activation = 'relu'), 
            tf.keras.layers.Dense(units=1)
            ]) 
        model.compile(loss='mean_squared_error', 
            optimizer=tf.keras.optimizers.Adam(0.001),
            metrics=[tf.keras.metrics.MeanAbsoluteError()]) 

        history = model.fit(X, y, epochs=500, batch_size=16, verbose=True,
            validation_split=0.01)   
        return model