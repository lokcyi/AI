import sys
import pandas as pd 
import tensorflow as tf 
import data_preparer

#train_df = pd.read_csv(r'D:\Project\MyPython\titanic\data\train_less.csv') # 400 筆
train_df = pd.read_csv(r'D:\Project\MyPython\titanic\data\train_full.csv') # 800 筆

# 挑選特徵
#cols = ['Survived','Pclass']
cols = ['Survived','Pclass','Name','Sex','Age','SibSp','Parch','Fare','Embarked']  
train_df = train_df[cols]  

train_features, train_labels = data_preparer.preprocess(train_df) #feature (= x), label (= y)





# 建立模型
print("\n[Info] 建立模型")  
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
  
model = Sequential()  
# 輸入層
#model.add(Dense(units=3, input_dim=1, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=10, input_dim=9, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(0.1))

# 隱藏層
#model.add(Dense(units=5, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=20, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=30, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=40, kernel_initializer='uniform', activation='relu', kernel_regularizer='l2'))
model.add(Dropout(0.2))

# 輸出層
model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
#print("\n[Info] Show model summary...")  
#model.summary()





# 進行訓練
print("\n[Info] 訓練中...")  
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  
#train_history = model.fit(x=train_features, y=train_labels, validation_split=0.1, epochs=10, batch_size=30, verbose=2)  
#train_history = model.fit(x=train_features, y=train_labels, validation_split=0.1, epochs=200, batch_size=30, verbose=2)  

early_stopping = tf.keras.callbacks.EarlyStopping(patience=25)
train_history = model.fit(x=train_features, y=train_labels, validation_split=0.1, epochs=200, batch_size=30, verbose=2, callbacks=early_stopping)  

#val_df = pd.read_csv(r'D:\Project\MyPython\titanic\data\val.csv')  
#val_features, val_labels = data_preparer.preprocess(val_df)
#train_history = model.fit(x=train_features, y=train_labels, validation_data=(val_features, val_labels), epochs=50, batch_size=30, verbose=2)  
#print("\n[Info] 訓練成效 (文字)")  
#print(train_history.history)





# 評估模型
#loss_val, acc_val, mse_val = model.evaluate(val_features, val_labels)
#print(f"\n評估模型 : Loss is {loss_val},\nAccuracy is {acc_val * 100},\nMSE is {mse_val}")





# 顯示結果
#print("\n[Info] 訓練成效 (圖表)")
import loss_plot
import acc_plot 
#loss_plot.draw(train_history)
#acc_plot.draw(train_history)  





#sys.exit("stop")

# 預測 (0 : 歿，1 : 存)
Jack = pd.Series([1, 'Jack', 3, 'male', 23, 1, 0, 5.0, 'S']) 
Rose = pd.Series([1, 'Rose', 1, 'female', 26, 1, 0, 100.0, 'S'])
JR_df = pd.DataFrame([list(Jack), list(Rose)], columns=['Survived','Name', 'Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])  

all_df = pd.concat([train_df, JR_df])

print("\n[Info] 預測中...")
features, labels = data_preparer.preprocess(all_df)
all_probability = model.predict(features)  
all_df.insert(len(all_df.columns), 'probability', all_probability) # 加入 生存機率 欄位
print("\n[Info] 預測結果 (傑克 & 蘿絲):\n%s\n" % (all_df[-2:]))
#print("\n[Info] 預測結果 (所有乘客):\n%s\n" % (all_df))


#test_df = pd.read_csv(r'D:\Project\MyPython\titanic\data\test.csv')  
#test_features, test_labels = data_preparer.preprocess(test_df)