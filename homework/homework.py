
import sys
import pandas as pd 
import tensorflow as tf 
#import data_preparer

print('TEST')
 
train_df = pd.read_csv('./Training_Data.csv') # 800 筆
print(train_df.columns)

 train_df = train_df.drop(['0I', '1C', '1D', '1F', '1G', '1I', '1M', '1N', '1P',
       '1T', '1U', '1V', '2D', '2E', '2F', '2I', '2M', '2N', '2P', '2T', '2U',
       '2V', '3D', '3E', '3G', '3I', '3K', '3M', '3N', '3P', '3S', '3T', '3U',
       '3V', '4D', '4E', '4I', '4M', '4N', '4T', '4U', '4V', '5D', '5E', '5I',
       '5M', '5N', '5P', '5S', '6D', '6I', '6N', '6P', '7D', '7I', '8D', '8I',
       '9D', 'DO', 'ES', 'GN', 'HR', 'LI', 'PO', 'PV', 'SP', 'SV', 'TM', 'TU',
       'TV', 'UG'], axis=1)  
print(train_df.columns)       
