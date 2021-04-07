import sys
import pandas as pd 
# import tensorflow as tf 
# featuretools for automated feature engineering
import featuretools as ft
import featuretools.variable_types as vtypes

# from pickle import dump
# from pickle import load
import numpy as np 
# from sklearn import metrics
# import joblib
# import matplotlib.pyplot as plt

# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import GridSearchCV
# from xgboost import plot_importance 

test_month =202101
df_Parts_org = pd.read_csv('./homework_Parts/dataNew/Parts_EQP_Output_ByMonth_20210407.csv', encoding = 'big5')
df_Tool_org = pd.read_csv('./homework_Parts/dataNew/ScmTrainingData_Monthly_60days.csv', encoding = 'big5')
outputFileName='Parts_Tools_60.csv'

df_Tool = df_Tool_org.copy(deep=False)
df_Parts = df_Parts_org.copy(deep=False)

df_Parts['MFG_MONTH'] = df_Parts['STOCK_EVENT_TIME'].apply(lambda x: pd.to_datetime(str(x), format='%Y/%m/%d %H:%M')).dt.strftime('%Y%m')
df_Parts = df_Parts.drop(['STOCK_EVENT_TIME'], axis=1)

df_Tool['TRkey'] = df_Tool['MFG_MONTH'].astype('str') +'_' +df_Tool['TOOL_ID']  
df_Parts['PRkey'] = df_Parts['MFG_MONTH'].astype('str') +'_' +df_Parts['EQP_NO']  

df_Parts = df_Parts[ df_Parts['MFG_MONTH'].astype(int) >= df_Tool['MFG_MONTH'].min().astype(int) ]
df_Parts = df_Parts[ df_Parts['MFG_MONTH'].astype(int) <= df_Tool['MFG_MONTH'].max().astype(int) ]

es = ft.EntitySet(id = 'PART')
#Entityset 把多個實體Entity進行合併
# Tool
es = es.entity_from_dataframe(entity_id = 'Tool',
            dataframe = df_Tool,  
            index ='TRkey',
             )
# Parts
es = es.entity_from_dataframe(entity_id = 'Parts',
             dataframe = df_Parts,  
            make_index = True, 
            index="ID",           
            )   
r= ft.Relationship(es['Tool']['TRkey'],es['Parts']['PRkey'])
es = es.add_relationship(r)

PartsTool_matrix,PartsTool_names= ft.dfs( entityset = es,
                agg_primitives=[],  # agg_primitives=['std','trend'], 
                trans_primitives=[],# trans_primitives=["percentile","is_weekend", "percentile"]
                primitive_options={
                },

                target_entity='Parts')
     
# list(PartsTool_matrix.columns)       
df = pd.DataFrame(PartsTool_matrix.to_records())
df.drop(['PRkey','ID'],axis=1).to_csv(outputFileName,index=False)


# df_train = df[ df['MFG_MONTH'].astype(int) < test_month]
# df_test = df[ df['MFG_MONTH'].astype(int) >=test_month]
