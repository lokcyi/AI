# 3. Group By 月 或 季 (不看機台)
import pandas as pd
import numpy as np
from Util.Data import Data 


df_Parts = pd.read_csv(r'D:\AI_Parts線邊合理量\08-MLFramework.v0.01\data\原始資料\Parts_EQP_Output_ByMonth_20210407.csv',encoding="big5hkscs" )
list_Parts = df_Parts['PART_NO'].unique().tolist()


# def accsum(def_result,target_cols='QTY'):    
#     _accsum=0
#     for index,row in def_result.iterrows():
#         #避免當分母為0 會無法計算
#         if row[target_cols]==0 and row['Predict']==0 :
#             row[target_cols] =1
#             row['Predict'] =1
#         elif row[target_cols] ==0 and row['Predict']!=0:
#             row[target_cols]  =0.00001

#         if row[target_cols] <0 :
#             row[target_cols]  =0.00001

#         if row['Predict'] <0 :
#             row['Predict']  =0

#         if 1- abs((row['Predict'] - row[target_cols])/row[target_cols] ) >0 :
#             _accsum+=(1- abs((row['Predict'] - row[target_cols])/row[target_cols] ))

#     return round(_accsum*100/def_result.shape[0],2)
    

# 3-1. 資料未增量 Group By 月
# 機台資料下去trian , Group By 月 
# for partno in  list_Parts:
#     sourcePathDNN ='./Report/Parts_Tools_Day30_'+partno+'_DNN.csv'
#     sourcePathXG = './Report/Parts_Tools_Day30_'+partno+'_XG_SCM.csv'
#     df_DNN  = pd.read_csv(sourcePathDNN )
#     df_XG   = pd.read_csv(sourcePathXG )
    
#     # df_DNN_SUM = df_DNN.set_index(['MFG_MONTH']).groupby('MFG_MONTH')[['QTY','Predict']].sum()
#     # df_XG_SUM = df_XG.set_index(['MFG_MONTH']).groupby('MFG_MONTH')[['QTY','Predict']].sum()
#     # print(df_DNN_85ECT0010_SUM)

#     df_DNN_SUM = df_DNN.groupby('MFG_MONTH')[['QTY','Predict']].sum().reset_index()
#     df_XG_SUM = df_XG.groupby('MFG_MONTH')[['QTY','Predict']].sum().reset_index()

#     accDNN = accsum (df_DNN_SUM)
#     accXG = accsum (df_XG_SUM)
#     print(partno+' Month XG ACC=', accXG)
#     print(partno+' Month DNN ACC=', accDNN)
    


# # 加入特徵:前3個月平均用量 & 機台資料下去trian , Group By 季  (效果比較不好)
# for partno in  list_Parts:
#     sourcePathDNN ='./Report/Parts_Tools_Day30_Quarter_'+partno+'_DNN.csv'
#     sourcePathXG = './Report/Parts_Tools_Day30_Quarter_'+partno+'_XG_SCM.csv'
#     df_DNN  = pd.read_csv(sourcePathDNN )
#     df_XG   = pd.read_csv(sourcePathXG )

#     df_DNN_SUM = df_DNN.groupby('MFG_MONTH')[['QTY','Predict']].sum().reset_index()
#     df_XG_SUM = df_XG.groupby('MFG_MONTH')[['QTY','Predict']].sum().reset_index()

#     accDNN = accsum (df_DNN_SUM)
#     accXG = accsum (df_XG_SUM)
# #     accDNN =accsum(df_DNN_SUM)*100
# #     accXG = accsum(df_XG_SUM)*100
#     print(partno+' Quarter XG ACC=', accXG)
#     print(partno+' Quarter DNN ACC=', accDNN)
    




# 3-2. 資料未增量 Group By 季
# 機台資料下去trian , Group By 季
for partno in  list_Parts:
    sourcePathXG = './Report/Parts_Tools_Day30_Quarter_'+partno+'_XG_SCM.csv'
    sourcePathDNN ='./Report/Parts_Tools_Day30_Quarter_'+partno+'_DNN.csv'
    
    df_DNN  = pd.read_csv(sourcePathDNN )
    df_XG   = pd.read_csv(sourcePathXG )
    
    # df_DNN_SUM = df_DNN.set_index(['MFG_MONTH']).groupby('MFG_MONTH')[['QTY','Predict']].sum()
    # df_XG_SUM = df_XG.set_index(['MFG_MONTH']).groupby('MFG_MONTH')[['QTY','Predict']].sum()
    # print(df_DNN_85ECT0010_SUM)

    df_DNN_SUM = df_DNN.groupby('MFG_MONTH')[['QTY','Predict']].sum().reset_index()
    df_XG_SUM = df_XG.groupby('MFG_MONTH')[['QTY','Predict']].sum().reset_index()

    accDNN = Data.accsum(df_DNN_SUM,'QTY')
    accXG = Data.accsum(df_XG_SUM,'QTY')
    print(partno+' Quarter XG ACC=', accXG)
    print(partno+' Quarter DNN ACC=', accDNN)



# # 3-3. 資料增量 Group By 月
# # 機台資料下去trian , Group By 月 
# for partno in  list_Parts:
#     sourcePathDNN ='./Report/Parts_Tools_Day30_AFMONTH_'+partno+'_DNN.csv'
#     sourcePathXG = './Report/Parts_Tools_Day30_AFMONTH_'+partno+'_XG_SCM.csv'
#     df_DNN  = pd.read_csv(sourcePathDNN )
#     df_XG   = pd.read_csv(sourcePathXG )
    
#     # df_DNN_SUM = df_DNN.set_index(['MFG_MONTH']).groupby('MFG_MONTH')[['QTY','Predict']].sum()
#     # df_XG_SUM = df_XG.set_index(['MFG_MONTH']).groupby('MFG_MONTH')[['QTY','Predict']].sum()
#     # print(df_DNN_85ECT0010_SUM)

#     df_DNN_SUM = df_DNN.groupby('MFG_MONTH')[['QTY','Predict']].sum().reset_index()
#     df_XG_SUM = df_XG.groupby('MFG_MONTH')[['QTY','Predict']].sum().reset_index()

#     # accDNN = (accsum (df_DNN_SUM).astype(float))*100
#     # accXG = (accsum (df_XG_SUM).astype(float))*100

#     accDNN =(accsum(df_DNN_SUM))*100
#     accXG = (accsum(df_XG_SUM))*100
#     print(partno+' Month XG ACC=', accXG)
#     print(partno+' Month DNN ACC=', accDNN)


# # 3-4. 資料增量 Group By 季
# # 機台資料下去trian , Group By 季 
# for partno in  list_Parts:
#     sourcePathDNN ='./Report/Parts_Tools_Day30_AFQuarter_'+partno+'_DNN.csv'
#     sourcePathXG = './Report/Parts_Tools_Day30_AFQuarter_'+partno+'_XG_SCM.csv'
#     df_DNN  = pd.read_csv(sourcePathDNN )
#     df_XG   = pd.read_csv(sourcePathXG )
    
#     # df_DNN_SUM = df_DNN.set_index(['MFG_MONTH']).groupby('MFG_MONTH')[['QTY','Predict']].sum()
#     # df_XG_SUM = df_XG.set_index(['MFG_MONTH']).groupby('MFG_MONTH')[['QTY','Predict']].sum()
#     # print(df_DNN_85ECT0010_SUM)

#     df_DNN_SUM = df_DNN.groupby('MFG_MONTH')[['QTY','Predict']].sum().reset_index()
#     df_XG_SUM = df_XG.groupby('MFG_MONTH')[['QTY','Predict']].sum().reset_index()

#     # accDNN = (accsum (df_DNN_SUM).astype(float))*100
#     # accXG = (accsum (df_XG_SUM).astype(float))*100

#     accDNN =(accsum(df_DNN_SUM))*100
#     accXG = (accsum(df_XG_SUM))*100
#     print(partno+' Quarter XG ACC=', accXG)
#     print(partno+' Quarter DNN ACC=', accDNN)