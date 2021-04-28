import pandas as pd
import os
import numpy as np
import sweetviz as sv

from Util.Logger import Logger
class EDA:
    log = Logger(name='MLFramework')
    @staticmethod
    def analysis(df,targetfeat):
        pairwise_analysis='on' #相關性和其他型別的資料關聯可能需要花費較長時間。如果超過了某個閾值，就需要設定這個引數為on或者off，以判斷是否需要分析資料相關性。
        report_train = sv.analyze([df, 'train'],
                                        target_feat= targetfeat,
                                        pairwise_analysis = pairwise_analysis
        )
        report_train.show_html(filepath='./report/EDA_AnalysisReport.html' ) # 儲存為html的格式

    @staticmethod
    def compare(df_train,df_test,targetfeat):
        pairwise_analysis='on' #相關性和其他型別的資料關聯可能需要花費較長時間。如果超過了某個閾值，就需要設定這個引數為on或者off，以判斷是否需要分析資料相關性。
        compare_subsets_report = sv.compare([df_train, 'Train'], # 使用compare
                            [df_test, 'Test'],
                             target_feat= targetfeat,
                            pairwise_analysis = pairwise_analysis
                             )


        compare_subsets_report.show_html(filepath='./report/EDA_CompareReport.html')
