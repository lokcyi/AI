import pandas as pd
import sweetviz as sv
pairwise_analysis='off' #相關性和其他型別的資料關聯可能需要花費較長時間。如果超過了某個閾值，就需要設定這個引數為on或者off，以判斷是否需要分析資料相關性。
train = pd.read_csv('D:/Projects/AI/poc/titanic/titanic/train.csv', index_col=0)
test = pd.read_csv('D:/Projects/AI/poc/titanic/titanic/test.csv', index_col=0)

report_train = sv.analyze([train, 'train']) # 'train'是指會給這個資料集命名為train
report_train.show_html(filepath='Basic_train_report.html') # 儲存為html的格式
feature_config = sv.FeatureConfig(skip='Name',  # 要忽略哪個特徵
                                  force_cat=['Pclass', 'Sex', 'SibSp', 'Parch', 'Ticket', 'Cabin'], # Categorical特徵
                                  force_num=['Age', 'Fare'], # Numerical特徵
                                  force_text=None) # Text特徵


report_train_with_target = sv.analyze([train, 'train'],
                                     target_feat='Survived', # 加入特徵變數
                                     feat_cfg=feature_config)
                                     
report_train_with_target.show_html(filepath='Basic_train_report_with_target.html')


compare_report = sv.compare([train, 'Training Data'], # 使用compare
                            [test, 'Test Data'],
                            'Survived',
                            feat_cfg=feature_config)

compare_report.show_html(filepath='Compare_train_test_report.html')

compare_subsets_report = sv.compare_intra(train,
                                          train['Sex']=='male', # 給條件區分
                                          ['Male', 'Female'], # 為兩個子資料集命名 
                                          target_feat='Survived',
                                          feat_cfg=feature_config)

compare_subsets_report.show_html(filepath='Compare_male_female_report.html')