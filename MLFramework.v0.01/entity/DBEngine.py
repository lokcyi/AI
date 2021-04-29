import pymssql as pymssql
import pandas as pd
import sys
import os
import Util.ConfigManager as cf
from cryptography.fernet import Fernet
#conda install -c anaconda cryptography
class DBEngine:
    def __init__(self,db_name,config_path='./config'):
        self.dbName = db_name
        self.config_path = config_path
        self.dbName = db_name
        self.ServerPrifix = 'Server'
        self.DBNamePrifix = 'Database'
        self.UserIDPrifix = 'uid'
        self.PwdPrifix = 'pwd'
        self.config_dic = {}
        self.config_k=b'XNg5-5Dph7HeGhYRj58XwEcEKaCda3i96rU1rqxZh0Y='
    def Query(self,sqlstring,params=None):
        conn = self.conn()
        cursor = conn.cursor(as_dict = True)
        cursor.execute(sqlstring)
        data=cursor.fetchall()
        df =pd.DataFrame(data)
        cursor.close()
        conn.close()
        return df

    def parsingConfig(self):
        cfo = cf.configManager(self.config_path)
        dbconnct_str = cfo.GetDBKey(self.dbName)
        dblist = dbconnct_str.split(';')
        for i in range(0,len(dblist)):
            temp =  dblist[i]
            split_temp = temp.split('=')
            self.config_dic[split_temp[0]] = split_temp[1]


    def conn(self):
        self.parsingConfig()
        fernet = Fernet(self.config_k)
        conn = pymssql.connect(host=self.config_dic[self.ServerPrifix],user = self.config_dic[self.UserIDPrifix],password = str(fernet.decrypt( bytes(self.config_dic[self.PwdPrifix]+'==', 'ascii')) , encoding='UTF-8'),database=self.config_dic[self.DBNamePrifix])
        return conn



