import pymssql as pymssql
import pandas as pd
import sys
import os
import Util.ConfigManager as cf
class DBEngine:
    def __init__(self,db_name):
        self.dbName = db_name
        self.ServerPrifix = 'Server'
        self.DBNamePrifix = 'Database'
        self.UserIDPrifix = 'uid'
        self.PwdPrifix = 'pwd'

        self.config_dic = {}
        self.config_path = '.\config'
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
        conn = pymssql.connect(host=self.config_dic[self.ServerPrifix],user = self.config_dic[self.UserIDPrifix],password = self.config_dic[self.PwdPrifix],database=self.config_dic[self.DBNamePrifix])
        return conn



