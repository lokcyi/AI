import pymssql as pymssql
import pandas as pd
import sys
import os
from cryptography.fernet import Fernet
parentDir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
sys.path.append(parentDir)
import Util.ConfigManager as cf

#conda install -c anaconda cryptography
class DBEngine:
    def __init__(self,db_name='',config_path='./config'):

        self.config_path = config_path
        self.dbName = db_name
        self.ServerPrifix = 'Server'
        self.DBNamePrifix = 'Database'
        self.UserIDPrifix = 'uid'
        self.PwdPrifix = 'pwd'
        self.config_dic = {}
        self.config_k=b'XNg5-5Dph7HeGhYRj58XwEcEKaCda3i96rU1rqxZh0Y='
        # self.cfo=None
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

    def encrpyt(self,data):
        fernet = Fernet(self.config_k)
        # then use the Fernet class instance
        # to encrypt the string string must must
        # be encoded to byte string before encryption
        encData = fernet.encrypt(data.encode())

        print("encrypt string: ", encData.decode("utf-8") [:-2])

if __name__ == "__main__":
    utl = DBEngine()
    token =utl.encrpyt('test')
    print(token)