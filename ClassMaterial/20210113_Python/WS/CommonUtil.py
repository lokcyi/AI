# -*- coding: utf-8 -*-
import pandas as pd
import json

class Util:        
    
    # Convert Json string to Pandas's DataFrames
    @staticmethod
    def ConvertJStr2PdDataFrame( jstr, jencoding = 'utf-8'):        
        return pd.read_json(jstr, encoding = jencoding)    
    
    # Convert Json string to Json Object
    @staticmethod
    def ConvertJStr2JObject( jstr):
        return json.loads(jstr)
    
    # Convert Pandas's DataFrames to Json String
    @staticmethod
    def ConvertPdDataFrame2JStr( pobj , jforce_ascii=False):
        return pobj.to_json(orient='records', force_ascii=jforce_ascii)
    
    #無條件捨去到小數以下n
    @staticmethod
    def GetRoundDownFloat( f_str, n):
        f_str = str(f_str)
        a, b, c = f_str.partition('.')
        c = (c+"0"*n)[:n]       # 先補滿0
        return float(".".join([a, c]))
