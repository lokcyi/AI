import sys
import os

parentDir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
sys.path.append(parentDir)
import entity.DBEngine as db_engine
# import DBEngine as db_engine 

 
class PPMDAO: 
    db_name = 'MPS'
    """
    def 撈取測試資料
    """    
    def get_toolg_kpi_by_Toolg_id(self, toolg_list=None):
        query = ['select * from PPM.dbo.VW_TOOLG_KPI ' ]
        if toolg_list is not None:
            if len(toolg_list) > 1:
                _toolg_list =tuple(toolg_list)
                query.append('where TOOLG_ID in {0} '.format(_toolg_list))   
            else:
                query.append('where TOOLG_ID = \'{0}\' '.format(toolg_list[0]))     

        query.append(' order by MFG_DATE,TOOLG_ID ')        
        conn = db_engine.DBEngine(self.db_name)
        df = conn.Query(' '.join(query))
        return df
            

if __name__ == "__main__": 
    PPM = PPMDAO()
    print(PPM.get_toolg_kpi_by_Toolg_id(['PK_DUVKrF']))
