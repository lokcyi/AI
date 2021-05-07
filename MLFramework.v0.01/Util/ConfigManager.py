#import
try:
    import xml.etree.ElementTree as ET
    import socket
except:
    print("model import error!")
class configManager:
    #xml
    common_config = None
    db_config = None

    config_path = None

    entry_key_word = 'Entry'
    db_prifix = 'DbConnectionString'
    server_type = 'Local'

    common_config_file_name = 'CommonParameter.config'
    db_config_file_name = 'DbConnectionString.config'

    common_config_dict = {}
    db_config_dict = {}
    config_path = './config'

    def __init__(self,config_path):
        self.config_path = config_path
        self.load_file()
        self.__get_server_type()

    def GetKey(self,key):
        #先用
        config_value = self.common_config_dict.get(key + '.' + self.db_prifix, None)
        if config_value == None:
            config_value = self.common_config_dict.get(key,None)
        return config_value

    def GetDBKey(self,key):
        final_key = self.db_prifix +'.' + key + '.' +self.server_type
        config_value = self.db_config_dict[final_key]
        return config_value
    #private, 判斷在LOCAL, SIT, UAT, PROD
    def __get_server_type(self):
        serverHostName = socket.gethostname()
        serverFullHostName = socket.getfqdn(serverHostName)
        serverHostIP = socket.gethostbyname(socket.gethostname())
        self.host_name = serverFullHostName

        server_key = ['common.LocalServer','common.SystemIntegrationTestingServer','common.UserAccessTestingServer','common.ProductionServer']
        mapping_server_type = ['Local','SystemIntegrationTesting','UserAccessTesting','Production']
        for idx,key in enumerate(server_key):
            productServerName = self.common_config_dict[key]
            find_index = productServerName.find(serverHostName)
            if find_index==-1:
                find_index = productServerName.find(serverFullHostName)
            if find_index==-1:
                find_index = productServerName.find(serverHostIP)
            if find_index != -1:
                self.server_type = mapping_server_type[idx]
                break
    def read_file(self,file_name):
        file_path = '{}/{}'.format(self.config_path,file_name) #self.config_path + "\\" + file_name
        f = open(file_path,encoding='utf8')
        lines = f.readlines()
        data_str = "".join(lines)
        f.close()
        return data_str

    def load_file(self):
        data_common = self.read_file(self.common_config_file_name)
        data_db = self.read_file(self.db_config_file_name)

        self.common_config = ET.fromstring(data_common)
        self.db_config = ET.fromstring(data_db)

        for node in self.common_config.iter(self.entry_key_word):
            key = node.find('key').text
            value = node.find('value').text
            self.common_config_dict[key] = value

        for node in self.db_config.iter(self.entry_key_word):
            key = node.find('key').text
            value = node.find('value').text
            self.db_config_dict[key] = value




