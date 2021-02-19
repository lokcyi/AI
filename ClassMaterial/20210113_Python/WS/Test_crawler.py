import requests
from bs4 import BeautifulSoup
import os
from datetime import datetime
from shutil import copyfile

total_news_count = 0
#聯合新聞網
def udn_news():
    global total_news_count
    nav_list = [] #導覽
    title_href_list = [] #標題
    text_list = [] #內容

    udn_home = requests.get('https://udn.com/news/index', verify=False)
    nav = BeautifulSoup(udn_home.text, 'html.parser')
    nav_content = nav.select('.navigation a')    
    for nav_item in nav_content:
        nav_item_href = nav_item.attrs['href']
        if nav_item_href.startswith('/news') and nav_item_href.find('index') == -1:
            nav_list.append(nav_item_href)
        
    for web_item in nav_list:
        web = requests.get('https://udn.com{0}'.format(web_item), verify=False)
        content_list = BeautifulSoup(web.text, 'html.parser')
        allTitle = content_list.select('.rounded-thumb__list h3 a')   
        for i in allTitle:
            attrs = i.attrs        
            content_href = attrs['href']
            content_url = 'https://udn.com{0}'.format(content_href)        
            title_href_list.append(content_url)
    news_count = 0
    for item in title_href_list:
        content_web = requests.get(item, verify=False)
        content = BeautifulSoup(content_web.text, 'html.parser')
        allElement = content.find('section',itemprop='articleBody')
        if allElement is not None:
            text = allElement.text.replace('\n', '').replace('\r', '')
            if text :
                text_list.append('Base_{0} : {1}'.format(str(total_news_count),text))
                total_news_count = total_news_count +1
                news_count = news_count + 1
    print('und news : {0}'.format(str(news_count)))
    return text_list

#自由時報
def ltn_news():
    global total_news_count
    nav_list = [] #導覽
    title_href_list = [] #標題
    text_list = [] #內容
    
    ltn_home = requests.get('https://www.ltn.com.tw/', verify=False)
    nav = BeautifulSoup(ltn_home.text, 'html.parser')
    nav_content = nav.select('.useMobi ul li')    
    for nav_item in nav_content:
        nav_item_href = nav_item.next.attrs['href']
        nav_list.append(nav_item_href)

    for web_item in nav_list:
        web = requests.get(web_item, verify=False)
        content_list = BeautifulSoup(web.text, 'html.parser')
        allTitle = content_list.select('.list li')   
        for i in allTitle:    
            content_href = i.contents[3].attrs['href']      
            title_href_list.append(content_href)
    news_count = 0
    for item in title_href_list:
        content_web = requests.get(item, verify=False)
        content = BeautifulSoup(content_web.text, 'html.parser')
        allElement = content.select('div.text.boxTitle.boxText p')
        if allElement is not None:
            text = ''
            for text_content in allElement:
                tmp = text_content.text.replace('\n', '').replace('\r', '').strip()
                text = text+tmp
            if text :    
                text_list.append('Base_{0} : {1}'.format(str(total_news_count),text))
                total_news_count = total_news_count +1
                news_count = news_count + 1
    print('ltn news : {0}'.format(str(news_count)))

    return  text_list

def join_file(output_folder,output_file,be_joined_file):
    dst_file = 'TrainData_'+datetime.today().strftime('%Y-%m-%d')+'.txt'
    #將爬蟲news file先複製到TrainData
    copyfile(os.path.join(output_folder,output_file),os.path.join(output_folder,dst_file))
    #取得TrainData現有資料筆數
    with open(os.path.join(output_folder,dst_file) , 'r', encoding = 'utf-8') as train_f :
        lines = train_f.readlines()
        line_count = len([l for l in lines if l.strip(' \n') != ''])

    with open(os.path.join(output_folder,dst_file) , 'a', encoding = 'utf-8') as f:
        read_f = open(os.path.join(output_folder,be_joined_file) , 'r', encoding = 'utf-8') 
        for line in read_f :
            FullText = ":".join(line.split(":")[1:]).strip()
            text = 'Base_{0} : {1}\n'.format(str(line_count),FullText)
            f.write(text)
            line_count = line_count + 1

def join_file_unique(output_folder,output_file,be_joined_file):
    dst_file = 'TrainData_'+datetime.today().strftime('%Y-%m-%d')+'_unique.txt'
    text_list = []
    #取得今日新聞資料
    t_f = open(os.path.join(output_folder,output_file) , 'r', encoding = 'utf-8')
    for line in t_f :
        FullText = ":".join(line.split(":")[1:]).strip()
        if FullText is not None and FullText != '':
            text_list.append(FullText)
    text_list = set(text_list)
    #取得待合併資料
    j_text_list = []
    j_f = open(os.path.join(output_folder,be_joined_file) , 'r', encoding = 'utf-8') 
    for line in j_f :
        FullText = ":".join(line.split(":")[1:]).strip()
        if FullText is not None and FullText != '':
            j_text_list.append(FullText)
    #寫入檔案    
    line_count = 0
    with open(os.path.join(output_folder,dst_file) , 'w', encoding = 'utf-8') as f:
        for item in j_text_list:
            text = 'Base_{0} : {1}\n'.format(str(line_count),item)
            f.write(text)
            line_count = line_count + 1
        for item in text_list:
            text = 'Base_{0} : {1}\n'.format(str(line_count),item)
            f.write(text)
            line_count = line_count + 1

def write_file(output_folder,output_file,write_type,news):
    with open(os.path.join(output_folder,output_file) , write_type, encoding = 'utf-8') as f:
        for item in news:
            f.write(item+'\n')
    f.close()

if __name__ == '__main__':
    output_folder = 'D:\\0-Learning\\Uplanning\\Class\WS\\TestData'
    output_file = 'News_'+datetime.today().strftime('%Y-%m-%d')+'.txt'    
    und_text = udn_news()
    write_file(output_folder,output_file,'w',und_text)
    ltn_text = ltn_news()
    write_file(output_folder,output_file,'a',ltn_text)
    # be_joined_file = 'TrainData_2020-11-10_unique.txt'
    # join_file(output_folder,output_file,be_joined_file)
    # join_file_unique(output_folder,output_file,be_joined_file)
