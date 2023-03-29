from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
import csv
from selenium.webdriver.common.by import By
from selenium import webdriver
import re
from time import sleep



# 위치가 담겨있는 csv파일 열기
f = open('pusan_loc.csv','r', encoding="utf-8" )
rdr = csv.reader(f)
 
roc = []
for line in rdr:
    roc.append(line[1])
 
f.close()
roc = roc[1:69]
# print(len(roc))
print(roc)

for abc in range(len(roc)):

    re = '부산 '
    search =  re + roc[abc]

    hrefs = []

    for page in range(1,11):
        headers = {
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.5563.111 Safari/537.36',
    }
        data = {
        'type': '',
        'query': search,
        'lat': '',
        'lng': '',
        'dis': '',
        'page': str(page),
        'chunk': '10',
        'rn': ''
        }
        response = requests.post('https://www.diningcode.com/2018/ajax/list.php', headers=headers, data=data)

        
        soup = BeautifulSoup(response.text, 'html.parser')
        stores = soup.find_all("a", class_="blink")
        
        for store in stores:
            hrefs.append(store['href'].split('rid=')[-1])
    # print(hrefs)

    front_url = 'https://www.diningcode.com/profile.php?rid='
    headers2 = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.5563.111 Safari/537.36',
    }
    f = open(f"res_data{abc}.csv", 'w', encoding='utf-8')
    f.write('Name,score,userscore,tag,roca,url,menu\n')
    f.write(search+','+'score'+','+'userscore'+','+'tag'+','+'roca'+','+'url'+','+'menu''\n')
    tot_str=''
    for code in hrefs:
        herf_url = front_url + code
        response = requests.post(herf_url, headers=headers2)
        soup = BeautifulSoup(response.text, 'html.parser')
        name = soup.find('div',class_='tit-point').get_text().replace('\n', '').replace(',','.')

        # print(name)
        roca = soup.find('li',class_='locat').get_text().replace('\n', '').replace(',', '')
        
        try:
            menu = soup.find('ul', class_='list Restaurant_MenuList').get_text().replace('\n', ' ').replace(',', '')
        # print(menu)
        except:
            menu = '없음'
    
        for k in range(1, len(menu)-1):
            if menu[k].isdigit() and menu[k-1] == ' ':
                menu = menu[:k-1] + ':' + menu[k:]
        
        
        
        try:
            score = soup.find('p', class_='grade').find('strong')
            for i in score:
                if len(i) == 3:
                    score = i
                else:
                    score = '15점'

        except:
            score = '15점'


        try:
            userscore = soup.find(id='lbl_review_point')
            userscore = userscore.get_text()
        except:
            userscore = '리뷰없음'

        tag = soup.find('div',class_='s-list pic-grade').find('div',class_='btxt').get_text().strip().replace(',',' ')
        temp = tag.find('| ') + 2
        tag = tag[temp:]
        f.write(name+','+str(score)+','+userscore+','+tag+','+roca+','+herf_url+','+menu+'\n')
    
    f.close()

    time.sleep(2) 





# location = ['부산광역시 영도구 봉래동2가 39-1 ','부산광역시 영도구 봉래동3가 17-2 ']


driver = webdriver.Chrome('chromedriver') # 열기
driver.implicitly_wait(3) # 3초안에 로드하면 넘어감, 못 하면 3초 기다림
driver.get('https://address.dawul.co.kr/#')
search_box = driver.find_element('xpath' , '//*[@id="input_juso"]')
search_button = driver.find_element('xpath' , '//*[@id="btnSch"]')


for abc in range(len(roc)):
    
    df1 = pd.read_csv(f'res_data{abc}.csv', encoding='utf-8')
    roca = df1['roca']
# print(roca)
    list1 = []

    for i in roca:
        search_box.send_keys(i) # 검색
        sleep(0.1)
        search_button.click() # 버튼 클릭
        sleep(0.1)
        loc = driver.find_element('xpath' , '//*[@id="insert_data_5"]')  # 검색
        # print(loc.text) # 프린트
        list1.append(loc.text)


    df1['coordinate'] = list1

    df1.to_csv(f'res_data{abc}.csv', index = False)