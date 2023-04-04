# 크롤링(다이닝코드 크롤링)

- 사용한 모듈

```python
from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
import csv
from selenium.webdriver.common.by import By
from selenium import webdriver
import re
from time import sleep
```

- 단순히 BeautifulSoup을 이용해서는 왠만한 사이트에서 크롤링이 불가능 했다
- 그래서 검색한 결과
```python
headers2 = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.5563.111 Safari/537.36',
}
# 헤더를 지정해주고
response = requests.post(herf_url, headers=headers2)
```
- 이렇게 해주면 아주 잘 되는 것을 볼 수 가 있었다

- 3/29 스터디 모임에서 나연님께서 이야기 해주셨던 대로
- 동적 할당, 즉 클릭이 필요한 작업이나, 스크롤을 내려야 하는 작업에서는
- selenium을 이용해야 했다
- 단순하게 개인 노트북으로 다운 받으려면 chrome driver을 이용하면 되는데
- aws lambda는 어떤지 모르겠다
- 아무튼 driver를 이용해서 검색창에 값을 넣고 버튼을 눌렀다

```python
search_box.send_keys(i) # 검색
search_button.click() # 버튼 클릭
```

- 그랬더니 원활 하게 성공
- 이후 가져왔던 데이터들을 pandas를 이용해서 테이블을 만들고 저장하였다

- 이렇게 해도 생각보다 에러도 많이 나고 결측치도 많았다
- 그런 값들을 일일이 전처리 해야 되는 일이 남아있다