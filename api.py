# api.py
import sys
import urllib.request
import urllib.parse
import json
import pandas as pd
import re
from datetime import datetime

# my_apikeys.py가 같은 경로에 있다고 가정
sys.path.append('./') 
import my_apikeys as mykeys

def get_naver_news(keyword, num_data=1000):
    """
    강의록의 수집 로직을 함수로 변환한 것
    """
    
    # 1. 인증키 가져오기 (my_apikeys.py 사용)
    client_id = mykeys.naver_client_id
    client_secret = mykeys.naver_client_secret

    # 2. 파라미터 설정
    display_count = 100 
    sort = 'date' 
    
    # 검색어 인코딩
    encText = urllib.parse.quote(keyword)
    results = []

    # 3. API 요청 (강의록 로직)
    for idx in range(1, num_data + 1, display_count):
        url = "https://openapi.naver.com/v1/search/news?query=" + encText \
            + f"&start={idx}&display={display_count}&sort={sort}"
        
        request = urllib.request.Request(url)
        request.add_header("X-Naver-Client-Id", client_id)
        request.add_header("X-Naver-Client-Secret", client_secret)
        
        try:
            response = urllib.request.urlopen(request)
            if response.getcode() == 200:
                response_body = response.read()
                response_dict = json.loads(response_body.decode('utf-8'))
                results.extend(response_dict['items'])
        except Exception as e:
            print(f"Error: {e}")
            break

    # 4. 데이터프레임 변환 및 전처리 (강의록 로직)
    if not results:
        return pd.DataFrame()

    df = pd.DataFrame()
    remove_tags = re.compile(r'<.*?>') # 태그 제거 패턴

    clean_data_list = [] # 속도 최적화를 위해 리스트 사용

    for item in results:
        try:
            # 날짜 변환
            pubDate = datetime.strptime(item['pubDate'], "%a, %d %b %Y %H:%M:%S +0900")
            # 태그 제거
            title = re.sub(remove_tags, '', item['title'])
            description = re.sub(remove_tags, '', item['description'])
            
            clean_data_list.append({
                'pubDate': pubDate,
                'title': title,
                'description': description
            })
        except Exception:
            continue

    # 리스트를 DataFrame으로 변환
    df = pd.DataFrame(clean_data_list)
    
    return df