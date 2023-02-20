import urllib.request
import pandas as pd
import requests

class Crawl():
    def __init__(self, client_id, client_secret, args):

        self.client_id = client_id
        self.client_secret = client_secret
        self.args = args
    
    def __call__(self, query, number):

        headers = {
        "X-Naver-Client-Id":self.client_id,
        "X-Naver-Client-Secret":self.client_secret
        }

        number_of_iter = (int(number) // 100) + 1

        crawl_result = []
        start = 1
        for i in range(number_of_iter):

            if i == (int(number) // 100):
                display = int(number) % 100
            else:
                display = 100
            

            data = {
                'query':query, # 검색어
                'display':display, # 검색 개수 default : 10, max : 100
                'start':start, # 검색 시작 위치 default : 1, max : 100
                'sort':self.args.sort # 검색 결과 정렬 방법 : sim 정확도순, date 날짜순
            }
            
            # 100 단위로 떨어지는 number 값이 들어오면 마지막 iter때 display 값이 0이됨
            if display:
                data = urllib.parse.urlencode(data)
                result = requests.get("https://openapi.naver.com/v1/search/news.json?", headers=headers, params=data)

                crawl_result.extend(result.json()['items'])

            start += 100

        # 아 여기는 원래 본문 없음
        crawl_result = pd.DataFrame(crawl_result)
        return crawl_result