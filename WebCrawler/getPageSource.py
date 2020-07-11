import json
import random
import time
import urllib
import requests
from requests import RequestException
from config import HEADERS


def get_page_html(url):
    headers = HEADERS
    try:
        time.sleep(random.randint(1, 2))
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            html = response.text
            print("已获取到html数据")
            return html
        return None
    except RequestException:
        return None


def get_json_data(url):
    time.sleep(random.randint(3, 4))
    url_session = requests.Session()
    html = url_session.get(url).text
    json_data = []
    if html == '':
        time.sleep(random.randint(3, 4))
        html = urllib.request.urlopen(url).read().decode('gbk', 'ignore')
        json_data = html[20:-2]
    else:
        json_data = html[1:-2]
    if json_data == '':
        print("访问过快，获取失败，10min后再访问")
        time.sleep(600)
        html = urllib.request.urlopen(url).read().decode('gbk', 'ignore')
        json_data = html[20:-2]
    data = json.loads(json_data)
    print("已获取到json数据")
    return data
