import re
from config import *
from WebCrawler.getPageSource import *


def get_search_url(keywords):
    search_url = SEARCH_URL.format(keywords)
    return search_url


def get_product_id(url, product_num=DEFAULT_PRODUCT_NUM):
    html = get_page_html(url)
    results = re.findall('<li .*?data-sku=\"(.*?)\"', html, re.S)
    print("已获取产品ID")
    return results[:product_num]


def get_product_data(product_id):
    try:
        product_url = PRODUCT_URL.format(product_id)
        product_html = get_page_html(product_url)
        good_id = (re.findall('>货号：(.*?)</li>', product_html, re.S))[0]
        brand = (re.findall("id=\"parameter-brand.*?title=\'(.*?)\'>品牌", product_html, re.S))[0]
        price_url = PRICE_URL.format(product_id)
        # print(price_url)
        json_data = get_json_data(price_url)
        price = json_data['p']
        product_data = {
            "good_id": good_id,
            "brand": brand,
            "price": price
        }
        print("已获取商品信息")
        return product_data
    except IndexError:
        return None


def get_comment_num(product_id):
    comment_url = COMMENT_URL.format(product_id, '1', '0')
    # print(comment_url)
    json_data = get_json_data(comment_url)
    comment_num = json_data['productCommentSummary']['commentCount']
    # print(comment_num)
    return comment_num


def get_comment_data(product_id, score, page, product):
    comment_url = COMMENT_URL.format(product_id, score, page)
    json_data = get_json_data(comment_url)
    data = json_data['comments']
    comment_data = []
    for i in range(len(data)):
        # yield {
        #     "good_id": product['good_id'],
        #     "brand": product['brand'],
        #     "price": product['price'],
        #     "creationTime": data[i]['creationTime'],
        #     "score": data[i]['score'],
        #     "comment": data[i]['content']
        # }
        comment = {
            "good_id": product['good_id'],
            "brand": product['brand'],
            "price": product['price'],
            "creationTime": data[i]['creationTime'],
            "score": data[i]['score'],
            "comment": data[i]['content']
        }
        comment_data.append(comment)
    print("已获取{}条数据".format(len(comment_data)))
    return comment_data