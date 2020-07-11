
DEFAULT_PRODUCT_NUM = 5

DEFAULT_COMMENT_NUM = 60000

SEARCH_URL = "https://search.jd.com/Search?keyword={}&enc=utf-8&pvid=72135190d8db4d388028dd72312dfe34"

PRODUCT_URL = "https://item.jd.com/{}.html"

COMMENT_URL = "https://club.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98&productId={" \
              "}&score={}&sortType=5&page={}&pageSize=10&isShadowSku=0&rid=0&fold=1 "

PRICE_URL = "https://p.3.cn/prices/mgets?skuIds=J_{}"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.1.6) Gecko/20091201 Firefox/3.5.6'
}

DATABASE_MONGO = {
    'host': 'localhost',
    'port': '127.0.0.1'
}

