from WebCrawler.getData import *
from WebCrawler.saveData import *

# keywords = input('输入商品关键字：')
search_url = get_search_url('美的热水器热评')
print("开始获取商品ID")
products = get_product_id(search_url)
for i in range(len(products)):
    print("开始获取第{}个商品信息".format(i + 1))
    product = get_product_data(products[i])
    print("开始获取第{}个商品评论数".format(i + 1))
    comment_num = get_comment_num(products[i])
    print("已第{}个商品评论数".format(i + 1), comment_num)
    for k in range(7, 8):
        for j in range(int(comment_num / 10)):
            print("开始获取第{}个商品第{}页评分{}的评论".format(i + 1, j + 1, k))
            comment_data = get_comment_data(products[i], k, j, product)
            print("第{}个商品第{}页评分{}的评论将存入数据库".format(i + 1, j + 1, k))
            if not comment_data:
                break
            save_to_mongo(comment_data)
            print("第{}个商品第{}页评分{}的评论已存入数据库".format(i + 1, j + 1, k))
