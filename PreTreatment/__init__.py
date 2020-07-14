import json

from Logger import Logger
from PreTreatment.dataProcess import del_duplicate, compress, get_tag_comment
from PreTreatment.findData import get_data_with_mongo

mylogger = Logger('PreTreatment').logger

if __name__ == '__main__':
    mylogger.info("获取数据库数据")
    result = get_data_with_mongo()
    mylogger.info("MongoDB获取数据量 -- {}".format(len(result)))

    mylogger.info("评论数据抽取")
    data = result.drop(['_id'], axis=1)
    data['creationDate'] = data['creationTime'].apply(lambda x: x[:10])
    data = data.drop(['creationTime'], axis=1)
    data.to_csv('../data/000_meidi_data_origin.txt',
                index=False, header=True)

    mylogger.info("评论数据去重")
    data_del_dup = del_duplicate(data)
    data_del_dup.to_csv('../data/001_meidi_data_deldup.txt',
                        index=False, header=True)
    mylogger.info('去重后数据量 -- {}'.format(len(data_del_dup)))

    mylogger.info("评论数据机械压缩机械压缩")
    data_compressed = data_del_dup.copy()
    data_compressed['compcomm'] = data_compressed['comment'].apply(compress)
    data_compressed['comment'] = data_compressed['compcomm']
    data_compressed = data_compressed.drop(['compcomm'], axis=1)
    temp = data_compressed["comment"].apply(len)
    data_compressed = data_compressed[temp > 4]
    data_compressed.to_csv('../data/002_meidi_data_comressed.txt', index=False, header=True)
    mylogger.info('机械压缩后数据量 -- {}'.format(len(data_compressed)))

    mylogger.info("标签评论数据的抽取")
    tag_comment_dict = get_tag_comment(data_del_dup)
    with open('../data/003_meidi_tagComment.json', 'w', encoding='utf-8') as fw:
        json.dump(tag_comment_dict, fw, indent=4)
    mylogger.info("抽取的评论标签 -- {}".format(tag_comment_dict.keys()))
    mylogger.info("数据预处理完成")
