import json
import pandas as pd
import pyLDAvis
import pyLDAvis.sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from Logger import Logger

mylogger = Logger('Model').logger


def load_data_csv(filepath):
    data = pd.read_csv(filepath, encoding='utf-8', header=None)
    data = data.drop(index=[0])
    data = data.drop(columns=[0])
    mylogger.debug('csv数据已经载入')
    return data


def load_data_json(filepath):
    json_file = open(filepath, encoding='utf-8')
    data = json.load(json_file)
    mylogger.debug('json数据已经载入')
    return data


def LDA(data, components, htmlfile=None):
    # 关键词提取和向量转化
    tf_vectorizer = CountVectorizer(max_features=1000,
                                    max_df=0.5,
                                    min_df=10,
                                    encoding='utf-8'
                                    )
    tf = tf_vectorizer.fit_transform(data)
    mylogger.debug('LDA模型训练开始')
    lda = LatentDirichletAllocation(n_components=components,
                                    max_iter=50,
                                    learning_method='online',
                                    learning_offset=50,
                                    random_state=0,
                                    )
    lda.fit(tf)
    mylogger.debug('LDA模型训练完成')
    result = pyLDAvis.sklearn.prepare(lda, tf, tf_vectorizer)
    pyLDAvis.save_html(result, '../data/LDA_Results/' + htmlfile)
    mylogger.debug('LDA模型可视化文件已生成')


