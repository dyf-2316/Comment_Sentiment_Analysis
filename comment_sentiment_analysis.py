# -*- coding:utf-8 -*-
# @Time： 2020/7/13 11:48 AM
# @Author: dyf-2316
# @FileName: commet_sentiment_analysis.py
# @Software: PyCharm
# @Project: Comment_Sentiment_Analysis
# @Description: visualize this project
import base64
import json
import webbrowser

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

plt.rcParams['font.sans-serif'] = ['SimHei']

st.title('基于python的电商产品评论数据情感分析')
st.markdown('🌈 组6  dyf-2316 :dog: Joshua_yi :see_no_evil: hzkzTech ☠️ msh :feet: &ensp; :star2: [GitHub]('
            'https://github.com/dyf-2316/Comment_Sentiment_Analysis) :star2:')


def login(path):
    """
    目标路径的跳转，在远程访问时无法使用
    :param path: (str)
    :return: None
    """
    # webbrowser.get('Safari').open(u"https://dyf-2316.github.io/LDA_Results/" + path)
    webbrowser.open(u"https://dyf-2316.github.io/LDA_Results/" + path)


@st.cache
def load_image(path):
    """
    加载路径下的图片内容
    :param path: (str)
    :return: (Image)
    """
    image = Image.open(path)
    return image


@st.cache
def load_data_origin():
    """
    加载原始数据
    :return: (DataFrame)
    """
    data_origin = pd.read_csv('data/000_meidi_data_origin.txt', encoding='utf-8')
    return data_origin


@st.cache
def load_data_deldup():
    """
    加载去重数据
    :return: (DataFrame)
    """
    data_deldup = pd.read_csv('data/001_meidi_data_deldup.txt', encoding='utf-8')
    return data_deldup


@st.cache
def load_data_compress():
    """
    加载机械压缩和短句删除后的数据
    :return: (DataFrame)
    """
    data_compress = pd.read_csv('data/002_meidi_data_comressed.txt', encoding='utf-8')
    return data_compress


@st.cache
def load_tag_comments_origin():
    """
    加载标签评论数据
    :return: (dict)
    """
    with open('data/003_meidi_tagComment.json', 'r', encoding='utf-8') as f:
        tag_comments = json.load(f)
    return tag_comments


@st.cache
def load_comments_sentiment():
    """
    加载情感分析后的数据
    :return: (DataFrame)
    """
    data_classify = pd.read_csv('data/004_meidi_data_sentiment.txt', encoding='utf-8')
    return data_classify


@st.cache
def load_neg_comment():
    """
    加载消极评论数据
    :return: (DataFrame)
    """
    data = pd.read_csv('data/005_neg_comment.csv', encoding='utf-8', header=None)
    data = data.drop(index=[0])
    data = data.drop(columns=[0])
    return data


@st.cache
def load_pos_comment():
    """
    加载积极评论数据
    :return: (DataFrame)
    """
    data = pd.read_csv('data/005_pos_comment.csv', encoding='utf-8', header=None)
    data = data.drop(index=[0])
    data = data.drop(columns=[0])
    return data


@st.cache
def load_tag_comments_pos():
    """
    加载积极标签评论数据
    :return: (dict)
    """
    with open('data/006_tag_pos_comments.json', 'r', encoding='utf-8') as f:
        tag_comments_pos = json.load(f)
    return tag_comments_pos


@st.cache
def load_tag_comments_neg():
    """
    加载消极标签评论数据
    :return: (dict)
    """
    with open('data/006_tag_neg_comments.json', 'r', encoding='utf-8') as f:
        tag_comments_neg = json.load(f)
    return tag_comments_neg


@st.cache
def load_LDA_coherence():
    """
    加载LDA主题连贯性结果数据
    :return: (DataFrame)
    """
    with open('data/007_LDA_cv_coherence.json', 'r', encoding='utf-8') as f:
        LDA_coherence = json.load(f)
    return LDA_coherence


###
# 以缓存装饰器的机制来对所用数据预先加载，提高使用流畅度

data_origin = load_data_origin()
data_deldup = load_data_deldup()
data_compress = load_data_compress()
tag_comment_origin = load_tag_comments_origin()
data_comment = load_comments_sentiment()
neg_comment = load_neg_comment()
pos_comment = load_pos_comment()
tag_comments_pos = load_tag_comments_pos()
tag_comments_neg = load_tag_comments_neg()
LDA_coherence = load_LDA_coherence()

###
# 页面侧边栏的定义

st.sidebar.markdown('# 项目目录')
section = st.sidebar.radio("请选择需要展示的项目模块：", ('☆ 项目介绍与规划', '① 数据采集与抽取', '② 数据预处理与探索', '③ 自训练情感分析模型', '④ 评论分词与改进',
                                             '⑤ 词云与语义网络构建', '⑥ LDA主题模型构建', '⑦ 交互诊断与反馈'))

tag = None
if section in ['⑤ 词云与语义网络构建', '⑥ LDA主题模型构建', '⑦ 交互诊断与反馈']:
    st.sidebar.markdown('# 评论标签')
    tag = st.sidebar.selectbox("  请选择分析的评论标签：",
                               ('总体评论', '外形外观', '恒温效果', '噪音大小', '出水速度', '安装服务', '耗能情况', '加热速度', '洗浴时间', '其他特色'))

topic_number = 5
if section in ['⑥ LDA主题模型构建']:
    st.sidebar.markdown('# LDA主题数')
    topic_number = st.sidebar.slider('  请选择需要训练的LDA主题数:', 3, 9, 5)

###
# 项目介绍与规划显示

if section == '☆ 项目介绍与规划':
    st.markdown('## 0. 项目介绍与规划')

    st.markdown('***')

    st.markdown('### 0.1 开发人员清单与分工')
    st.markdown("""- dyf-2316：
    1. 数据爬取与数据抽取
    2. 数据预处理以及标签评论的处理
    3. Streamlit可视化和交互式呈现最终结果
    4. 将项目搭建在Heroku云服务器上
    5. 撰写会议记录
    6. 部署协调全组的工作
- hzkzTech：
    1. 对整体评论数据与分标签评论数据进行LDA主题分析模型建模及可视化
    2. 使用ROST对分标签评论数据进行语义网络分析
    3. 需求文档撰写及更新等
    4. 对主题分析模型调参优化
- msh：
    1. 对整体评论数据和分标签评论数据进行词云的绘制
    2. 完成交互诊断与反馈
    3. networkx绘制语义网络
    4. 撰写相关文档
- Joshua_yi: 
    1. 使用RoBERTa - wwm - ext预训练网络实现文本情感预测，对比ROST预测正确率提升25 %，并对不同的标签的评论进行情感分析
    2. 完成模型部分可视化
    3. 完成文本分词与改进
    4. 将项目搭建在Heroku云服务器上
    5. 完成项目展示报告，项目立项书等
    """)

    st.markdown('***')

    st.markdown('### 0.2 项目开发环境清单')
    environment_list = pd.DataFrame(
        {'名称': ['Windows10 & MacBook Pro', 'pycharm + google colab', 'python3.7', 'MongoDB', 'GitHub', 'Heroku'],
         '环境细节': ['项目开发硬件环境', '项目开发所用IDE与远程jupyter', 'python的版本', '数据存储的数据库', '项目版本管理', '项目展示平台']
         })
    st.table(environment_list)

    st.markdown('***')

    st.markdown('### 0.3 项目开发周期表')
    image = load_image('data/source/develop_schedule.png')
    st.image(image, use_column_width=True)

    st.markdown('***')

    st.markdown('### 0.4 项目开发流程图')
    image = load_image('data/source/flow_chart.png')
    st.image(image, use_column_width=True)

    st.markdown('***')

    st.markdown('### 0.5 项目实用技术')
    tech_list = pd.DataFrame(
        {'模块': ['数据爬取', '数据存储', '数据预处理', '隐含信息挖掘', '情感分析', '评论分词', '词云与语义网络', 'LDA主题模型', '模型评估与诊断', '可视化'],
         '技术细节': ['对url进行精确解析', '将数据存储在MongoDB中', '数据去重、机械压缩、短句去除',
                  '提取带标签的评论', '分词、去除停用词、词性过滤（只取动形名等实词）', '使用哈工大公开的预训练网络RoBERTa-wwm-ext进行文本情感预测',
                  '对不同标签的正负面评论分别使用词云和ROST进行剪枝语义网络分析', '使用pyLDAvis进行动态可视化，实现较好的交互性，并调节参数得到最优的主题提取',
                  '针对不同标签的评论定向分析产品不同方面的优劣势及其卖点', '使用streamlit将项目展示在网页上，并运行于Heroku云服务器上']
         })
    st.table(tech_list)

    st.markdown('***')

    st.markdown('### 0.6 遇到的问题以及解决方法')
    st.markdown(' - 多人使用git版本管理中遇到文件冲突的问题')
    st.markdown('解决办法：从网上查找资料，回退版本')
    st.markdown(' - 爬虫数据质量差，数据量少，差评比例小')
    st.markdown('解决办法：观察url格式遍历参数值来尽可能获取多的数据量。')
    st.markdown(' - 错误格式以及访问过快使得整个爬虫进程中断')
    st.markdown('解决办法：加入日记的记录，以及完善异常处理机制')
    st.markdown(' - 全面的提取标签数据，以及标签数据的存储')
    st.markdown('解决办法：观察字符串的规范，设计算法全面提取标签数据，考虑标签数据不等长，采用字典形式并json存储')
    st.markdown(' - 在使用RoBERTa-wwm-ext预训练模型进行情感预测中模型loss不收敛')
    st.markdown('解决办法：换用不同的优化器和超参数')
    st.markdown(' - 在tensorboard使用时频繁报错')
    st.markdown('解决办法：改为使用tensorboardX')
    st.markdown(' - 分词效果质量差，使得主题提取中高频关键词语义表达模糊')
    st.markdown('解决办法：在分词中考虑词性，根据保留不同的词性集所得到LAD结果质量，最终选择只保留动形名词')
    st.markdown(' - 生成词云图片的分辨率不高')
    st.markdown('解决办法：调整图片大小和词汇大小')
    st.markdown(' - 对于Negative的标签数据进行lda主题分析，出现模型运行错误')
    st.markdown('解决办法：Negative的标签数据量较少，减少主题数量，重新建模')
    st.markdown(' - 项目部署到远端服务器后无法对html文件在浏览器中访问')
    st.markdown('解决办法：将文件部署到github.io上可以直接使用url对资源访问')

    st.markdown('***')

###
# 数据采集与抽取部分展示

if section == '① 数据采集与抽取':
    st.markdown('## 1. 数据采集与抽取')

    st.markdown('***')

    st.markdown('### 1.1 数据爬取')
    st.markdown(' - 该项目在京东商城爬取美的热水器品牌原始数据，共计 {} 条'.format(len(data_origin)))
    st.markdown(' - 数据框（下方展示）中包含\n'
                '   - 产品型号(good_id)\n'
                '   - 品牌名称(brand)\n'
                '   - 价格(price)\n'
                '   - 评论(comment)\n'
                '   - 评论日期(creationDate)')
    if st.checkbox('Show origin data'):
        st.dataframe(data_origin, 900, 400)

    code_webcrawler = '''
    COMMENT_URL = "https://club.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98&productId={}&score={}&sortType=5&page={}&pageSize=10&isShadowSku=0&rid=0&fold=1 "

    def get_comment_data(product_id, score, page, product):
        """
        获取product_id相应score和page的评论数据，与商品基本信息一起构成数据字典返回
        :param product_id: (str) 商品ID
        :param score: (int) 商品评分
        :param page: (int) 评论页数
        :param product: (dict) 商品基本信息的数据字典
        :return: (dict) 商品信息及评论数据字典
        """
        comment_url = COMMENT_URL.format(product_id, score, page)
        mylogger.debug("获取商品评论页面URL -- {}".format(comment_url))

        json_data = get_json_data(comment_url)
        data = json_data['comments']
        comment_data = []
        for i in range(len(data)):
            comment = {
                "good_id": product['good_id'],
                "brand": product['brand'],
                "price": product['price'],
                "creationTime": data[i]['creationTime'],
                "score": data[i]['score'],
                "comment": data[i]['content']
            }
            comment_data.append(comment)
        mylogger.debug("获取商品评论数据 -- counts:{}".format(len(comment_data)))
        return comment_data
    '''
    st.code(code_webcrawler)

    st.markdown('***')

    st.markdown('### 1.2 数据存储（MongoDB）')
    code_save_mongo = '''
    def save_to_mongo(data):
        """
        将数据存入mongoDB
        :param data: (dict) 需要存入数据库的数据字典
        :return:
        """
        client = MongoClient(**DATABASE_MONGO)
        mongo_db = client['WebCrawler']
        mongo_table = mongo_db['comment']
        for item in data:
            mongo_table.collection.insert_one(item)
            mylogger.debug("存入数据 -- {}".format(item))
    '''
    st.code(code_save_mongo)

    st.markdown('***')

    st.markdown('### 1.3 评论抽取')
    st.markdown('从原始中抽取评论数据（下方展示）')
    if st.checkbox('Show origin comments'):
        data_origin = load_data_origin()
        comments_origin = pd.DataFrame(data_origin.comment)
        st.dataframe(comments_origin, 500, 400)

    st.markdown('***')

###
# 数据预处理与探索部分展示

if section == '② 数据预处理与探索':
    st.markdown('## 2. 数据预处理与探索')

    st.markdown('***')

    st.markdown('### 2.1 数据去重')
    st.markdown('评论数据进行去重后，共计 {} 条'.format(len(data_deldup)))
    if st.checkbox('Show delete duplicate comments'):
        comments_deldup = pd.DataFrame(data_deldup.comment)
        st.dataframe(comments_deldup, 500, 400)

    code_deldup = '''
    def del_duplicate(data):
        """
        对数据框评论栏去重
        :param data: (DataFrame)
        :return: (DataFrame)
        """
        result = data.drop_duplicates(['comment'], ignore_index=True)
        return result
    '''
    st.code(code_deldup)

    st.markdown('***')

    st.markdown('### 2.2 机械压缩与短句删除')
    st.markdown('评论数据进行机械压缩与短句删除后，共计 {} 条'.format(len(data_compress)))
    if st.checkbox('Show compressed comments'):
        comments_compress = pd.DataFrame(data_compress.comment)
        st.dataframe(comments_compress, 500, 400)
    code_compress = '''
    def compress(comment, n=4):
        """
        对文本进行正序、逆序机械压缩，短句删除的处理
        :param n: 所需删除短句的长度
        :param comment: (str)
        :return: (str)
        """
        comp_comm = []
        comment = str(comment)
        if len(comment) <= n:
            pass
        else:
            comment = re.sub("[\s,./?'\"|\]\[{}+_)(*&^%$#@!~`=\-]+|[+\-，。/！？、～@#¥%…&*（）—；：‘’“”]+", " ", comment)
            comp_list = list(comment)
            comp_list = compressed_text(comp_list)
            comp_list = compressed_text(comp_list[::-1])
            comp_list = comp_list[::-1]
            if len(comp_list) <= n:
                pass
            else:
                comp_comm = []
                comp_comm = (''.join(comp_list)).replace("\\n", "")
        return comp_comm
    '''
    st.code(code_compress)

    st.markdown('***')

    st.markdown('### 2.3 隐含信息挖掘 -- 标签评论的获取')
    st.markdown('观察评论数据，发现部分评论数据有固定的主题标签，将其爬取下来进行商品某方面特征定向分析。')
    if st.checkbox('Show tag comments'):
        st.json(tag_comment_origin)
    code_extract_tag = '''
    def extract_tag_comment(comment):
        """
        对评论数据进行标签评论抽取，若符合标签评论规范则返回标签评论字典，若不符合则返回None
        :param comment: (str)
        :return: (dict/None)
        """
        comment_lines = comment.split("\\n")
        tag_comment_dict = {}
        if len(comment) < 2:
            return None
        else:
            for lines in comment_lines:
                tag_comment = lines.split('：')
                if len(tag_comment) == 2:
                    if len(tag_comment[0]) < 6:
                        tag_comment_dict[tag_comment[0]] = tag_comment[1]
                else:
                    continue
        if tag_comment_dict:
            return tag_comment_dict
        else:
            return None
    '''
    st.code(code_extract_tag)

    st.markdown('***')

    st.markdown('### 2.4 数据特征探索')

    st.markdown('#### 2.4.1 美的热水器商品评论的评分分布图')
    score = pd.DataFrame(data_compress.score.value_counts())
    score.plot.pie(subplots=True)
    st.pyplot()

    st.markdown('#### 2.4.2 美的热水器商品评分走势图')
    date_score = pd.DataFrame(
        data_compress['score'].groupby(data_compress.creationDate.apply(lambda x: str(x)[:-3])).mean())
    st.line_chart(date_score)

    st.sidebar.markdown('# SparkClouds')
    kind = st.sidebar.selectbox('请选择要分析的评论评分高低', ['low score(评分1、2、3)', 'high score(评分4、5)'])

    st.markdown('#### 2.4.3 高低分评论词频时序图 (SparkClouds)')
    if kind == 'low score(评分1、2、3)':
        image = load_image('data/source/lowscore_keywords.png')
        st.image(image, use_column_width=True)
    if kind == 'high score(评分4、5)':
        image = load_image('data/source/highscore_keywords.png')
        st.image(image, use_column_width=True)
    st.markdown(
        '*参考文献：[SparkClouds: Visualizing Trends in Tag Clouds](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=5613457)*'
    )

    st.markdown('***')

###
# 自训练情感模型部分显示

if section == '③ 自训练情感分析模型':
    st.markdown('## 3. 自训练情感分析模型')

    st.markdown('***')

    st.markdown('### 3.1 模型神经网络结构图网络')

    b1 = st.checkbox('显示结构图')
    if b1:
        file_ = open('data/Sentiment_Results/result_loss.gif', "rb")
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")
        file_.close()

        st.markdown(
            f'<img src="data:image/gif;base64,{data_url}" alt="cat gif" style="height:500px;">',
            unsafe_allow_html=True,
        )

    st.markdown('***')

    st.markdown('### 3.2 模型训练Loss曲线')
    b2 = st.checkbox('显示Loss曲线')
    if b2:
        file_ = open('data/Sentiment_Results/resulit_struction.gif', "rb")
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")
        file_.close()

        st.markdown(
            f'<img src="data:image/gif;base64,{data_url}" alt="cat gif" style="height:500px;">',
            unsafe_allow_html=True,
        )

    st.markdown('***')

    st.markdown('### 3.3模型效果')
    st.markdown('#### 3.3.1 分类指标的对比')
    df = pd.DataFrame({'Roberta-wwm-ext': [0.9282735613010842, 0.9089481946624803, 0.9538714991762768],
                       'Bert-base-chinese': [0.87, 0.8284883720930233, 0.9375],
                       'snowNLP': [0.87, 0.8680781758957655, 0.8766447368421053],
                       'ROSTCM6': [0.675, 0.6319612590799032, 0.8585526315789473]},
                      index=['Accuracy', 'Precision', 'Recall']
                      )
    st.table(df)

    st.markdown('#### 3.3.2 Roc曲线图')
    image = load_image('data/source/Roc_graph.png')
    st.image(image, width=600)

    st.markdown('***')

    st.markdown('### 3.4 情感分析结果在评论评分中的分布')
    score_1 = data_comment[data_comment.score == 1].sentiment.value_counts()
    score_2 = data_comment[data_comment.score == 2].sentiment.value_counts()
    score_3 = data_comment[data_comment.score == 3].sentiment.value_counts()
    score_4 = data_comment[data_comment.score == 4].sentiment.value_counts()
    score_5 = data_comment[data_comment.score == 5].sentiment.value_counts()
    score = pd.DataFrame([score_1, score_2, score_3, score_4, score_5], index=[1, 2, 3, 4, 5])
    score = score.T
    score[1] = score[1] / (score.iloc[0, 0] + score.iloc[1, 0])
    score[2] = score[2] / (score.iloc[0, 1] + score.iloc[1, 1])
    score[3] = score[3] / (score.iloc[0, 2] + score.iloc[1, 2])
    score[4] = score[4] / (score.iloc[0, 3] + score.iloc[1, 3])
    score[5] = score[5] / (score.iloc[0, 4] + score.iloc[1, 4])
    score = score.T
    # score.plot.bar(stacked=True, color=['purple', 'skyblue'], alpha=0.5)
    # st.pyplot()
    score[0] = score[0] * -1
    st.bar_chart(score, width=100, use_container_width=True)

    st.markdown('***')

###
# 显示评论分词与改进

if section == '④ 评论分词与改进':
    st.markdown('## 4. 评论分词与改进')

    st.markdown('***')

    st.markdown('### 4.1 总体评论分词')
    if st.checkbox('展示总体积极评论分词'):
        st.dataframe(pos_comment)
    if st.checkbox('展示总体消极评论分词'):
        st.dataframe(neg_comment)

    st.markdown('***')

    st.markdown('### 4.2 标签评论分词')
    if st.checkbox('展示标签积极评论分词'):
        st.json(tag_comments_pos)
    if st.checkbox('展示标签消极评论分词'):
        st.json(tag_comments_neg)

    st.markdown('***')

    st.markdown('### 4.3 评论分词改进')
    code_cut_word = '''
    def str_cut(comment, stop_words):
        """
        切分字符串，并去除停用词，保留v, n, a, d, vd, an, ad，返回处理好的单词
        :param comment: (str) 文本评论
        :param stop_words: (list) 停用词
        :return: object_list (list) 处理好的词
        """
        seg_list = psg.cut(comment)
        object_list = []
        for word in seg_list:  # 循环读取每个分词
            # 获得需要的词性，去除停用词
            if word.word not in stop_words and (word.flag in ['v', 'n', 'a', 'd', 'vd', 'an', 'ad']): 
                object_list.append(word.word)  
        return object_list
    '''
    st.code(code_cut_word)

    st.markdown('***')

###
# 词云与语音网络构建部分的显示

if section == '⑤ 词云与语义网络构建':
    st.markdown('## 5. 词云与语义网络构建')

    st.markdown('***')

    st.markdown('### 5.1 美的热水器商品评论（{}）的词云'.format(tag))
    image = load_image('data/word_cloud/{}_pos.jpg'.format(tag))
    st.write('')
    st.image(image, caption='{} 积极情感词云'.format(tag), width=500)
    st.write('')
    image = load_image('data/word_cloud/{}_neg.jpg'.format(tag))
    st.image(image, caption='{} 消极情感词云'.format(tag), width=500)
    st.write('')

    st.markdown('***')

    st.markdown('### 5.2 美的热水器商品评论（{}）的语义网络'.format(tag))
    st.write('')
    image = load_image('data/word_net/{}_pos/net.jpg'.format(tag))
    st.image(image, caption='{} 积极情感语义网络'.format(tag), use_column_width=True)
    st.write('')
    image = load_image('data/word_net/{}_neg/net.jpg'.format(tag))
    st.image(image, caption='{} 消极情感语义网络'.format(tag), use_column_width=True)
    st.write('')

    st.markdown('***')

###
# LDA主题模型构建部分的展示

if section == '⑥ LDA主题模型构建':
    st.markdown('## 6. LDA主题模型构建')

    st.markdown('***')

    st.markdown('### 6.1 LDA关键字与主题提取\n')
    # if st.button('正面评论LDA结果展示'):
    #     login(u'lda_{}_{}_pos.html'.format(tag, topic_number))
    st.markdown(
        '### - [正面评论LDA结果展示](https://dyf-2316.github.io/LDA_Results/lda_{}_{}_pos.html)'.format(tag, topic_number))

    # if st.button('负面评论LDA结果展示'):
    #     login(u'lda_{}_{}_neg.html'.format(tag, topic_number))
    st.markdown(
        '### - [负面评论LDA结果展示](https://dyf-2316.github.io/LDA_Results/lda_{}_{}_neg.html)'.format(tag, topic_number))

    code_LDA = '''
    def LDA(data, components, htmlfile=None):
        """
        训练LDA模型，同时生成可视化文件
        :param data: (list) 文档列表
        :param components: (int) 指定主题数  
        :param htmlfile: (str) 可视化文件存储路径
        :return: None
        """
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
        pyLDAvis.save_html(result, 'data/LDA_Results/' + htmlfile)
        mylogger.debug('LDA模型可视化文件已生成')
    '''
    st.code(code_LDA)

    st.markdown('***')

    st.markdown('### 6.2 LDA模型优化')
    st.markdown('用 `cv_coherence` 来度量主题的连贯性，以此来选择出最优的 主题数 作为超参数')

    st.markdown('#### 6.2.1 {} 正面评论模型数优化'.format(tag))
    data_cv_pos = pd.DataFrame([LDA_coherence[tag]['pos']], index=['cv'])
    data_cv_pos = data_cv_pos.T
    data_cv_pos = (data_cv_pos['cv'] - data_cv_pos['cv'].min()) / (data_cv_pos['cv'].max() - data_cv_pos['cv'].min())
    st.line_chart(data_cv_pos)

    st.markdown('#### 6.2.1 {} 负面评论模型数优化'.format(tag))
    data_cv_neg = pd.DataFrame([LDA_coherence[tag]['neg']], index=['cv'])
    data_cv_neg = data_cv_neg.T
    data_cv_neg = (data_cv_neg['cv'] - data_cv_neg['cv'].min()) / (data_cv_neg['cv'].max() - data_cv_neg['cv'].min())
    st.line_chart(data_cv_neg)

    st.markdown('***')

###
# 交互诊断与反馈部分的展示

if section == '⑦ 交互诊断与反馈':
    st.markdown('## 7 交互诊断与反馈')

    st.markdown('***')

    if tag == '总体评论':
        st.markdown('### 总体评论')
        st.markdown('''
        > 优：排名靠前的主题词有：“不错”、“外观”、“服务”、“质量”。
    - 在此分析结果中，各topic之间的距离很近，甚至出现了重叠的情况，可以得出正面评论中主题词之间的相关性较高，多集中在产品外形外观好看，使用体验好，安装服务质量高这三方面。由此可以得出，消费者认为美的品牌值得信赖，热水器使用体验好，各方面性能较为出色。
  ''')
        st.markdown('''
        > 劣：排名前5的主题词是“安装费”、“热水器”、“配件”、“收费”和“客服”。
    - topic1、topic6和topic4都反映出了售后客服服务不好的现象，topic3、topic7和topic9反映出了服务费高和配件不完整的问题，topic2和topic8反映出了热水器存在漏水等质量问题。由此可以得出，京东商城上售卖的美的热水器存在少数故障设备和残次品，并且上门安装服务和售后服务整体水平有待提高。
        ''')

    if tag == '外形外观':
        st.markdown('### 外形外观')
        st.markdown('''
        > 优：在LDA主题分析结果中，“外观”、“好看”、“时尚”、“漂亮”、“大方”等主题词排名靠前
    - 各topic分布也较紧密，topic8离各topic集中分布的位置较远，其中的关键词是“简约”、“小巧”、“空间”等，说明一部分消费者更倾心于简单、节约空间的特点。另外，还有在多个topic中都有“科技感”一次出现，说明带有智能家居、可远程遥控的产品更受青睐。由此可以得出，美的热水器外观外形美观、时尚、大方、有科技感，带有触摸显示屏幕的热水器更能吸引消费者。
	    ''')
        st.markdown('''
        > 劣：在词云中，存在“磕碰”、“凹坑”、“老土”等关键词，但显示比例小
    - 说明存在运送过程中造成热水器外观磨损的情况，以及极少数消费者不喜欢热水器的外形外观。由此可以得出，物流过程还需要提升服务水平。
        ''')

    if tag == '恒温效果':
        st.markdown('### 恒温效果')
        st.markdown('''
            > 优：在LDA主题分析结果中，评论关键词多集中在“效果”、“稳定”、“不错”上
        - topic1、2和6反映了热水器恒温效果稳定的特点，topic4反映了热水器控制精准的特点。由此可以得出，热水器恒温效果好，没有忽冷忽热，能够做到全程恒温，并且水温调节便捷、精准。
    	''')
        st.markdown('''
            > 劣：劣：在语义网络图中，只有个别的线条指向“不行”、“垃圾”等词汇；在词云中，分布有占比很小的“凉水”、“冻死”等关键词。
        - 由此可以得出，热水器加温和恒温效果较好，出现问题的是极小部分。
        ''')

    if tag == '噪音大小':
        st.markdown('### 噪音大小')
        st.markdown('''
            > 优：在LDA主题分析结果中，出现频次较高的关键词有：“很小”、“接受”、“满意”、“静音”等
        - 各topic很紧密，重叠度较高，说明热水器噪音很小。由此得出，大部分设备噪音小，运行安静，在多数人可接受的范围
        ''')
        st.markdown('''
            > 劣：在词云和语义网络中，有个别负面的词汇，但是还出现了可以接受意义的关键词
        - 由此可见，少数设备可能声音有点大，但是总体处于人们可以接受的范围内。
        ''')

    if tag == '出水速度':
        st.markdown('### 出水速度')
        st.markdown('''
            > 优：在LDA主题分析结果中，排名考前的关键词有：“出水”、“速度”、“很快”、“热水”
        - topic1、2和3其中的关键词反映了热水器出水速度快；topic4、6、7和反映了热水器水压高、出水量大；topic5反映了热水器出水稳定；由此得出，热水器在出水方面的性能很好，出水速度快，供水量大。
        ''')
        st.markdown('''
            > 劣：在词云中，出现了“差评”、“瀑布”、“哗哗、“延迟”等低比例词汇
        - 由此推断出，可能存在少数设备有漏水或者出水慢的现象。
        ''')

    if tag == '安装服务':
        st.markdown('### 安装服务')
        st.markdown('''
            > 优：在LDA主题分析结果中，“专业”、“满意”、“态度”、“细心”等关键词出现较多
        - 各个主题的集中分布也体现了安装人员专业、耐心细心、热情负责的三大特点。由此可以分析出，大多数安装人员认真负责，服务态度好，安装速度快，赢得了消费者的赞许和认可。
        ''')
        st.markdown('''
            > 劣：在LDA主题分析结果中，“收费”、“安装费”、“配件”等关键词出现较多
        - topic1、2、3、4、5、6和8集中反映了产品配件不齐全，需要单独购买，安装费高的情况。topic7单独反映了安装人员态度恶劣的情况。由此得出，在安装过程中存在少数安装人员态度恶劣，乱收费的情况，另外缺少配件和安装费较贵也是需要改进的地方。
        ''')

    if tag == '耗能情况':
        st.markdown('### 耗能情况')
        st.markdown('''
            > 优：在LDA主题分析结果中，评论关键词集中在“节能”、“不错”上
        - topic中体现了有用电和燃气两种功能方式；在词云中，可以看出大部分产品属于一级能耗的标准，较为节能；在语义网络中，“满意”、“能耗”、“省电”三者之间的边很密集，说明相关度高。由此得出，无论是用电还是用燃气的热水器能耗都较低，绿色环保。
        ''')
        st.markdown('''
            > 劣：在LDA主题分析结果中，“暂时”、“没有”、“使用”和“知道”等关键词出现频次高
        - topic1中“耗电”这一关键词占比高。由此可以得出，多数用户对于能耗方面直观感受不强，少数用户认为有点费电。
        ''')

    if tag == '加热速度':
        st.markdown('### 加热速度')
        st.markdown('''
            > 优：在LDA主题结果分析中，排名前5的主题词是：“加热”、“很快”、“速度”、“挺快”、“非常”
        - topic3中体现了热水器加热功能使用方便的特点，其他topic则集中体现了加热快的特点。在词云中，出现了“功率”、“3000W”、“2100W”等词汇。由此可以推断出，大功率热水器加热速度很快、效果好，并且使用方便。
        ''')
        st.markdown('''
            > 劣：在词云中，出现了“太差”、“几十分钟”、“慢点”、“功率”等占比较低的负面词汇
        - 可以推断出，存在少部分设备加热速度慢的情况，可能是由于购买的热水器的功率较小，以及消费者主观方面的不同判断。
        ''')

    if tag == '洗浴时间':
        st.markdown('### 洗浴时间')
        st.markdown('''
            > 优：在LDA主题结果分析中，“够用”、“时间”、“足够”、“洗浴”等关键词出现次数较多
        - 所有的topic中都反映出了水量够用、满足多数家庭需要的特点。由此可以推出，多数产品水箱较大，可使用的时间长，能满足大多数家庭的需要。
        ''')
        st.markdown('''
            > 劣：在词云中，出现频次较多词汇多是如“一个”、“两三个”的数量词
        - 代表可以使用的人数，可以得出有少数热水器只能够一个人使用，多数热水器可以够两到三人使用，基本上可以满足多数家庭的需要
        ''')

    if tag == '其他特色':
        st.markdown('### 其他特色')
        st.markdown('''
            > 优：在LDA主题结果分析中，“智能”、“安全”、“物流”等词出现较多
        - topic2和4反映了热水器操作方便，可以推断带有智能家居属性的设备更受青睐。Topic3中“断电”、“安全”出现频次高，说明热水器有保护机制，安全性高。由此可以得到热水器的一些其他特点，如安全性高，物流速度快，另外还可以得出提升智能性可以更加吸引消费者。
        ''')
        st.markdown('''
            > 劣：在词云中，出现了“太贵”、“安装”、“客服”、“恶劣”等词汇
        - 这说明部分消费者认为热水器定价偏贵，有的售后客服服务质量较差。由此得出，热水器的售后服务水平参差不齐，需要进一步提升；价格因素受制于消费者的个人条件，并且评价比例很低，仅供参考，热水器的总体性价比还是较高的。
        ''')

    st.markdown('***')
