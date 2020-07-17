# Comment_Sentiment_Analysis
### Repository introduction
There are two most import branches in this Repository
+ master 
+ heroku

### master branch

The master branch contains all of the code and data in this project
+ config.py: some config setting in this project
+ decorator.py: some decorator function used 
+ flowchart.xmind: the flow chart of the whole project
+ Logger.py: the logging section
+ WebCrawler: the package get data and store them in MongoDB from JingDong
+ PreTreatment: the package use for data preprocess
+ SentimentClassification: the package use for sentiment predict,
 train the pretrained model RoBerta-wwm-ext and eval the text
    + data: directory contain train, dev, test data for train the RoBerta model
    + RoBerta-wwm-ext-Net: show the RoBerta-wwm-ext network structure with the tensorboard 
    + runs: some info recorded during training see with tensorbard
    + model_config.py: there are two pretrained models can be used **bert-base-chinese** and **RoBerta-wwm-ext**
   
+ Model: a directory to generate LDA, word clouds, spark clouds
    + msyh.ttc is a character style
+ Visualize: a directory use streamlit to edit the web page

the pretrained **bert-base-chinese** and **RoBerta-wwm-ext** models

and with trained **bert-base-chinese** and **RoBerta-wwm-ext** models  can be download in this url
(these trained models are not best models but not effect as a demo show) 1.42GB

url：[https://pan.baidu.com/s/1QFGB8xRQ1DX7Rf8la3R5ww](https://pan.baidu.com/s/1QFGB8xRQ1DX7Rf8la3R5ww)
pwd：fl9e

#### How can I use it?
download four models mentioned above 
and organize them

```
│  eval.py
│  makedata.py
│  model.py
│  model_train.py
│  model_config.py
│  __init__.py
│
├─bert
│  ├─bert-base-chinese
│  │      config.json
│  │      pytorch_model.bin
│  │      vocab.txt
│  │
│  ├─chinese_roberta_wwm_ext_pytorch
│  │      config.json
│  │      pytorch_model.bin
│  │      vocab.txt
│  │
│  └─model
│          model_0.93_1594606009.5589077.pth
│          model_1-bert-base-chinese.pth
│
├─data
│      dev.tsv
│      test.tsv
│      train.tsv
│
├─RoBerta-wwm-ext-Net
│      events.out.tfevents.1594626455.8c795c4e9e5c
│
├─runs
│  ├─.ipynb_checkpoints
│  ├─Jul13_15-37-57_62e081efbf10--train
│  │      events.out.tfevents.1594654677.62e081efbf10.123.0
│  │
│  └─Jul14_13-42-44_f8ddfa0b8e45--train
│          events.out.tfevents.1594734164.f8ddfa0b8e45.124.1

```
and run \__init\__.py

### heroku branch

this branch is use for deploy the app on Heroku
