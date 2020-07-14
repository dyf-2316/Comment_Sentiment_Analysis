# -*- coding:utf-8 -*-
# @Time： 2020-07-12 15:47
# @Author: Joshua_yi
# @FileName: modelTrain.py
# @Software: PyCharm
# @Project: Comment_Sentiment_Analysis
# @Description:
# main.py

from transformers import BertForSequenceClassification
import torch
import time
from SentimentClassification import model_config
from SentimentClassification.makedataFile import data_loader
from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter

# 之后需要加tensorboard
# writer = SummaryWriter(comment='train')
# 使用gpu还是cpu
# 检测当前环境设备
device = ['cpu', 'gpu'][torch.cuda.is_available()]
# 选择是否使用gpu
use_gpu = (device == 'gpu') and model_config.USE_GPU


class model_train(object):
    def __init__(self, epochs=model_config.EPOCH, bert_model_path=model_config.BERT_MODEL):
        print('start train ... ')
        self.epochs = epochs
        print('load pretrained model ...')
        self.model = BertForSequenceClassification.from_pretrained(bert_model_path, num_labels=model_config.NUM_LABEL)
        if use_gpu: self.model.cuda()
        print('load data ...')
        trainset = data_loader(model_config.DATA_PATH, data_type='train')
        self.trainloader = DataLoader(trainset, batch_size=model_config.BATCH_SIZE, shuffle=True,
                                 num_workers=model_config.NUM_WORKERS)

        testset = data_loader(model_config.DATA_PATH, data_type='test')
        self.testloader = DataLoader(testset, batch_size=model_config.BATCH_SIZE, shuffle=False,
                                num_workers=model_config.NUM_WORKERS)
        self.acc = []
        pass

    def train_one_epoch(self, epoch):
        print(f'train--Epoch: {epoch + 1}')
        optimizer = torch.optim.Adam(self.model.parameters(), lr=model_config.LR)
        self.model.train()
        start_time = time.time()
        print_step = 10
        for batch_idx, (sue, label, posi) in enumerate(self.trainloader):
            if use_gpu:
                sue = sue.cuda()
                posi = posi.cuda()
                label = label.unsqueeze(1).cuda()

            optimizer.zero_grad()
            # 输入参数为词列表、位置列表、标签
            outputs = self.model(sue, position_ids=posi, labels=label)
            loss, logits = outputs[0], outputs[1]
            loss.backward()
            optimizer.step()
            if batch_idx % print_step == 0:
                print("epoch:[%d|%d] [%d|%d] loss:%f" % (epoch + 1, self.epochs, batch_idx, len(self.trainloader), loss.mean()))
                # writer.add_scalar(tag='loss', scalar_value=loss.mean(), global_step=(epoch * len(self.trainloader)//print_step) + batch_idx)
        print(f'train epoch{epoch} ' + "time:%.3f" % (time.time() - start_time))
        pass

    def test_one_epoch(self, epoch):
        print(f'test-Epoch: {epoch + 1}')
        start_time = time.time()
        self.model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for batch_idx, (sue, label, posi) in enumerate(self.testloader):
                if use_gpu:
                    sue = sue.cuda()
                    posi = posi.cuda()
                    labels = label.unsqueeze(1).cuda()
                    label = label.cuda()
                else:
                    labels = label.unsqueeze(1)

                outputs = self.model(sue, labels=labels)
                loss, logits = outputs[:2]
                _, predicted = torch.max(logits.data, 1)

                total += sue.size(0)
                correct += predicted.data.eq(label.data).cpu().sum()
        print(f'test epoch{epoch} ' + "time:%.3f" % (time.time() - start_time))
        acc = (1.0 * correct.numpy()) / total

        if acc > max(self.acc):
            self.save_model('model')
        self.acc.append(acc)

        print("Acc:%.3f" % (acc))
        # writer.add_scalar(tag='acc', scalar_value=acc, global_step=epoch)
        pass

    def train_epochs(self):
        """
        多次迭代训练
        :return:
        """
        print(f'begin train ... {self.epochs}')
        for epoch in range(self.epochs):
            self.train_one_epoch(epoch)
            self.test_one_epoch(epoch)
        pass

    def save_model(self, model_path):
        """
        保存模型
        :param model_path:
        :return:
        """
        # 保存最新的模型
        model_file = model_path + f'/model_{self.acc[-1]}_{time.strftime("%Y%m%d-%X")}.pth'
        print('-' * 100)
        print(f'model save to {model_file}')
        print('-' * 100)
        torch.save(self.model.state_dict(), model_file)

if __name__ == '__main__':
    train = model_train(epochs=1)
    train.train_epochs()
    train.save_model('./bert/model')
