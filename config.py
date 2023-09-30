# coding: UTF-8
import torch
import os
from os.path import join
from tqdm import tqdm


def get_class_list(file_path):
    # with open(file_path, 'r', encoding='UTF-8') as f:
    #     return [json.loads(line.strip())['label_desc'] for line in tqdm(f) if line.strip()]

    # multi_classification
    with open(file_path, 'r', encoding='UTF-8') as f:
        return [line.strip('\n') for line in tqdm(f.readlines())]

def create_dir(dirs):
    if not os.path.isdir(dirs):
        os.makedirs(dirs)
    return dirs


def best_learning_rate(pretrained):
    return 2e-5


class BaseConfig(object):

    """配置参数"""

    def __init__(self, cuda, model_name, pretrained, seq_length, dropout,
                  epochs_num, batch_size, learning_rate, report_steps, no_kg, no_vm):

        # train
        cuda_available = torch.cuda.is_available()
        self.device = torch.device('cuda' if cuda_available and cuda else 'cpu')  # 使用cpu/gpu训练
        self.require_improvement = 1000                                           # 超过2000batch效果没提升，提前结束训练
        self.epochs_num = epochs_num                                              # epoch数
        self.batch_size = batch_size                                                      # mini-batch大小
        self.learning_rate = learning_rate                                        # 学习率
        self.report_steps = report_steps
        self.warmup = 0.1
        self.max_seq_length = seq_length                                                 # 每句话处理成的长度(短填长切)
        self.pretrained_model_path = './models/' + pretrained                     # 预训练模型路径
        self.acc_percent = 0.8                                                    # 标签为真判定
        self.dropout = dropout
        self.no_kg = no_kg
        self.no_vm = no_vm
        self.output_dir = create_dir('./outputs/'
                                     + model_name
                                     + "_num_epochs="
                                     + str(self.epochs_num)
                                     + "_batch_size="
                                     + str(self.batch_size)
                                     + "_learning_rate="
                                     + str(self.learning_rate)
                                     + "_max_seq_length="
                                     + str(self.max_seq_length)
                                     + "_no_kg="
                                     + str(self.no_kg)
                                     + "_no_vm="
                                     + str(self.no_vm))                     # 结果保存路径
        # dataset
        # self.data_dir = 'data/dataset'
        # self.train_path = join(self.data_dir, 'train.json')                       # 训练集
        # self.dev_path = join(self.data_dir, 'dev.json')                           # 验证集
        # self.test_path = join(self.data_dir, 'test.json')                         # 测试集
        # self.label_path = join(self.data_dir, 'labels.json')                      # 标签
        # self.class_list = get_class_list(self.label_path)                         # 标签列表
        # self.label_number = len(self.class_list)                                  # 标签个数
        self.seed = 7

        # dataset
        self.data_dir = './datasets/book_multilabels_task'
        self.train_path = join(self.data_dir, 'train.tsv')  # 训练集
        self.dev_path = join(self.data_dir, 'dev.tsv')  # 验证集
        self.test_path = join(self.data_dir, 'test.tsv')  # 测试集
        self.label_path = join(self.data_dir, 'labels.txt')  # 标签
        self.class_list = get_class_list(self.label_path)  # 标签列表
        self.label_number = len(self.class_list)  # 标签个数





