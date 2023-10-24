# coding: UTF-8
import torch
import os
import json
from os.path import join
from tqdm import tqdm


# def get_class_list_json(file_path):
#     with open(file_path, 'r', encoding='UTF-8') as f:
#         return [json.loads(line.strip())['label_desc'] for line in tqdm(f) if line.strip()]

def get_class_list(file_path):

    if(file_path.split('.')[2] == 'json'):
        with open(file_path, 'r', encoding='UTF-8') as f:
            class_list = [json.loads(line.strip())['label_desc'] for line in tqdm(f) if line.strip()]
    elif(file_path.split('.')[2] == 'txt'):
        with open(file_path, 'r', encoding='UTF-8') as f:
            class_list = [line.strip('\n') for line in tqdm(f.readlines())]
    return class_list

def create_dir(dirs):
    if not os.path.isdir(dirs):
        os.makedirs(dirs)
    return dirs


def best_learning_rate(pretrained):
    return 2e-5

dataset_files = {
    'tnews_public_train': './datasets/tnews_public/train.json',
    'tnews_public_dev': './datasets/tnews_public/dev.json',
    'tnews_public_test': './datasets/tnews_public/test.json',
    'tnews_public_labels': './datasets/tnews_public/labels.json',
    'book_multilabels_task_slice_train': './datasets/book_multilabels_task_slice/train.tsv',
    'book_multilabels_task_slice_dev': './datasets/book_multilabels_task_slice/dev.tsv',
    'book_multilabels_task_slice_test': './datasets/book_multilabels_task_slice/test.tsv',
    'book_multilabels_task_slice_labels': './datasets/book_multilabels_task_slice/labels.txt',
    'book_multilabels_task_train': './datasets/book_multilabels_task/train.tsv',
    'book_multilabels_task_dev': './datasets/book_multilabels_task/dev.tsv',
    'book_multilabels_task_test': './datasets/book_multilabels_task/test.tsv',
    'book_multilabels_task_labels': './datasets/book_multilabels_task/labels.txt',
}

class BaseConfig(object):

    """配置参数"""

    def __init__(self, cuda, model_name, pretrained, dataset, sentence_num, No, seq_length, dropout,
                  epochs_num, batch_size, learning_rate, report_steps, pooling, no_kg, no_vm):

        # train
        cuda_available = torch.cuda.is_available()
        self.device = torch.device('cuda' if cuda_available and cuda else 'cpu')  # 使用cpu/gpu训练
        self.require_improvement = 2000                                           # 超过2000batch效果没提升，提前结束训练
        self.epochs_num = epochs_num                                              # epoch数
        self.batch_size = batch_size                                              # mini-batch大小
        self.learning_rate = learning_rate                                        # 学习率
        self.report_steps = report_steps
        self.warmup = 0.1
        self.sentence_num = sentence_num
        self.max_seq_length = seq_length                                          # 每句话处理成的长度(短填长切)
        self.pretrained_model_path = './models/' + pretrained                     # 预训练模型路径
        self.acc_percent = 0.9                                                    # 标签为真判定
        self.dropout = dropout
        self.pooling = pooling
        self.no_kg = no_kg
        self.no_vm = no_vm
        self.output_dir = create_dir('./outputs/' + str(No))                     # 结果保存路径

        self.seed = 7



        # dataset
        self.train_path = dataset_files[dataset + '_train']  # 训练集
        self.dev_path = dataset_files[dataset + '_dev']  # 验证集
        self.test_path = dataset_files[dataset + '_test']  # 测试集
        self.label_path = dataset_files[dataset + '_labels']  # 标签
        self.class_list = get_class_list(self.label_path)  # 标签列表
        self.label_number = len(self.class_list)  # 标签个数





