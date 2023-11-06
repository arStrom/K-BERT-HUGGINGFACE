# -*- encoding:utf-8 -*-
"""
  This script provides an k-BERT exmaple for classification.
"""
import torch
import json
import argparse
import collections
import torch.nn as nn
from config import BaseConfig
from transformers import BertConfig, ErnieConfig
from utils.vocab import Vocab
from utils.seed import set_seed
from brain import KnowledgeGraph
from multiprocessing import Process, Pool
import dataloader
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tokenizer import tokenizer as Tokenizer
import numpy as np
from train import train
from evaluate import evaluate
from test import test
import MultiLabelSequenceClassification as MLCModels
import SingleLabelSequenceClassification as SLCModels
import MultiLabelSequenceClassificationSlice as MLCSliceModels


MLCModel = {
    'ernie-rcnn-new': MLCModels.ErnieRCNNForMultiLabelSequenceClassificationNew,
    'bert': MLCModels.BertForMultiLabelSequenceClassification,
    'bert-rcnn': MLCModels.BertRCNNForMultiLabelSequenceClassification,
    'bert-cnn': MLCModels.BertCNNForMultiLabelSequenceClassification,
    'bert-rnn': MLCModels.BertRNNForMultiLabelSequenceClassification,
    'ernie': MLCModels.ErnieForMultiLabelSequenceClassification,
    'ernie-rcnn': MLCModels.ErnieRCNNForMultiLabelSequenceClassification,
    'ernie-rcnn-catmaxpool': MLCModels.ErnieRCNNForMultiLabelSequenceClassificationCatMaxPool,
    'ernie-rcnn-catlstm':MLCModels.ErnieRCNNForMultiLabelSequenceClassificationCatLSTM,
    'ernie-rcnn-catlstmwide':MLCModels.ErnieRCNNForMultiLabelSequenceClassificationCatLSTMWide,
    'ernie-cnn': MLCModels.ErnieCNNForMultiLabelSequenceClassification,
    'ernie-rnn': MLCModels.ErnieRNNForMultiLabelSequenceClassification
}

SLCModel = {
    'bert': SLCModels.BertForSequenceClassification,
    'ernie-rcnn': SLCModels.ErnieRCNNForSequenceClassification,
}


sentence_num = {
    'tnews_public': 1,
    'tnews_public_slice': 2,
    'book_multilabels_task': 1,
    'book_multilabels_task_slice': 3,
}

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model', default='bert', type=str, help='choose a model')
    parser.add_argument('--pretrained', default='google', type=str, help='choose a pretreined model')
    parser.add_argument('--cuda', action='store_true', help='True use GPU, False use CPU')
    parser.add_argument('--task', default='SLC', help='task type')
    parser.add_argument('--dataset', default='book_review', help='dataset name')

    # Model options.
    parser.add_argument("--No", type=int, required=True, help="Experiment number.")
    parser.add_argument("--seq_length", type=int, default=256,
                        help="Sequence length.")
    parser.add_argument("--bidirectional", action="store_true", help="Specific to recurrent model.")
    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")

    # Subword options.
    parser.add_argument("--subword_type", choices=["none", "char"], default="none",
                        help="Subword feature type.")
    parser.add_argument("--sub_vocab_path", type=str, default="models/sub_vocab.txt",
                        help="Path of the subword vocabulary file.")
    parser.add_argument("--subencoder", choices=["avg", "lstm", "gru", "cnn"], default="avg",
                        help="Subencoder type.")
    parser.add_argument("--sub_layers_num", type=int, default=2, help="The number of subencoder layers.")

    # Tokenizer options.
    parser.add_argument("--tokenizer", choices=["bert", "char", "word", "space"], default="bert",
                        help="Specify the tokenizer." 
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Word tokenizer supports online word segmentation based on jieba segmentor."
                             "Space tokenizer segments sentences into words according to space."
                        )

    # Training options.
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout.")
    parser.add_argument("--epochs_num", type=int, default=5,
                        help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size.")
    parser.add_argument("--pretrained_learning_rate", type=float, default=2e-5,
                        help="pretrained_learning_rate.")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="learning_rate.")
    parser.add_argument("--report_steps", type=int, default=100,
                        help="Specific steps to print prompt.")

    # kg
    parser.add_argument("--kg_name", required=True, help="KG name or path")
    parser.add_argument("--workers_num", type=int, default=1, help="number of process for loading dataset")
    parser.add_argument("--no_vm", action="store_true", help="Disable the visible_matrix")
    parser.add_argument("--no_kg", action="store_true", help="Disable the visible_matrix")

    args = parser.parse_args()

    # 如果不启用kg，那么vm也不启用
    # 如果启用kg，vm根据用户自定义
    args.no_vm = args.no_kg if args.no_kg else args.no_vm
    model_name = args.model
    base_config = BaseConfig(args.cuda, model_name, args.pretrained, args.dataset, sentence_num[args.dataset], args.No, args.seq_length, args.dropout, 
                        args.epochs_num, args.batch_size, args.pretrained_learning_rate, args.learning_rate, args.report_steps, args.pooling,
                        args.no_kg, args.no_vm)

    model_type = model_name.split('-')
    if model_type[0] == 'bert':
        model_config = BertConfig.from_pretrained(
            base_config.pretrained_model_path + '/config.json',
            num_labels = base_config.label_number
        )
    else:
        model_config = ErnieConfig.from_pretrained(
            base_config.pretrained_model_path + '/config.json',
            num_labels = base_config.label_number
        )

    # 设置随机种子
    set_seed(base_config.seed)

    args.labels_num = base_config.label_number 

    # 加载词汇表.
    vocab = Vocab()
    vocab.load(base_config.pretrained_model_path + '/vocab.txt')
    args.vocab = vocab

    args.sentence_num = sentence_num[args.dataset]
    # 加载分类模型
    if args.task == 'SLC':
        model = SLCModel[model_name].from_pretrained(base_config.pretrained_model_path, config=model_config, base_config = base_config)
    elif args.task == 'MLC':
        model = MLCModel[model_name].from_pretrained(base_config.pretrained_model_path, config=model_config, base_config = base_config)
    else:
        raise NameError("任务名称错误")
    
    print("model: ",args.model)
    print("pretrained: ",args.pretrained)
    print("dataset: ",args.dataset)
    print("task: ",args.task)

    print("seq_length: ",args.seq_length)
    print("hidden_dropout_prob: ",model_config.hidden_dropout_prob)
    print("attention_probs_dropout_prob: ",model_config.attention_probs_dropout_prob)
    print("dropout_rnn: ",model.dropout_rnn if hasattr(model, 'dropout_rnn') else 'None')
    print("rnn_hidden: ",model.rnn_hidden if hasattr(model, 'rnn_hidden') else 'None')
    print("rnn_layers: ",model.num_layers if hasattr(model, 'num_layers') else 'None')
    print("epochs_num: ",args.epochs_num)
    print("batch_size: ",args.batch_size)
    print("pretrained_learning_rate: ",args.pretrained_learning_rate)
    print("learning_rate: ",args.learning_rate)
    print("report_steps: ",args.report_steps)
    print("acc_percent: ",base_config.acc_percent)
    print("kg_name: ",args.kg_name)
    print("no_kg: ",args.no_kg)
    print("no_vm: ",args.no_vm)
    print("GPU: ",torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    
    # 使用DataParallel包装器来使用多个GPU
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
    print("device: ",base_config.device)
    model = model.to(base_config.device)

    # 构建知识图谱.
    if args.kg_name == 'none':
        spo_files = []
    else:
        spo_files = [args.kg_name]
    
    if not args.no_kg:
        kg = KnowledgeGraph(spo_files=spo_files, predicate=True)
    else:
        kg = None
    
    #训练.
    print("Start training.")

    tokenizer = Tokenizer(vocab, base_config.max_seq_length, kg)

    # 训练数据
    train_dataset = dataloader.read_dataset(base_config.train_path, tokenizer, 
                                            workers_num=args.workers_num, dataset=args.dataset, 
                                            class_list=base_config.class_list, with_kg=not args.no_kg)
    train_dataset = dataloader.myDataset(train_dataset)
    train_batch = DataLoader(train_dataset,batch_size=base_config.batch_size, shuffle=True)

    # 验证数据
    dev_dataset = dataloader.read_dataset(base_config.dev_path, tokenizer, 
                                          workers_num=args.workers_num, dataset=args.dataset, 
                                          class_list=base_config.class_list, with_kg=not args.no_kg)
    dev_dataset = dataloader.myDataset(dev_dataset)
    dev_batch = DataLoader(dev_dataset,batch_size=base_config.batch_size)

    # 测试数据
    test_dataset = dataloader.read_dataset(base_config.test_path, tokenizer, 
                                           workers_num=args.workers_num, dataset=args.dataset, 
                                           class_list=base_config.class_list, with_kg=not args.no_kg)
    test_dataset = dataloader.myDataset(test_dataset)
    test_batch = DataLoader(test_dataset,batch_size=base_config.batch_size)

    # evaluate(model, dev_batch, base_config, is_test = False)

    train(model, train_batch, dev_batch, test_batch, config=base_config, task=args.task)

    # Evaluation phase.
    print("Final evaluation on the test dataset.")
    test(model,test_batch,base_config,args.task)

if __name__ == "__main__":
    main()
