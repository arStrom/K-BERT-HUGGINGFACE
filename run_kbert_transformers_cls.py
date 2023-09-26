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
from transformers import BertConfig
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
import MultiLabelSequenceClassification as MLCModels
import SingleLabelSequenceClassification as SLCModels


MLCModel = {
    'rbt3': MLCModels.BertForMultiLabelSequenceClassification,
    'bert': MLCModels.BertForMultiLabelSequenceClassification,
    'bert-rcnn': MLCModels.BertRCNNForMultiLabelSequenceClassification,
    'bert-cnn': MLCModels.BertCNNForMultiLabelSequenceClassification,
    'bert-rnn': MLCModels.BertRNNForMultiLabelSequenceClassification,
    'roberta-rcnn': MLCModels.BertRCNNForMultiLabelSequenceClassification,
    'roberta-cnn': MLCModels.BertCNNForMultiLabelSequenceClassification,
    'roberta-rnn': MLCModels.BertRNNForMultiLabelSequenceClassification,
    'ernie': MLCModels.ErnieForMultiLabelSequenceClassification,
    'ernie-rcnn': MLCModels.ErnieRCNNForMultiLabelSequenceClassification,
    'ernie-cnn': MLCModels.ErnieCNNForMultiLabelSequenceClassification,
    'ernie-rnn': MLCModels.ErnieRNNForMultiLabelSequenceClassification
}

SLCModel = {
    'bert': SLCModels.BertForSequenceClassification,
}

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model', default='bert', type=str, help='choose a model')
    parser.add_argument('--pretrained', default='google', type=str, help='choose a pretreined model')
    parser.add_argument('--cuda', action='store_true', help='True use GPU, False use CPU')

    # Model options.

    parser.add_argument("--seq_length", type=int, default=256,
                        help="Sequence length.")
    parser.add_argument("--encoder", choices=["bert", "lstm", "gru", \
                                                   "cnn", "gatedcnn", "attn", \
                                                   "rcnn", "crnn", "gpt", "bilstm"], \
                                                   default="bert", help="Encoder type.")
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
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout.")
    parser.add_argument("--epochs_num", type=int, default=5,
                        help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size.")
    parser.add_argument("--learning_rate", type=int, default=2e-5,
                        help="Sequence length.")
    parser.add_argument("--report_steps", type=int, default=100,
                        help="Specific steps to print prompt.")

    # kg
    parser.add_argument("--kg_name", required=True, help="KG name or path")
    parser.add_argument("--workers_num", type=int, default=1, help="number of process for loading dataset")
    parser.add_argument("--no_vm", action="store_true", help="Disable the visible_matrix")
    parser.add_argument("--no_kg", action="store_true", help="Disable the visible_matrix")

    args = parser.parse_args()

    model_name = args.model
    config = BaseConfig(args.cuda, model_name, args.pretrained, args.seq_length, args.dropout, 
                        args.epochs_num, args.batch_size, args.learning_rate, args.report_steps,
                        args.no_kg, args.no_vm)
    model_config = BertConfig.from_pretrained(
        config.pretrained_model_path + '/config.json',
        num_labels = config.label_number
    )

    # 设置随机种子
    set_seed(config.seed)

    args.labels_num = config.label_number 

    # 加载词汇表.
    vocab = Vocab()
    vocab.load(config.pretrained_model_path + '/vocab.txt')
    args.vocab = vocab

    # 加载分类模型
    model = SLCModel[model_name].from_pretrained(config.pretrained_model_path, config=model_config, args = args)

    # 使用DataParallel包装器来使用多个GPU
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
    print("device: ",config.device)
    model = model.to(config.device)

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
    tokenizer = Tokenizer(vocab, config.columns, config.max_seq_length, kg)

    train_dataset = dataloader.read_dataset(config.train_path, tokenizer, workers_num=args.workers_num, with_kg=not args.no_kg)
    train_dataset = dataloader.myDataset(train_dataset)
    train_batch = DataLoader(train_dataset,batch_size=config.batch_size)

    dev_dataset = dataloader.read_dataset(config.dev_path, tokenizer, workers_num=args.workers_num, with_kg=not args.no_kg)
    dev_dataset = dataloader.myDataset(dev_dataset)
    dev_batch = DataLoader(dev_dataset,batch_size=config.batch_size)

    test_dataset = dataloader.read_dataset(config.test_path, tokenizer, workers_num=args.workers_num, with_kg=not args.no_kg)
    test_dataset = dataloader.myDataset(test_dataset)
    test_batch = DataLoader(test_dataset,batch_size=config.batch_size)
    
    train(model, train_batch, dev_batch, test_batch, config=config)

    # Evaluation phase.
    print("Final evaluation on the test dataset.")

    if torch.cuda.device_count() > 1:
        model.module.load_state_dict(torch.load(config.output_dir + '/pytorch_model.bin'))
    else:
        model.load_state_dict(torch.load(config.output_dir + '/pytorch_model.bin'))
    evaluate(model, test_batch, config, is_test = True)


if __name__ == "__main__":
    main()
