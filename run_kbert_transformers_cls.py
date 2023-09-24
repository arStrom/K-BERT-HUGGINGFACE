# -*- encoding:utf-8 -*-
"""
  This script provides an k-BERT exmaple for classification.
"""
import torch
import json
import random
import argparse
import collections
import torch.nn as nn
from config import BaseConfig
from transformers import BertConfig
from utils.vocab import Vocab
from utils.optimizers import  BertAdam
from utils.seed import set_seed
from brain import KnowledgeGraph
from multiprocessing import Process, Pool
from knowledgeable import add_knowledge_worker
import numpy as np
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

    # Path options.
    parser.add_argument("--train_path", type=str, required=True,
                        help="Path of the trainset.")
    parser.add_argument("--dev_path", type=str, required=True,
                        help="Path of the devset.") 
    parser.add_argument("--test_path", type=str, required=True,
                        help="Path of the testset.")

    # Model options.
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size.")
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
    parser.add_argument("--report_steps", type=int, default=100,
                        help="Specific steps to print prompt.")

    # Evaluation options.
    parser.add_argument("--mean_reciprocal_rank", action="store_true", help="Evaluation metrics for DBQA dataset.")

    # kg
    parser.add_argument("--kg_name", required=True, help="KG name or path")
    parser.add_argument("--workers_num", type=int, default=0, help="number of process for loading dataset")
    parser.add_argument("--no_vm", action="store_true", help="Disable the visible_matrix")

    args = parser.parse_args()

    model_name = args.model
    config = BaseConfig(args.cuda, model_name, args.pretrained, args.dropout, args.no_vm)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    model = model.to(device)

    # 构建知识图谱.
    if args.kg_name == 'none':
        spo_files = []
    else:
        spo_files = [args.kg_name]
    kg = KnowledgeGraph(spo_files=spo_files, predicate=True)

    


if __name__ == "__main__":
    main()
