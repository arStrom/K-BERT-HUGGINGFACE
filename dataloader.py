
import torch
import numpy as np
from tqdm import tqdm
import copy
import json
from multiprocessing import Process, Pool
from torch.utils.data import Dataset

class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    def __init__(self, text_a, text_b=None, text_c=None, label=None, guid=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def read_json(input_file):
    """Reads a json list file."""
    with open(input_file, "r", encoding='UTF-8') as f:
        reader = f.readlines()
        return [json.loads(line.strip()) for line in reader]
    

def creat_label_sentences(path, class_list):
    """Creates examples for the training and dev sets."""
    label_number = len(class_list)
    sentences = []
    with open(path, mode='r', encoding="utf-8") as f:
        for (i, line) in enumerate(f):
            line = line.strip().split('\t')
            text_a = line[0]
            label = np.zeros((label_number,), dtype=int)
            for i in range(label_number):
                if class_list[i] in line:
                    label[i] = 1
            sentences.append(InputExample(text_a=text_a, label=label))
    return sentences
    

def creat_multi_label_sentences(path, class_list):
    """Creates examples for the training and dev sets."""
    label_number = len(class_list)
    sentences = []
    with open(path, mode='r', encoding="utf-8") as f:
        for (i, line) in enumerate(f):
            line = line.strip().split('\t')
            text_a = line[0]
            label = np.zeros((label_number,), dtype=int)
            for i in range(label_number):
                if class_list[i] in line:
                    label[i] = 1
            sentences.append(InputExample(text_a=text_a, label=label))
    return sentences


def creat_TNEWS(path, class_list):
    """Creates examples for the training and dev sets."""
    label_number = len(class_list)
    sentences = []
    lines = read_json(path)
    for (i, line) in enumerate(lines):
        sentence = line['sentence']
        keywords = line['keywords']
        label_name = line['label_desc']
        label = np.zeros((label_number,), dtype=int)
        for (x,class_name) in enumerate(class_list):
            if class_name == label_name:
                label[x] = 1
                break
        sentences.append(InputExample(text_a=sentence + keywords, label=label))
    return sentences


def creat_TNEWS_slice(path, class_list):
    """Creates examples for the training and dev sets."""
    label_number = len(class_list)
    sentences = []
    lines = read_json(path)
    for (i, line) in enumerate(lines):
        sentence = line['sentence']
        keywords = line['keywords']
        label_name = line['label_desc']
        label = np.zeros((label_number,), dtype=int)
        for (x,class_name) in enumerate(class_list):
            if class_name == label_name:
                label[x] = 1
                break 
        sentences.append(InputExample(text_a=sentence, text_b=keywords, label=label))
    return sentences

def creat_multi_label_sentences_slice(path, class_list):
    """Creates examples for the training and dev sets."""
    label_number = len(class_list)
    sentences = []
    with open(path, mode='r', encoding="utf-8") as f:
        for (i, line) in enumerate(f):
            if i == 0 :
                continue
            line = line.strip().split('\t')
            title = line[0]
            keyword = line[1]
            summary = line[2]
            label = np.zeros((label_number,), dtype=int)
            for i in range(label_number):
                if class_list[i] in line:
                    label[i] = 1
            sentences.append(InputExample(text_a=title, text_b=keyword, text_c=summary, label=label))
    return sentences



# 读取数据集
def read_dataset(path, tokenizer, workers_num=1, dataset=None, class_list=None, with_kg = True):

    read_dataset_process = {
        'tnews_public': creat_TNEWS,
        'tnews_public_slice': creat_TNEWS_slice,
        'book_multilabels_task': creat_multi_label_sentences,
        'book_multilabels_task_slice': creat_multi_label_sentences_slice,
    }

    print("Loading sentences from {}".format(path))
    sentences = read_dataset_process[dataset](path, class_list)
    encoder = tokenizer.encode_with_knowledge if with_kg else tokenizer.encode

    sentence_num = len(sentences)

    print("There are {} sentence in total. We use {} processes to inject knowledge into sentences.".format(sentence_num, workers_num))
    if workers_num > 1:
        params = []
        sentence_per_block = int(sentence_num / workers_num) + 1
        for i in range(workers_num):
            params.append((sentences[i*sentence_per_block: (i+1)*sentence_per_block], i))
        pool = Pool(workers_num)
        res = pool.map(encoder, *params)
        pool.close()
        pool.join()
        dataset = [sample for block in res for sample in block]
    else:
        params = (sentences, 0)
        dataset = encoder(*params)
    return dataset


# class myDataset(Dataset): #继承Dataset
#     def __init__(self, dataset): #__init__是初始化该类的一些基础参数
#         self.dataset = dataset   #数据集
    
#     def __len__(self):#返回整个数据集的大小
#         return len(self.dataset)
    
#     def __getitem__(self,index):#根据索引index返回dataset[index]
#         input_id = torch.LongTensor(self.dataset[index][0])
#         label_id = torch.FloatTensor(self.dataset[index][1])
#         mask_id = torch.LongTensor(self.dataset[index][2])
#         pos_id = torch.LongTensor(self.dataset[index][3])
#         vm = torch.LongTensor(self.dataset[index][4])
#         return input_id, mask_id, pos_id, vm, label_id  #返回该样本

class myDataset(Dataset): #继承Dataset
    def __init__(self, dataset): #__init__是初始化该类的一些基础参数
        self.dataset = dataset   #数据集
    
    def __len__(self):#返回整个数据集的大小
        return len(self.dataset)
    
    def __getitem__(self,index):#根据索引index返回dataset[index]
        input_ids = torch.LongTensor(self.dataset[index][0])
        mask_ids = torch.LongTensor(self.dataset[index][1])
        pos_ids = torch.LongTensor(self.dataset[index][2])
        vms = torch.LongTensor(np.array(self.dataset[index][3]))

        label_ids = torch.FloatTensor(self.dataset[index][4])
        return input_ids, mask_ids, pos_ids, vms, label_ids  #返回该样本

