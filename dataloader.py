
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

    def __init__(self, title, keyword=None, summary=None, label=None, guid=None):
        self.guid = guid
        self.title = title
        self.keyword = keyword
        self.summary = summary
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
            sentences.append((label,text_a))
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
            sentences.append((label,text_a))
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
            sentences.append(InputExample(title=title, keyword=keyword, summary=summary, label=label))
    return sentences

def read_dataset(path, tokenizer, workers_num=1, task = 'SLC', class_list=None, with_kg = True):

    print("Loading sentences from {}".format(path))
    if task == 'SLC':
        sentences = []
        with open(path, mode='r', encoding="utf-8") as f:
            for line_id, line in enumerate(f):
                if line_id == 0:
                    continue
                sentences.append(line)
        encoder = tokenizer.encode_with_knowledge if with_kg else tokenizer.encode
    elif task == 'MLC':
        sentences = creat_multi_label_sentences(path,class_list)
        encoder = tokenizer.encode_with_knowledge if with_kg else tokenizer.encode
    elif task == 'MLC-slice':
        sentences = creat_multi_label_sentences_slice(path,class_list)
        encoder = tokenizer.encode_with_knowledge_slice if with_kg else tokenizer.encode_slice

    sentence_num = len(sentences)

    print("There are {} sentence in total. We use {} processes to inject knowledge into sentences.".format(sentence_num, workers_num))
    if workers_num > 1:
        params = []
        sentence_per_block = int(sentence_num / workers_num) + 1
        for i in range(workers_num):
            params.append((sentences[i*sentence_per_block: (i+1)*sentence_per_block], i))
        pool = Pool(workers_num)
        # res = pool.map(tokenizer.encode_with_knowledge, *params) if with_kg else pool.map(tokenizer.encode, *params)
        res = pool.map(encoder, *params)
        pool.close()
        pool.join()
        dataset = [sample for block in res for sample in block]
    else:
        params = (sentences, 0)
        dataset = encoder(*params)
        # dataset = tokenizer.encode_with_knowledge(*params) if with_kg else tokenizer.encode(*params)

    return dataset

def read_dataset_slice(path, tokenizer, workers_num=1, task = 'SLC', class_list=None, with_kg = True):

    print("Loading sentences from {}".format(path))
    if task == 'SLC':
        sentences = []
        with open(path, mode='r', encoding="utf-8") as f:
            for line_id, line in enumerate(f):
                if line_id == 0:
                    continue
                sentences.append(line)
    elif task == 'MLC':
        sentences = creat_multi_label_sentences(path,class_list)
    elif task == 'MLC-slice':
        sentences = creat_multi_label_sentences_slice(path,class_list)

    sentence_num = len(sentences)

    print("There are {} sentence in total. We use {} processes to inject knowledge into sentences.".format(sentence_num, workers_num))
    if workers_num > 1:
        params = []
        sentence_per_block = int(sentence_num / workers_num) + 1
        for i in range(workers_num):
            params.append((sentences[i*sentence_per_block: (i+1)*sentence_per_block], i))
        pool = Pool(workers_num)
        res = pool.map(tokenizer.encode_with_knowledge_slice, *params) if with_kg else pool.map(tokenizer.encode_slice, *params)
        pool.close()
        pool.join()
        dataset = [sample for block in res for sample in block]
    else:
        params = (sentences, 0)
        dataset = tokenizer.encode_with_knowledge_slice(*params) if with_kg else tokenizer.encode_slice(*params)

    return dataset

class myDataset(Dataset): #继承Dataset
    def __init__(self, dataset): #__init__是初始化该类的一些基础参数
        self.dataset = dataset   #数据集
    
    def __len__(self):#返回整个数据集的大小
        return len(self.dataset)
    
    def __getitem__(self,index):#根据索引index返回dataset[index]
        input_id = torch.LongTensor(self.dataset[index][0])
        label_id = torch.LongTensor([self.dataset[index][1]])
        mask_id = torch.LongTensor(self.dataset[index][2])
        pos_id = torch.LongTensor(self.dataset[index][3])
        vm = torch.LongTensor(self.dataset[index][4])
        return input_id, label_id, mask_id, pos_id, vm  #返回该样本


class myDataset_slice(Dataset): #继承Dataset
    def __init__(self, dataset): #__init__是初始化该类的一些基础参数
        self.dataset = dataset   #数据集
    
    def __len__(self):#返回整个数据集的大小
        return len(self.dataset)
    
    def __getitem__(self,index):#根据索引index返回dataset[index]
        # dic_keys = ['title','keyword','summary']
        # date_dic = {}
        # input_ids = []
        # mask_ids = []
        # pos_ids = []
        # vms = []
        # for i in range(3):
        #     input_id = torch.LongTensor(self.dataset[index][i][0])
        #     mask_id = torch.LongTensor(self.dataset[index][i][1])
        #     pos_id = torch.LongTensor(self.dataset[index][i][2])
        #     vm = torch.LongTensor(self.dataset[index][i][3])

        #     input_ids.append(input_id)
        #     mask_ids.append(mask_id)
        #     pos_ids.append(pos_id)
        #     vms.append(vm)
        
        # input_ids = torch.stack([torch.LongTensor(self.dataset[index][i][0]) for i in range(3)], 0)
        # mask_ids = torch.stack([torch.LongTensor(self.dataset[index][i][1]) for i in range(3)], 0)
        # pos_ids = torch.stack([torch.LongTensor(self.dataset[index][i][2]) for i in range(3)], 0)
        # vms = torch.stack([torch.LongTensor(self.dataset[index][i][3]) for i in range(3)], 0)

        input_ids = torch.LongTensor(self.dataset[index][0])
        mask_ids = torch.LongTensor(self.dataset[index][1])
        pos_ids = torch.LongTensor(self.dataset[index][2])
        vms = torch.LongTensor(np.array(self.dataset[index][3]))

        label_id = torch.FloatTensor(self.dataset[index][4])
        return input_ids, mask_ids, pos_ids, vms, label_id  #返回该样本

