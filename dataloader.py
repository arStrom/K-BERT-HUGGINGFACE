
import torch
import random
from multiprocessing import Process, Pool
from torch.utils.data import Dataset

def read_dataset(path, tokenizer, workers_num=1, with_kg = True):

    print("Loading sentences from {}".format(path))
    sentences = []
    with open(path, mode='r', encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                continue
            sentences.append(line)
    sentence_num = len(sentences)

    print("There are {} sentence in total. We use {} processes to inject knowledge into sentences.".format(sentence_num, workers_num))
    if workers_num > 1:
        params = []
        sentence_per_block = int(sentence_num / workers_num) + 1
        for i in range(workers_num):
            params.append((sentences[i*sentence_per_block: (i+1)*sentence_per_block], i))
        pool = Pool(workers_num)
        res = pool.map(tokenizer.encode_with_knowledge, *params) if with_kg else pool.map(tokenizer.encode, *params)
        pool.close()
        pool.join()
        dataset = [sample for block in res for sample in block]
    else:
        params = (sentences, 0)
        dataset = tokenizer.encode_with_knowledge(*params) if with_kg else tokenizer.encode(*params)

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

