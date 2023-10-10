import torch
import os
from os.path import join
from tqdm import tqdm
import copy
import numpy as np
import json

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

def get_class_list(file_path):
    # with open(file_path, 'r', encoding='UTF-8') as f:
    #     return [json.loads(line.strip())['label_desc'] for line in tqdm(f) if line.strip()]

    # multi_classification
    with open(file_path, 'r', encoding='UTF-8') as f:
        return [line.strip('\n') for line in tqdm(f.readlines())]

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

# label_path = './datasets/book_multilabels_task/labels.txt'
# train_path = './datasets/book_multilabels_task/train.tsv'
# class_list = get_class_list(label_path)
# sentences = creat_multi_label_sentences_slice(train_path,class_list)
# print(sentences)


x = torch.rand(5, 3)
y = torch.rand(5, 3)
print("x, y ",x, y)
#第一种
print("x+y ",x + y)
#第二种
print("add(x,y) ",torch.add(x, y))
#第三种
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print("result ", result)
#第四种
y.add_(x)
print("y.add ", y)
