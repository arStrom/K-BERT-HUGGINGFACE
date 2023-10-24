
import sys
from utils.constants import *
import numpy as np

class tokenizer:

    def __init__(self, vocab, max_length=256, kg=None):
        self.vocab = vocab
        self.kg = kg
        self.max_length = max_length

    # def encode(self, sentences, p_id=0):
    #     sentences_num = len(sentences)
    #     dataset = []
    #     columns = {"label":0, "text_a":1}
    #     for line_id, line in enumerate(sentences):
    #         if line_id % 10000 == 0:
    #             print("Progress of process {}: {}/{}".format(p_id, line_id, sentences_num))
    #             sys.stdout.flush()
    #         # line = line.strip().split('\t')
    #         try:
    #             label = line[columns["label"]]
    #             tokens = [CLS_TOKEN] + list(line[columns["text_a"]])
    #             tokens_lenth = len(tokens)
    #             token_ids = [self.vocab.get(t) for t in tokens]
    #             pos_ids = [i for i in range(0, tokens_lenth)]
    #             mask = [1 for i in range(0, tokens_lenth)]
    #             if tokens_lenth < self.max_length:
    #                 pad_num = self.max_length - tokens_lenth
    #                 token_ids += [self.vocab.get(PAD_TOKEN)] * pad_num
    #                 pos_ids += [self.max_length-1] * pad_num
    #                 mask += [0] * pad_num
    #             else:
    #                 token_ids = token_ids[:self.max_length]
    #                 pos_ids = pos_ids[:self.max_length]
    #                 mask = mask[:self.max_length]
    #             dataset.append((token_ids, label, mask, pos_ids, np.zeros(1)))
    #         except:
    #             print("Error line: ", line_id)
    #     return dataset


    # def encode_with_knowledge(self, sentences, p_id=0):
    #     sentences_num = len(sentences)
    #     dataset = []
    #     columns = {"label":0, "text_a":1}
    #     for line_id, line in enumerate(sentences):
    #         if line_id % 10000 == 0:
    #             print("Progress of process {}: {}/{}".format(p_id, line_id, sentences_num))
    #             sys.stdout.flush()
    #         # line = line.strip().split('\t')
    #         try:
    #             label = line[columns["label"]]
    #             text = CLS_TOKEN + line[columns["text_a"]]

    #             tokens, pos, vm, _ = self.kg.add_knowledge_with_vm([text], add_pad=True, max_length=self.max_length)
    #             tokens = tokens[0]
    #             pos = pos[0]
    #             vm = vm[0].astype("bool")

    #             token_ids = [self.vocab.get(t) for t in tokens]
    #             mask = [1 if t != PAD_TOKEN else 0 for t in tokens]

    #             dataset.append((token_ids, label, mask, pos, vm))
    #         except:
    #             print("Error line: ", line)
    #     return dataset


    def encode(self, sentences, p_id=0):
        sentences_num = len(sentences)
        dataset = []
        for line_id, inputexample in enumerate(sentences):
            if line_id % 10000 == 0:
                print("Progress of process {}: {}/{}".format(p_id, line_id, sentences_num))
                sys.stdout.flush()
            # line = line.strip().split('\t')
            example = [[],[],[],[]]
            try:
                sentences = [inputexample.text_a, inputexample.text_b, inputexample.text_c]
                for i,sentence in enumerate(sentences):
                    if sentence is None:
                        continue
                    tokens = [CLS_TOKEN] + list(sentence)
                    tokens_lenth = len(tokens)
                    token_ids = [self.vocab.get(t) for t in tokens]
                    pos_ids = [i for i in range(0, tokens_lenth)]
                    mask = [1 for i in range(0, tokens_lenth)]
                    if tokens_lenth < self.max_length:
                        pad_num = self.max_length - tokens_lenth
                        token_ids += [self.vocab.get(PAD_TOKEN)] * pad_num
                        pos_ids += [self.max_length-1] * pad_num
                        mask += [0] * pad_num
                    else:
                        token_ids = token_ids[:self.max_length]
                        pos_ids = pos_ids[:self.max_length]
                        mask = mask[:self.max_length]

                    example[0].append(token_ids)
                    example[1].append(mask)
                    example[2].append(pos_ids)
                    example[3].append(np.ones(1))

                example.append(inputexample.label)
                dataset.append(example)
            except:
                print("Error line: ", line_id)
        return dataset
    

    def encode_with_knowledge(self, sentences, p_id=0):
        sentences_num = len(sentences)
        dataset = []
        for line_id, inputexample in enumerate(sentences):
            if line_id % 10000 == 0:
                print("Progress of process {}: {}/{}".format(p_id, line_id, sentences_num))
                sys.stdout.flush()
            # line = line.strip().split('\t')
            example = [[],[],[],[]]
            try:
                sentences = [inputexample.text_a, inputexample.text_b, inputexample.text_c]
                for i,sentence in enumerate(sentences):
                    if sentence is None:
                        continue
                    text = CLS_TOKEN + sentence
                    tokens, pos, vm, _ = self.kg.add_knowledge_with_vm([text], add_pad=True, max_length=self.max_length)
                    tokens = tokens[0]
                    pos = pos[0]
                    vm = vm[0].astype("bool")
                    token_ids = [self.vocab.get(t) for t in tokens]
                    mask = [1 if t != PAD_TOKEN else 0 for t in tokens]

                    tokens = [CLS_TOKEN] + list(sentence)
                    tokens_lenth = len(tokens)
                    token_ids = [self.vocab.get(t) for t in tokens_lenth]
                    pos_ids = [i for i in range(0, tokens_lenth)]
                    mask = [1 for i in range(0, tokens_lenth)]
                    if tokens_lenth < self.max_length:
                        pad_num = self.max_length - tokens_lenth
                        token_ids += [self.vocab.get(PAD_TOKEN)] * pad_num
                        pos_ids += [self.max_length-1] * pad_num
                        mask += [0] * pad_num
                    else:
                        token_ids = token_ids[:self.max_length]
                        pos_ids = pos_ids[:self.max_length]
                        mask = mask[:self.max_length]

                    example[0].append(token_ids)
                    example[1].append(mask)
                    example[2].append(pos_ids)
                    example[3].append(np.ones(1))

                example.append(inputexample.label)
                dataset.append(example)

            except:
                print("Error line: ", line_id)
        return dataset