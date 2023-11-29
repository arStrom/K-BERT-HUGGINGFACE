
import sys
from utils.constants import *
import numpy as np

class tokenizer:

    def __init__(self, vocab, max_length=256, kg=None):
        self.vocab = vocab
        self.kg = kg
        self.max_length = max_length

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
                    mask = [1 for i in range(0, tokens_lenth)]
                    pos_ids = [i for i in range(0, tokens_lenth)]
                    if tokens_lenth < self.max_length:
                        pad_num = self.max_length - tokens_lenth
                        token_ids += [self.vocab.get(PAD_TOKEN)] * pad_num
                        mask += [0] * pad_num
                        pos_ids += [self.max_length-1] * pad_num
                    else:
                        token_ids = token_ids[:self.max_length]
                        mask = mask[:self.max_length]
                        pos_ids = pos_ids[:self.max_length]

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
            # try:
            sentences = [inputexample.text_a, inputexample.text_b, inputexample.text_c]
            for i,sentence in enumerate(sentences):
                if sentence is None:
                    continue
                text = CLS_TOKEN + sentence
                tokens, pos_ids, vm, _ = self.kg.add_knowledge_with_vm([text], add_pad=True, max_length=self.max_length)
                tokens = tokens[0]
                token_ids = [self.vocab.get(t) for t in tokens]
                mask = [1 if t != PAD_TOKEN else 0 for t in tokens]
                pos_ids = pos_ids[0]
                vm = vm[0].astype("bool")

                example[0].append(token_ids)
                example[1].append(mask)
                example[2].append(pos_ids)
                example[3].append(vm)

            example.append(inputexample.label)
            dataset.append(example)

            # except Exception as ex:
            #     print("出现如下异常%s"%ex)
            #     print("Error line: ", line_id)
        return dataset
    
    def encode_add_knowledge(self, sentences, p_id=0):
        sentences_num = len(sentences)
        dataset = []
        for line_id, inputexample in enumerate(sentences):
            if line_id % 10000 == 0:
                print("Progress of process {}: {}/{}".format(p_id, line_id, sentences_num))
                sys.stdout.flush()
            # line = line.strip().split('\t')
            example = [[],[],[],[]]
            # try:
            sentences = [inputexample.text_a, inputexample.text_b, inputexample.text_c]
            entities = []
            for i,sentence in enumerate(sentences):
                if sentence is None:
                    sentence = ''
                entities += self.kg.add_knowledge([sentence], add_pad=True, max_length=self.max_length)
                tokens = [CLS_TOKEN] + list(sentence)
                tokens_lenth = len(tokens)
                token_ids = [self.vocab.get(t) for t in tokens]
                mask = [1 for i in range(0, tokens_lenth)]
                pos_ids = [i for i in range(0, tokens_lenth)]
                if tokens_lenth < self.max_length:
                    pad_num = self.max_length - tokens_lenth
                    token_ids += [self.vocab.get(PAD_TOKEN)] * pad_num
                    mask += [0] * pad_num
                    pos_ids += [self.max_length-1] * pad_num
                else:
                    token_ids = token_ids[:self.max_length]
                    mask = mask[:self.max_length]
                    pos_ids = pos_ids[:self.max_length]

                example[0].append(token_ids)
                example[1].append(mask)
                example[2].append(pos_ids)
                example[3].append(np.ones(1))

            entities_tokens = [CLS_TOKEN] + list("-".join(entities))
            entities_lenth = len(entities_tokens)
            entities_ids = [self.vocab.get(t) for t in entities_tokens]
            entities_mask = [1 for i in range(0, entities_lenth)]
            entities_pos_ids = [i for i in range(0, entities_lenth)]
            if entities_lenth < self.max_length:
                pad_num = self.max_length - entities_lenth
                entities_ids += [self.vocab.get(PAD_TOKEN)] * pad_num
                entities_mask += [0] * pad_num
                entities_pos_ids += [self.max_length-1] * pad_num
            else:
                entities_ids = entities_ids[:self.max_length]
                entities_mask = entities_mask[:self.max_length]
                entities_pos_ids = entities_pos_ids[:self.max_length]
            example[0].append(entities_ids)
            example[1].append(entities_mask)
            example[2].append(entities_pos_ids)
            example[3].append(np.ones(1))

            example.append(inputexample.label)
            dataset.append(example)

        return dataset