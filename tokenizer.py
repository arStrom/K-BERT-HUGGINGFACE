
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
        columns = {"label":0, "text_a":1}
        for line_id, line in enumerate(sentences):
            if line_id % 10000 == 0:
                print("Progress of process {}: {}/{}".format(p_id, line_id, sentences_num))
                sys.stdout.flush()
            line = line.strip().split('\t')
            try:
                label = int(line[columns["label"]])
                tokens = [CLS_TOKEN] + list(line[columns["text_a"]])
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
                vm = np.zeros((self.max_length,self.max_length))
                vm = vm[0].astype("bool")
                dataset.append((token_ids, label, mask, pos_ids, vm))
            except:
                print("Error line: ", line_id)
        return dataset


    def encode_with_knowledge(self, sentences, p_id=0):
        sentences_num = len(sentences)
        dataset = []
        columns = {"label":0, "text_a":1}
        for line_id, line in enumerate(sentences):
            if line_id % 10000 == 0:
                print("Progress of process {}: {}/{}".format(p_id, line_id, sentences_num))
                sys.stdout.flush()
            line = line.strip().split('\t')
            try:
                label = int(line[columns["label"]])
                text = CLS_TOKEN + line[columns["text_a"]]

                tokens, pos, vm, _ = self.kg.add_knowledge_with_vm([text], add_pad=True, max_length=self.max_length)
                tokens = tokens[0]
                pos = pos[0]
                vm = vm[0].astype("bool")

                token_ids = [self.vocab.get(t) for t in tokens]
                mask = [1 if t != PAD_TOKEN else 0 for t in tokens]

                dataset.append((token_ids, label, mask, pos, vm))
            except:
                print("Error line: ", line)
        return dataset


    def encode_slice(self, sentences, p_id=0):
        sentences_num = len(sentences)
        dataset = []
        for line_id, inputexample in enumerate(sentences):
            if line_id % 10000 == 0:
                print("Progress of process {}: {}/{}".format(p_id, line_id, sentences_num))
                sys.stdout.flush()
            # line = line.strip().split('\t')
            try:
                label = inputexample.label
                title_tokens = [CLS_TOKEN] + list(inputexample.title)
                keyword_tokens = [CLS_TOKEN] + list(inputexample.keyword)
                summary_tokens = [CLS_TOKEN] + list(inputexample.summary)

                title_tokens_lenth = len(title_tokens)
                keyword_tokens_lenth = len(keyword_tokens)
                summary_tokens_lenth = len(summary_tokens)

                title_token_ids = [self.vocab.get(t) for t in title_tokens]
                keyword_token_ids = [self.vocab.get(t) for t in keyword_tokens]
                summary_token_ids = [self.vocab.get(t) for t in summary_tokens]

                title_pos_ids = [i for i in range(0, title_tokens_lenth)]
                keyword_pos_ids = [i for i in range(0, keyword_tokens_lenth)]
                summary_pos_ids = [i for i in range(0, summary_tokens_lenth)]

                title_mask = [1 for i in range(0, title_tokens_lenth)]
                keyword_mask = [1 for i in range(0, keyword_tokens_lenth)]
                summary_mask = [1 for i in range(0, summary_tokens_lenth)]

                if title_tokens_lenth < self.max_length:
                    pad_num = self.max_length - title_tokens_lenth
                    title_token_ids += [self.vocab.get(PAD_TOKEN)] * pad_num
                    title_pos_ids += [self.max_length-1] * pad_num
                    title_mask += [0] * pad_num
                else:
                    title_token_ids = title_token_ids[:self.max_length]
                    title_pos_ids = title_pos_ids[:self.max_length]
                    title_mask = title_mask[:self.max_length]
                title_vm = np.zeros((self.max_length,self.max_length))
                title_vm = title_vm.astype("bool")

                if keyword_tokens_lenth < self.max_length:
                    pad_num = self.max_length - keyword_tokens_lenth
                    keyword_token_ids += [self.vocab.get(PAD_TOKEN)] * pad_num
                    keyword_pos_ids += [self.max_length-1] * pad_num
                    keyword_mask += [0] * pad_num
                else:
                    keyword_token_ids = keyword_token_ids[:self.max_length]
                    keyword_pos_ids = keyword_pos_ids[:self.max_length]
                    keyword_mask = keyword_mask[:self.max_length]
                keyword_vm = np.zeros((self.max_length,self.max_length))
                keyword_vm = keyword_vm.astype("bool")

                if summary_tokens_lenth < self.max_length:
                    pad_num = self.max_length - summary_tokens_lenth
                    summary_token_ids += [self.vocab.get(PAD_TOKEN)] * pad_num
                    summary_pos_ids += [self.max_length-1] * pad_num
                    summary_mask += [0] * pad_num
                else:
                    summary_token_ids = summary_token_ids[:self.max_length]
                    summary_pos_ids = summary_pos_ids[:self.max_length]
                    summary_mask = summary_mask[:self.max_length]
                summary_vm = np.zeros((self.max_length,self.max_length))
                summary_vm = summary_vm.astype("bool")


                dataset.append(((title_token_ids, title_mask, title_pos_ids, title_vm), 
                               (keyword_token_ids, keyword_mask, keyword_pos_ids, keyword_vm), 
                               (summary_token_ids, summary_mask, summary_pos_ids, summary_vm), 
                               label))
            except:
                print("Error line: ", line_id)
        return dataset
    

    def encode_with_knowledge_slice(self, sentences, p_id=0):
        sentences_num = len(sentences)
        dataset = []
        for line_id, inputexample in enumerate(sentences):
            if line_id % 10000 == 0:
                print("Progress of process {}: {}/{}".format(p_id, line_id, sentences_num))
                sys.stdout.flush()
            # line = line.strip().split('\t')
            # try:
            label = inputexample.label
            title_text = CLS_TOKEN + inputexample.title
            keyword_text = CLS_TOKEN + inputexample.keyword
            summary_text = CLS_TOKEN + inputexample.summary

            title_tokens, title_pos, title_vm, _ = self.kg.add_knowledge_with_vm([title_text], add_pad=True, max_length=self.max_length)
            
            title_tokens = title_tokens[0]
            title_pos = title_pos[0]
            title_vm = title_vm[0].astype("bool")

            title_token_ids = [self.vocab.get(t) for t in title_tokens]
            title_mask = [1 if t != PAD_TOKEN else 0 for t in title_tokens]
            
            keyword_tokens, keyword_pos, keyword_vm, _ = self.kg.add_knowledge_with_vm([keyword_text], add_pad=True, max_length=self.max_length)
            
            keyword_tokens = keyword_tokens[0]
            keyword_pos = keyword_pos[0]
            keyword_vm = keyword_vm[0].astype("bool")

            keyword_token_ids = [self.vocab.get(t) for t in keyword_tokens]
            keyword_mask = [1 if t != PAD_TOKEN else 0 for t in keyword_tokens]
            
            summary_tokens, summary_pos, summary_vm, _ = self.kg.add_knowledge_with_vm([summary_text], add_pad=True, max_length=self.max_length)
            
            summary_tokens = summary_tokens[0]
            summary_pos = summary_pos[0]
            summary_vm = summary_vm[0].astype("bool")

            summary_token_ids = [self.vocab.get(t) for t in summary_tokens]
            summary_mask = [1 if t != PAD_TOKEN else 0 for t in summary_tokens]

            dataset.append(((title_token_ids, title_mask, title_pos, title_vm), 
                            (keyword_token_ids, keyword_mask, title_pos, keyword_vm), 
                            (summary_token_ids, summary_mask, title_pos, summary_vm), 
                            label))
            # except:
            #     print("Error line: ", line_id)
        return dataset