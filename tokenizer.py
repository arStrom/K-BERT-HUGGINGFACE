
import sys
from utils.constants import *

class tokenizer:

    def __init__(self, vocab, sentences, columns, kg=None):
        self.vocab = vocab
        self.sentences = sentences
        self.columns = columns
        self.kg = kg


    def encode(self,p_id):
        sentences_num = len(self.sentences)
        dataset = []
        for line_id, line in enumerate(self.sentences):
            if line_id % 10000 == 0:
                print("Progress of process {}: {}/{}".format(p_id, line_id, sentences_num))
                sys.stdout.flush()
            line = line.strip().split('\t')
            try:
                if len(line) == 2:
                    label = int(line[self.columns["label"]])
                    text = CLS_TOKEN + list(line[self.columns["text_a"]])



    
                    tokens, pos, vm, _ = self.kg.add_knowledge_with_vm([text], add_pad=True, max_length=args.seq_length)
                    tokens = tokens[0]
                    pos = pos[0]
                    vm = vm[0].astype("bool")

                    token_ids = [self.vocab.get(t) for t in tokens]
                    mask = [1 if t != PAD_TOKEN else 0 for t in tokens]

                    dataset.append((token_ids, label, mask, pos, vm))
                
                elif len(line) == 3:
                    label = int(line[self.columns["label"]])
                    text = CLS_TOKEN + line[self.columns["text_a"]] + SEP_TOKEN + line[self.columns["text_b"]] + SEP_TOKEN

                    tokens, pos, vm, _ = kg.add_knowledge_with_vm([text], add_pad=True, max_length=args.seq_length)
                    tokens = tokens[0]
                    pos = pos[0]
                    vm = vm[0].astype("bool")

                    token_ids = [self.vocab.get(t) for t in tokens]
                    mask = []
                    seg_tag = 1
                    for t in tokens:
                        if t == PAD_TOKEN:
                            mask.append(0)
                        else:
                            mask.append(seg_tag)
                        if t == SEP_TOKEN:
                            seg_tag += 1

                    dataset.append((token_ids, label, mask, pos, vm))
                else:
                    pass
            except:
                print("Error line: ", line)
        return dataset
        pass


    def encode_with_knowledge(self, p_id=0):
        sentences_num = len(self.sentences)
        dataset = []
        for line_id, line in enumerate(self.sentences):
            if line_id % 10000 == 0:
                print("Progress of process {}: {}/{}".format(p_id, line_id, sentences_num))
                sys.stdout.flush()
            line = line.strip().split('\t')
            try:
                if len(line) == 2:
                    label = int(line[self.columns["label"]])
                    text = CLS_TOKEN + line[self.columns["text_a"]]
    
                    tokens, pos, vm, _ = self.kg.add_knowledge_with_vm([text], add_pad=True, max_length=args.seq_length)
                    tokens = tokens[0]
                    pos = pos[0]
                    vm = vm[0].astype("bool")

                    token_ids = [self.vocab.get(t) for t in tokens]
                    mask = [1 if t != PAD_TOKEN else 0 for t in tokens]

                    dataset.append((token_ids, label, mask, pos, vm))
                
                elif len(line) == 3:
                    label = int(line[self.columns["label"]])
                    text = CLS_TOKEN + line[self.columns["text_a"]] + SEP_TOKEN + line[self.columns["text_b"]] + SEP_TOKEN

                    tokens, pos, vm, _ = kg.add_knowledge_with_vm([text], add_pad=True, max_length=args.seq_length)
                    tokens = tokens[0]
                    pos = pos[0]
                    vm = vm[0].astype("bool")

                    token_ids = [self.vocab.get(t) for t in tokens]
                    mask = []
                    seg_tag = 1
                    for t in tokens:
                        if t == PAD_TOKEN:
                            mask.append(0)
                        else:
                            mask.append(seg_tag)
                        if t == SEP_TOKEN:
                            seg_tag += 1

                    dataset.append((token_ids, label, mask, pos, vm))
                
                elif len(line) == 4:  # for dbqa
                    qid=int(line[self.columns["qid"]])
                    label = int(line[self.columns["label"]])
                    text_a, text_b = line[self.columns["text_a"]], line[self.columns["text_b"]]
                    text = CLS_TOKEN + text_a + SEP_TOKEN + text_b + SEP_TOKEN

                    tokens, pos, vm, _ = kg.add_knowledge_with_vm([text], add_pad=True, max_length=args.seq_length)
                    tokens = tokens[0]
                    pos = pos[0]
                    vm = vm[0].astype("bool")

                    token_ids = [self.vocab.get(t) for t in tokens]
                    mask = []
                    seg_tag = 1
                    for t in tokens:
                        if t == PAD_TOKEN:
                            mask.append(0)
                        else:
                            mask.append(seg_tag)
                        if t == SEP_TOKEN:
                            seg_tag += 1
                    
                    dataset.append((token_ids, label, mask, pos, vm, qid))
                else:
                    pass
                
            except:
                print("Error line: ", line)
        return dataset
        pass