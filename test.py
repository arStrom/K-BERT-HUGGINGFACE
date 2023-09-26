# from transformers import BertTokenizerFast
# # from pytorch_pretrained_bert import BertTokenizer

# from config import BaseConfig

# line = '人工智能是一门极富挑战性的科学。'
# CLS_TOKEN = '[CLS]'
# text = [CLS_TOKEN] + list(line)
# print(text)

import numpy as np

vm = np.zeros((256,256))
vm = vm[0].astype("bool")