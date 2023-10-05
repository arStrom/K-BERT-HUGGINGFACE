# from transformers import BertTokenizerFast
# # from pytorch_pretrained_bert import BertTokenizer

# from config import BaseConfig

# line = '人工智能是一门极富挑战性的科学。'
# CLS_TOKEN = '[CLS]'
# text = [CLS_TOKEN] + list(line)
# print(text)

# import torch
# import numpy as np
# import time
# from datetime import timedelta


# def get_time_dif(start_time):
#     """
#     获取时间间隔
#     """
#     end_time = time.time()
#     time_dif = end_time - start_time
#     return timedelta(seconds=int(round(time_dif)))

# seq_length = 256
# vm = np.zeros((seq_length,seq_length))
# vm = vm[0].astype("bool")
# vm = torch.LongTensor(vm)
# token_type_ids = torch.LongTensor(np.zeros(seq_length))
# start_time = time.time()
# for i in range(10000):
#     encoder_attention_mask1 = (token_type_ids > 0). \
#             unsqueeze(1). \
#             repeat(1, seq_length, 1). \
#             unsqueeze(1)
#     encoder_attention_mask1 = encoder_attention_mask1.float()
#     encoder_attention_mask1 = (1.0 - encoder_attention_mask1) * -10000.0
#     time_dif = get_time_dif(start_time)
# print("Time: {}".format(time_dif))

# for i in range(10000):
#     encoder_attention_mask2 = vm.unsqueeze(1)
#     encoder_attention_mask2 = encoder_attention_mask2.float()
#     encoder_attention_mask2 = (1.0 - encoder_attention_mask2) * -10000.0
# print("Time: {}".format(time_dif))

# import numpy as np

# labels = ['A', 'B', 'C', 'D']

# y_true = np.array([[0, 1, 0, 1],
#                    [0, 1, 1, 0],
#                    [1, 0, 1, 1]])

# y_pred = np.array([[0, 1, 1, 0],
#                    [0, 1, 1, 0],
#                    [0, 1, 0, 1]])

# import sklearn.metrics as metrics

# print(metrics.accuracy_score(y_true,y_pred)) # 0.33333333
# print(metrics.accuracy_score(np.array([[0, 1], [1, 1]]), np.ones((2, 2)))) # 0.5
# print(metrics.classification_report(y_true, y_pred, target_names=labels, digits=4))

model_name = 'bert-rcnn'
model_type = model_name.split('-')

print(model_type)

