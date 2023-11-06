import torch
from transformers import BertPreTrainedModel, \
    BertModel, \
    ErniePreTrainedModel, \
    ErnieModel
from torch import nn
import torch.nn.functional as F
import math

class MultiHeadedAttention(nn.Module):
    """
    Each head is a self-attention operation.
    self-attention refers to https://arxiv.org/pdf/1706.03762.pdf
    """
    def __init__(self, hidden_size, heads_num, dropout):
        super(MultiHeadedAttention, self).__init__()
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        self.per_head_size = hidden_size // heads_num

        self.linear_layers = nn.ModuleList([
                nn.Linear(hidden_size, hidden_size) for _ in range(3)
            ])
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, key, value, query, mask):
        """
        Args:
            key: [batch_size x seq_length x hidden_size]
            value: [batch_size x seq_length x hidden_size]
            query: [batch_size x seq_length x hidden_size]
            mask: [batch_size x 1 x seq_length x seq_length]

        Returns:
            output: [batch_size x seq_length x hidden_size]
        """
        batch_size, seq_length, hidden_size = key.size()
        heads_num = self.heads_num
        per_head_size = self.per_head_size

        def shape(x):
            return x. \
                   contiguous(). \
                   view(batch_size, seq_length, heads_num, per_head_size). \
                   transpose(1, 2)

        def unshape(x):
            return x. \
                   transpose(1, 2). \
                   contiguous(). \
                   view(batch_size, seq_length, hidden_size)


        query, key, value = [l(x). \
                             view(batch_size, -1, heads_num, per_head_size). \
                             transpose(1, 2) \
                             for l, x in zip(self.linear_layers, (query, key, value))
                            ]

        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / math.sqrt(float(per_head_size)) 
        scores = scores + mask
        probs = nn.Softmax(dim=-1)(scores)
        probs = self.dropout(probs)
        output = unshape(torch.matmul(probs, value))
        output = self.final_linear(output)
        
        return output
    
class MultiTextAttention(nn.Module):
    """
    Each head is a self-attention operation.
    self-attention refers to https://arxiv.org/pdf/1706.03762.pdf
    """
    def __init__(self, hidden_size, sentence_num, dropout):
        super(MultiHeadedAttention, self).__init__()
        self.hidden_size = hidden_size
        self.sentence_num = sentence_num
        self.linear_layers = nn.ModuleList([
                nn.Linear(hidden_size, hidden_size) for _ in range(sentence_num)
            ])
        self.attention = Attention(dropout)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, querys, keys, values, masks):
        """
        Args:
            querys: [sentence_num x batch_size x seq_length x hidden_size]
            keys: [sentence_num x batch_size x seq_length x hidden_size]
            values: [sentence_num x batch_size x seq_length x hidden_size]
            masks: [sentence_num x batch_size x 1 x seq_length x seq_length]

        Returns:
            output: [batch_size x seq_length x hidden_size]
        """
        batch_size, sentence_num, seq_length, hidden_size = keys.size()
        output_batch = []
        for i in range(sentence_num):
            query, key, value, mask = querys[i], keys[i], values[i], masks[i]
            query, key, value = [l(x) for l, x in zip(self.linear_layers, (query, key, value, mask))]
            output = self.attention(query, key, value)
            output_batch.append(output)

        mixoutput = self.final_linear(output)
        
        return mixoutput

class Attention(nn.Module):
    """
    Each head is a self-attention operation.
    self-attention refers to https://arxiv.org/pdf/1706.03762.pdf
    """
    def __init__(self, dropout):
        super(Attention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask):
        """
        Args:
            query: [batch_size x seq_length x hidden_size]
            key: [batch_size x seq_length x hidden_size]
            value: [batch_size x seq_length x hidden_size]
            mask: [batch_size x 1 x seq_length x seq_length]

        Returns:
            output: [batch_size x seq_length x hidden_size]
        """
        batch_size, seq_length, hidden_size = key.size()

        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / math.sqrt(float(hidden_size))
        scores = scores + mask
        probs = nn.Softmax(dim=-1)(scores)
        probs = self.dropout(probs)
        output = torch.matmul(probs, value)
        
        return output

class ErnieRCNNForMultiLabelSequenceClassificationNew(ErniePreTrainedModel):

    def __init__(self, config, base_config):
        super(ErnieRCNNForMultiLabelSequenceClassificationNew, self).__init__(config)
        # 句子个数
        self.sentence_num = base_config.sentence_num
        self.num_labels = config.num_labels
        self.ernie = ErnieModel(config, add_pooling_layer=False)
        for param in self.ernie.parameters():
            param.requires_grad = True

        self.rnn_hidden = 256
        self.num_layers = 2
        self.dropout_rnn = 0.2
        self.pad_size = base_config.max_seq_length  # 每句话处理成的长度(短填长切)
        self.pooling = base_config.pooling
        self.lstm = nn.LSTM(
            config.hidden_size, self.rnn_hidden, self.num_layers,
            bidirectional=True, batch_first=True, dropout=self.dropout_rnn
        )
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier = nn.Linear(self.rnn_hidden * 2 + config.hidden_size, self.num_labels)
        self.cat = torch.cat
        self.relu = F.relu
        # 在池化层拼接
        self.maxpool = nn.MaxPool1d(self.pad_size)
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()
        self.use_vm = False if base_config.no_vm or base_config.no_kg else True
        print("[BertClassifier] use visible_matrix: {}".format(self.use_vm))
        self.init_weights()

    def forward(self, input_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch, labels):
        
        sequence_output_batch = []
        pooled_output_batch = []
        encoder_attention_mask_batch = []
        for i in range(self.sentence_num):
            input_ids = input_ids_batch[i]
            attention_mask = mask_ids_batch[i]
            position_ids = pos_ids_batch[i]
            visible_matrix = vms_batch[i]

            seq_length = input_ids.size(1)
                
            # Generate mask according to segment indicators.
            # mask: [batch_size x 1 x seq_length x seq_length]
            if visible_matrix is None or not self.use_vm:
                encoder_attention_mask = (attention_mask > 0). \
                        unsqueeze(1). \
                        repeat(1, seq_length, 1). \
                        unsqueeze(1)
                encoder_attention_mask = encoder_attention_mask.float()
                encoder_attention_mask = (1.0 - encoder_attention_mask) * -10000.0
            else:
                encoder_attention_mask = visible_matrix.unsqueeze(1)
                encoder_attention_mask = encoder_attention_mask.float()
                encoder_attention_mask = (1.0 - encoder_attention_mask) * -10000.0

            # token_type_ids实际上是attention_mask
            outputs = self.ernie(input_ids,
                                attention_mask=attention_mask,
                                encoder_attention_mask=encoder_attention_mask,
                                position_ids=position_ids)
                                
            sequence_output = outputs[0]

            # Target.
            if self.pooling == "mean":
                pool_output = torch.mean(sequence_output, dim=1)
            elif self.pooling == "max":
                pool_output = torch.max(sequence_output, dim=1)[0]
            elif self.pooling == "last":
                pool_output = sequence_output[:, -1, :]
            else:
                pool_output = sequence_output[:, 0, :]
            pool_output = torch.tanh(self.pooler(pooled_output))

            pooled_output_batch.append(pool_output)
            sequence_output_batch.append(sequence_output)
            encoder_attention_mask_batch.append(encoder_attention_mask)

        # 序列隐藏信息
        sequence_output = torch.cat(sequence_output_batch, 1)

        pooled_output = torch.cat(pooled_output_batch, 1)

        out, h_n = self.lstm(sequence_output)
        # out = self.cat((sequence_output, out), 2)
        out = self.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze()
        # out = out.permute(0, 2, 1)
        # out = self.dropouts(out)
        if len(out.shape) == 1:
            out = out.unsqueeze(0)
        mixoutput = torch.cat([pool_output,out],-1)
        logits = self.sigmoid(self.classifier(mixoutput))

        loss = self.criterion(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
        return loss, logits