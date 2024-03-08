import torch
from transformers import BertPreTrainedModel, \
    BertModel, \
    ErniePreTrainedModel, \
    ErnieModel
from torch import nn
import torch.nn.functional as F
import math


class RCNN(nn.Module):
    def __init__(self, input_size, rnn_hidden, num_layers, base_config):
        super().__init__()
        self.num_layers = num_layers
        self.dropout_rnn = 0.2
        self.pad_size = base_config.max_seq_length  # 每句话处理成的长度(短填长切)
        self.lstm = nn.LSTM(
            input_size, rnn_hidden, self.num_layers,
            bidirectional=True, batch_first=True, dropout=self.dropout_rnn
        )
        self.cat = torch.cat
        self.relu = F.relu
        self.maxpool = nn.MaxPool1d(self.pad_size)
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()
    
    def forward(self,bert_output):
        out, _ = self.lstm(bert_output)
        out = self.cat((bert_output, out), 2)
        out = self.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze()

        return out
    

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

        query, key, value = [LinerLayer(x).
                             view(batch_size, -1, heads_num, per_head_size).
                             transpose(1, 2)
                             for LinerLayer, x in zip(self.linear_layers, (query, key, value))]

        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / math.sqrt(float(per_head_size)) 
        scores = scores + mask
        probs = nn.Softmax(dim=-1)(scores)
        probs = self.dropout(probs)
        output = unshape(torch.matmul(probs, value))
        output = self.final_linear(output)
        
        return output


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x-mean) / (std+self.eps) + self.beta
    
def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class PositionwiseFeedForward(nn.Module):
    """ Feed Forward Layer """
    def __init__(self, hidden_size, feedforward_size):
        super(PositionwiseFeedForward, self).__init__()
        self.linear_1 = nn.Linear(hidden_size, feedforward_size)
        self.linear_2 = nn.Linear(feedforward_size, hidden_size)
           
    def forward(self, x):
        inter = gelu(self.linear_1(x))
        output = self.linear_2(inter)
        return output


class MultiTextAttention(nn.Module):
    """
    Each head is a self-attention operation.
    self-attention refers to https://arxiv.org/pdf/1706.03762.pdf
    """
    def __init__(self, hidden_size, sentence_num, dropout):
        super(MultiTextAttention, self).__init__()
        self.hidden_size = hidden_size
        self.sentence_num = sentence_num
        self.linear_layers = nn.ModuleList([
                nn.Linear(hidden_size, hidden_size) for _ in range(sentence_num)
            ])
        self.attention = Attention(hidden_size, dropout)
        self.att_layer = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(hidden_size * sentence_num)

        self.final_linear = nn.Linear(hidden_size * sentence_num, hidden_size * sentence_num)

    def forward(self, fields, mask, pool_output):
        """
        Args:
            querys: [batch_size x sentence_num , seq_length , hidden_size]
            keys: [batch_size x sentence_num , seq_length , hidden_size]
            values: [batch_size x sentence_num , seq_length , hidden_size]
            masks: [batch_size x sentence_num , seq_length , seq_length]

        Returns:
            output: [batch_size x seq_length x hidden_size]
        """
        batch_size, seq_length, hidden_size = fields[0].size()

        fields = [l(x) for l, x in zip(self.linear_layers, fields)]
    
        query = torch.stack(fields).transpose(0,1).flatten(0,1)
        key = query
        value = query

        # query, key, value = [l(x) for l, x in zip(self.linear_layers, (query, key, value))]

        mask = mask.squeeze(dim=1)
        att_output = self.attention(query, key, value, mask)
        att_output = self.att_layer(att_output)


        att_output = att_output.view(batch_size, self.sentence_num, seq_length, hidden_size)
        att_output = att_output.transpose(1,2).contiguous().view(batch_size, seq_length, -1)

        pool_output = pool_output.unsqueeze(1)
        pool_output = pool_output.view(batch_size, self.sentence_num, 1, hidden_size)
        pool_output = pool_output.transpose(1,2).contiguous().view(batch_size, 1, -1)

        att_output = torch.cat([pool_output,att_output],1)
        att_output = gelu(self.final_linear(att_output))

        att_output = self.dropout(att_output)
        att_output = self.layer_norm(att_output)

        return att_output


class Attention(nn.Module):
    """
    Each head is a self-attention operation.
    self-attention refers to https://arxiv.org/pdf/1706.03762.pdf
    """
    def __init__(self, hidden_size, dropout):
        super(Attention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(hidden_size)

    def forward(self, query, key, value, mask):
        """
        Args:
            query: [batch_size x seq_length x hidden_size]
            key: [batch_size x seq_length x hidden_size]
            value: [batch_size x seq_length x hidden_size]
            mask: [batch_size x seq_length x seq_length]
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

class ErnieRCNNForSequenceClassificationNew(ErniePreTrainedModel):

    def __init__(self, config, base_config):
        super(ErnieRCNNForSequenceClassificationNew, self).__init__(config)
        # 句子个数
        self.sentence_num = base_config.sentence_num
        self.hidden_size = config.hidden_size
        self.num_labels = config.num_labels
        self.ernie = ErnieModel(config, add_pooling_layer=False)
        for param in self.ernie.parameters():
            param.requires_grad = True

        self.multi_text_attention = MultiTextAttention(config.hidden_size, 
                                                       self.sentence_num,
                                                       0.5)

        self.rnn_hidden = 768 * 3
        self.num_layers = 2
        self.dropout_rnn = 0.2
        self.pad_size = base_config.max_seq_length  # 每句话处理成的长度(短填长切)
        self.pooling = base_config.pooling

        self.lstm = nn.LSTM(
            config.hidden_size * self.sentence_num, self.rnn_hidden, self.num_layers,
            bidirectional=True, batch_first=True, dropout=self.dropout_rnn
        )

        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)

        self.classifier = nn.Linear(self.rnn_hidden * 2, self.num_labels)

        self.cat = torch.cat
        self.relu = F.relu
        # 在池化层拼接
        self.maxpool = nn.MaxPool1d(self.pad_size + 1)
        self.softmax = nn.Softmax(dim=1)
        self.criterion = nn.CrossEntropyLoss()
        self.use_vm = False if base_config.no_vm or base_config.no_kg else True
        print("[BertClassifier] use visible_matrix: {}".format(self.use_vm))
        self.init_weights()

    def forward(self, input_ids, mask_ids, pos_ids, vms, labels):
        
        # 合并前两个维度 batch_size 和 sentences_num
        input_ids = input_ids.flatten(0,1)
        mask_ids = mask_ids.flatten(0,1)
        pos_ids = pos_ids.flatten(0,1)
        vms = vms.flatten(0,1)

        seq_length = input_ids.size(1)
        if vms is None or not self.use_vm:
            encoder_attention_mask = (mask_ids > 0). \
                    unsqueeze(1). \
                    repeat(1, seq_length, 1). \
                    unsqueeze(1)
            encoder_attention_mask = encoder_attention_mask.float()
            encoder_attention_mask = (1.0 - encoder_attention_mask) * -10000.0
        else:
            encoder_attention_mask = vms.unsqueeze(1)
            encoder_attention_mask = encoder_attention_mask.float()
            encoder_attention_mask = (1.0 - encoder_attention_mask) * -10000.0

        outputs = self.ernie(input_ids,
                            attention_mask=mask_ids,
                            encoder_attention_mask=encoder_attention_mask,
                            position_ids=pos_ids)
                            
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
        pool_output = torch.tanh(self.pooler(pool_output))

        sequence_output = sequence_output.view(-1, self.sentence_num, seq_length, self.hidden_size).transpose(0,1)

        att_output = self.multi_text_attention([x for x in sequence_output],
                                               encoder_attention_mask, pool_output)

        lstm_out, h_n = self.lstm(att_output)
        lstm_out = self.relu(lstm_out)

        lstm_out = lstm_out.permute(0, 2, 1)
        cnn_out = self.maxpool(lstm_out).squeeze()

        logits = self.softmax(self.classifier(cnn_out))
        loss = self.criterion(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
        logits = self.softmax(logits)
        return loss, logits

        # logits = self.sigmoid(self.classifier(cnn_out))
        # if len(logits.shape) == 1:
        #     logits = logits.unsqueeze(0)

        # loss = self.criterion(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
        # if len(loss.shape) == 1:
        #     loss = loss.unsqueeze(0)
        # return loss, logits



class BertRCNNForSequenceClassificationNew(BertPreTrainedModel):

    def __init__(self, config, base_config):
        super(BertRCNNForSequenceClassificationNew, self).__init__(config)
        # 句子个数
        self.sentence_num = base_config.sentence_num
        self.hidden_size = config.hidden_size
        self.num_labels = config.num_labels
        self.bert = BertModel(config, add_pooling_layer=False)

        self.multi_text_attention = MultiTextAttention(config.hidden_size, 
                                                       self.sentence_num,
                                                       0.5)

        self.rnn_hidden = 768 * 3
        self.num_layers = 2
        self.dropout_rnn = 0.2
        self.pad_size = base_config.max_seq_length  # 每句话处理成的长度(短填长切)
        self.pooling = base_config.pooling

        self.lstm = nn.LSTM(
            config.hidden_size * self.sentence_num, self.rnn_hidden, self.num_layers,
            bidirectional=True, batch_first=True, dropout=self.dropout_rnn
        )

        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)

        self.classifier = nn.Linear(self.rnn_hidden * 2, self.num_labels)

        self.cat = torch.cat
        self.relu = F.relu
        # 在池化层拼接
        self.maxpool = nn.MaxPool1d(self.pad_size + 1)
        self.softmax = nn.Softmax(dim=1)
        self.criterion = nn.CrossEntropyLoss()
        self.use_vm = False if base_config.no_vm or base_config.no_kg else True
        print("[BertClassifier] use visible_matrix: {}".format(self.use_vm))
        self.init_weights()

    def forward(self, input_ids, mask_ids, pos_ids, vms, labels):
        
        # 合并前两个维度 batch_size 和 sentences_num
        input_ids = input_ids.flatten(0,1)
        mask_ids = mask_ids.flatten(0,1)
        pos_ids = pos_ids.flatten(0,1)
        vms = vms.flatten(0,1)

        seq_length = input_ids.size(1)
        if vms is None or not self.use_vm:
            encoder_attention_mask = (mask_ids > 0). \
                    unsqueeze(1). \
                    repeat(1, seq_length, 1). \
                    unsqueeze(1)
            encoder_attention_mask = encoder_attention_mask.float()
            encoder_attention_mask = (1.0 - encoder_attention_mask) * -10000.0
        else:
            encoder_attention_mask = vms.unsqueeze(1)
            encoder_attention_mask = encoder_attention_mask.float()
            encoder_attention_mask = (1.0 - encoder_attention_mask) * -10000.0

        outputs = self.bert(input_ids,
                            attention_mask=mask_ids,
                            encoder_attention_mask=encoder_attention_mask,
                            position_ids=pos_ids)
                            
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
        pool_output = torch.tanh(self.pooler(pool_output))

        sequence_output = sequence_output.view(-1, self.sentence_num, seq_length, self.hidden_size).transpose(0,1)

        att_output = self.multi_text_attention([x for x in sequence_output],
                                               encoder_attention_mask, pool_output)

        lstm_out, h_n = self.lstm(att_output)
        lstm_out = self.relu(lstm_out)

        lstm_out = lstm_out.permute(0, 2, 1)
        cnn_out = self.maxpool(lstm_out).squeeze()

        logits = self.softmax(self.classifier(cnn_out))
        loss = self.criterion(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
        logits = self.softmax(logits)
        return loss, logits


























class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, args):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output_layer_1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.output_layer_2 = nn.Linear(config.hidden_size, config.num_labels)
        self.pooling = args.pooling
        self.softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.NLLLoss()
        self.use_vm = False if args.no_vm or args.no_kg else True
        print("[BertClassifier] use visible_matrix: {}".format(self.use_vm))
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None, visible_matrix=None):
        
        seq_length = input_ids.size(1)
        
        # Generate mask according to segment indicators.
        # mask: [batch_size x 1 x seq_length x seq_length]
        if visible_matrix is None or not self.use_vm:
            encoder_attention_mask = (token_type_ids > 0). \
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
        outputs = self.bert(input_ids,
                            attention_mask=token_type_ids,
                            encoder_attention_mask=encoder_attention_mask,
                            head_mask=head_mask)
        
        output = outputs[0]
        # Target.
        if self.pooling == "mean":
            output = torch.mean(output, dim=1)
        elif self.pooling == "max":
            output = torch.max(output, dim=1)[0]
        elif self.pooling == "last":
            output = output[:, -1, :]
        else:
            output = output[:, 0, :]
        output = torch.tanh(self.output_layer_1(output))

        logits = self.output_layer_2(output)
        loss = self.criterion(self.softmax(logits.view(-1, self.num_labels)), labels.view(-1))
        return loss, logits
    

class ErnieRCNNForSequenceClassification(ErniePreTrainedModel):

    def __init__(self, config, base_config):
        super(ErnieRCNNForSequenceClassification, self).__init__(config)
        self.sentence_num = base_config.sentence_num
        self.num_labels = config.num_labels
        self.ernie = ErnieModel(config, add_pooling_layer=False)
        for param in self.ernie.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output_layer_1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.rnn_hidden = 256 * base_config.sentence_num
        self.num_layers = 2
        self.dropout_rnn = 0.2
        self.pad_size = base_config.max_seq_length  # 每句话处理成的长度(短填长切)
        self.pooling = base_config.pooling
        self.lstm = nn.LSTM(
            config.hidden_size * base_config.sentence_num, self.rnn_hidden, self.num_layers,
            bidirectional=True, batch_first=True, dropout=self.dropout_rnn
        )
        self.classifier = nn.Linear(self.rnn_hidden * 2 + config.hidden_size * base_config.sentence_num, self.config.num_labels)
        self.cat = torch.cat
        self.relu = F.relu
        self.maxpool = nn.MaxPool1d(self.pad_size)
        self.softmax = nn.Softmax(dim=1)
        self.criterion = nn.CrossEntropyLoss()
        self.use_vm = False if base_config.no_vm or base_config.no_kg else True
        print("[BertClassifier] use visible_matrix: {}".format(self.use_vm))
        self.init_weights()

    def forward(self, input_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch, labels):
        
        output_batch = []
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
            output_batch.append(outputs[0])

        output = torch.cat(output_batch, 2)

        out, _ = self.lstm(output)
        out = self.cat((output, out), 2)
        out = self.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze()
        # out = self.dropouts(out)

        logits = self.classifier(out)
        loss = self.criterion(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
        logits = self.softmax(logits)
        return loss, logits