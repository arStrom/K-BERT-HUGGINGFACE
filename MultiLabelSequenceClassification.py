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
        self.feedforward_size = 3072
        self.linear_layers = nn.ModuleList([
                nn.Linear(hidden_size, hidden_size) for _ in range(sentence_num)
            ])
        self.attention = Attention(hidden_size, dropout)
        self.final_linear = nn.Linear(hidden_size * sentence_num, hidden_size * sentence_num)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(hidden_size * sentence_num)

        # self.feed_forward = PositionwiseFeedForward(
        #     hidden_size * sentence_num, self.feedforward_size
        # )

        # self.dropout_2 = nn.Dropout(dropout)
        # self.layer_norm_2 = LayerNorm(hidden_size * sentence_num)

    def forward(self, querys, keys, values, masks, pooled_output):
        """
        Args:
            querys: [sentence_num x batch_size x seq_length x hidden_size]
            keys: [sentence_num x batch_size x seq_length x hidden_size]
            values: [sentence_num x batch_size x seq_length x hidden_size]
            masks: [sentence_num x batch_size x 1 x seq_length x seq_length]

        Returns:
            output: [batch_size x seq_length x hidden_size]
        """
        sentence_num = len(querys)
        output_batch = []
        for i in range(sentence_num):
            query, key, value, mask = querys[i], keys[i], values[i], masks[i]
            batch_size, seq_length, hidden_size = query.size()
            query, key, value = [l(x) for l, x in zip(self.linear_layers, (query, key, value))]
            mask = mask.squeeze(dim=1)
            att_output = self.attention(query, key, value, mask)
            output_batch.append(att_output)

        output = torch.cat(output_batch,-1)

        pooled_output = pooled_output.unsqueeze(1)
        output = torch.cat([pooled_output,output],1)

        # hidden = torch.cat(querys,-1)

        output = self.final_linear(output)
        mixoutput = gelu(self.dropout(output))
        mixoutput = self.layer_norm(mixoutput)
        
        # inter = self.dropout_1(output)
        # inter = self.layer_norm_1(inter + hidden)
        # mixoutput = self.dropout_2(self.feed_forward(inter))
        # mixoutput = self.layer_norm_2(output + inter) 

        return mixoutput


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
    
class ErnieRCNNForMultiLabelSequenceClassificationNew(ErniePreTrainedModel):

    def __init__(self, config, base_config):
        super(ErnieRCNNForMultiLabelSequenceClassificationNew, self).__init__(config)
        # 句子个数
        self.sentence_num = base_config.sentence_num
        self.num_labels = config.num_labels
        self.ernie = ErnieModel(config, add_pooling_layer=False)
        for param in self.ernie.parameters():
            param.requires_grad = True

        self.multi_text_attention = MultiTextAttention(config.hidden_size, 
                                                       self.sentence_num,
                                                       0.5)

        self.rnn_hidden = 768 * self.sentence_num
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
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()
        self.use_vm = False if base_config.no_vm or base_config.no_kg else True
        print("[BertClassifier] use visible_matrix: {}".format(self.use_vm))
        self.init_weights()

    def forward(self, input_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch, labels):
        
        sequence_output_batch = []
        pool_output_batch = []
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
            pool_output = torch.tanh(self.pooler(pool_output))

            pool_output_batch.append(pool_output)
            sequence_output_batch.append(sequence_output)
            encoder_attention_mask_batch.append(encoder_attention_mask)
        # 序列隐藏信息
        # sequence_output = torch.cat(sequence_output_batch, 1)
        pooled_output = torch.cat(pool_output_batch, 1)

        att_output = self.multi_text_attention(sequence_output_batch, sequence_output_batch, sequence_output_batch,
                                               encoder_attention_mask_batch, pooled_output)

        lstm_out, h_n = self.lstm(att_output)

        lstm_out = self.relu(lstm_out)

        lstm_out = lstm_out.permute(0, 2, 1)
        cnn_out = self.maxpool(lstm_out).squeeze()
        # out = out.permute(0, 2, 1)
        # out = self.dropouts(out)

        # if len(lstm_out.shape) == 1:
        #     lstm_out = lstm_out.unsqueeze(0)
        # mixoutput = torch.tanh(self.pooler(mixoutput))

        logits = self.sigmoid(self.classifier(cnn_out))

        loss = self.criterion(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
        return loss, logits

























class BertForMultiLabelSequenceClassification(BertPreTrainedModel):

    def __init__(self, config, base_config):
        super(BertForMultiLabelSequenceClassification, self).__init__(config)
        # 句子个数
        self.sentence_num = base_config.sentence_num
        self.num_labels = config.num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output_layer_1 = nn.Linear(config.hidden_size * self.sentence_num, config.hidden_size)
        self.output_layer_2 = nn.Linear(config.hidden_size, config.num_labels)
        self.pooling = base_config.pooling
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()
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
            outputs = self.bert(input_ids,
                                attention_mask=attention_mask,
                                encoder_attention_mask=encoder_attention_mask,
                                position_ids=position_ids)
            output_batch.append(outputs[0])

        output = torch.cat(output_batch, 2)


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
        logits = self.sigmoid(self.output_layer_2(output))
        loss = self.criterion(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
        return loss, logits

class BertRCNNForMultiLabelSequenceClassification(BertPreTrainedModel):

    def __init__(self, config, base_config):
        super(BertRCNNForMultiLabelSequenceClassification, self).__init__(config)
        # 句子个数
        self.sentence_num = base_config.sentence_num
        self.num_labels = config.num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
        for param in self.bert.parameters():
            param.requires_grad = True
        
        # 相应扩大rnn的隐藏层大小
        self.rnn_hidden = 256 * self.sentence_num
        self.num_layers = 2
        self.dropout_rnn = 0.2
        self.pad_size = base_config.max_seq_length  # 每句话处理成的长度(短填长切)
        self.pooling = base_config.pooling
        # 在rnn的维度拼接
        self.lstm = nn.LSTM(
            config.hidden_size * self.sentence_num, self.rnn_hidden, self.num_layers,
            bidirectional=True, batch_first=True, dropout=self.dropout_rnn
        )
        self.cat = torch.cat
        self.relu = F.relu
        self.maxpool = nn.MaxPool1d(self.pad_size)
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()

        self.classifier = nn.Linear(self.rnn_hidden * 2 + config.hidden_size * self.sentence_num, self.config.num_labels)

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
            outputs = self.bert(input_ids,
                                attention_mask=attention_mask,
                                encoder_attention_mask=encoder_attention_mask,
                                position_ids=position_ids)
            output_batch.append(outputs[0])

        output = torch.cat(output_batch, 2)

        # RCNN融合特征
        out, _ = self.lstm(output)
        out = self.cat((output, out), 2)
        out = self.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze()
        # out = out.permute(0, 2, 1)
        # out = self.dropouts(out)

        logits = self.sigmoid(self.classifier(out))
        loss = self.criterion(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
        return loss, logits

class BertRNNForMultiLabelSequenceClassification(BertPreTrainedModel):

    def __init__(self, config, base_config):
        super(BertRNNForMultiLabelSequenceClassification, self).__init__(config)
        # 句子个数
        self.sentence_num = base_config.sentence_num
        self.num_labels = config.num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.rnn_hidden = 768 * self.sentence_num
        self.num_layers = 2
        self.dropout_rnn = 0.2
        self.pad_size = base_config.max_seq_length  # 每句话处理成的长度(短填长切)
        self.lstm = nn.LSTM(
            config.hidden_size * self.sentence_num, self.rnn_hidden, self.num_layers,
            bidirectional=True, batch_first=True, dropout=self.dropout_rnn
        )
        self.lstm_dropout = nn.Dropout(self.dropout_rnn)
        self.classifier = nn.Linear(self.rnn_hidden * 2, self.config.num_labels)
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()
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
            outputs = self.bert(input_ids,
                                attention_mask=attention_mask,
                                encoder_attention_mask=encoder_attention_mask,
                                position_ids=position_ids)
            output_batch.append(outputs[0])

        output = torch.cat(output_batch, 2)

        out, _ = self.lstm(output)
        out = self.lstm_dropout(out)
        logits = self.sigmoid(self.classifier(out[:, -1, :]))

        loss = self.criterion(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
        return loss, logits

class BertCNNForMultiLabelSequenceClassification(BertPreTrainedModel):

    def __init__(self, config, base_config):
        super(BertCNNForMultiLabelSequenceClassification, self).__init__(config)
        # 句子个数
        self.sentence_num = base_config.sentence_num
        self.num_labels = config.num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
        for param in self.bert.parameters():
            param.requires_grad = True
            
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.filter_sizes = (2, 3, 4)  # 卷积核尺寸
        self.num_filters = 256  # 卷积核数量(channels数)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (k, config.hidden_size * self.sentence_num)) for k in self.filter_sizes])
        self.dropout_cnn = 0.1
        self.cnn_dropout = nn.Dropout(self.dropout_cnn)
        self.classifier = nn.Linear(self.num_filters * len(self.filter_sizes), self.config.num_labels)
        self.cat = torch.cat
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()
        self.use_vm = False if base_config.no_vm or base_config.no_kg else True
        print("[BertClassifier] use visible_matrix: {}".format(self.use_vm))
        self.init_weights()

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

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
            outputs = self.bert(input_ids,
                                attention_mask=attention_mask,
                                encoder_attention_mask=encoder_attention_mask,
                                position_ids=position_ids)
            output_batch.append(outputs[0])

        output = torch.cat(output_batch, 2)

        out = output.unsqueeze(1)
        out = self.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.cnn_dropout(out)
        logits = self.sigmoid(self.classifier(out))

        loss = self.criterion(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
        return loss, logits


class ErnieForMultiLabelSequenceClassification(ErniePreTrainedModel):

    def __init__(self, config, base_config):
        super(ErnieForMultiLabelSequenceClassification, self).__init__(config)
        # 句子个数
        self.sentence_num = base_config.sentence_num
        self.num_labels = config.num_labels
        self.ernie = ErnieModel(config, add_pooling_layer=False)
        for param in self.ernie.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output_layer_1 = nn.Linear(config.hidden_size * self.sentence_num, config.hidden_size)
        self.output_layer_2 = nn.Linear(config.hidden_size, config.num_labels)
        self.pooling = base_config.pooling
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()
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
        logits = self.sigmoid(self.output_layer_2(output))
        loss = self.criterion(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
        return loss, logits

class ErnieRCNNForMultiLabelSequenceClassification(ErniePreTrainedModel):

    def __init__(self, config, base_config):
        super(ErnieRCNNForMultiLabelSequenceClassification, self).__init__(config)
        # 句子个数
        self.sentence_num = base_config.sentence_num
        self.num_labels = config.num_labels
        self.ernie = ErnieModel(config, add_pooling_layer=False)
        for param in self.ernie.parameters():
            param.requires_grad = True

        # 相应扩大rnn的隐藏层大小
        self.rnn_hidden = 256 * self.sentence_num
        self.num_layers = 2
        self.dropout_rnn = 0.2
        self.pad_size = base_config.max_seq_length  # 每句话处理成的长度(短填长切)
        self.pooling = base_config.pooling
        self.SelfAttention = MultiHeadedAttention(config.hidden_size * self.sentence_num, self.sentence_num, 0.1)

        # 在rnn的维度拼接
        self.lstm = nn.LSTM(
            config.hidden_size * self.sentence_num, self.rnn_hidden, self.num_layers,
            bidirectional=True, batch_first=True, dropout=self.dropout_rnn
        )
        self.classifier = nn.Linear(self.rnn_hidden * 2 + config.hidden_size * self.sentence_num, self.config.num_labels)
        self.cat = torch.cat
        self.relu = F.relu
        self.maxpool = nn.MaxPool1d(self.pad_size)
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()
        self.use_vm = False if base_config.no_vm or base_config.no_kg else True
        print("[BertClassifier] use visible_matrix: {}".format(self.use_vm))
        self.init_weights()

    def forward(self, input_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch, labels):
        
        output_batch = []
        mask_batch = []
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

            mask_batch.append(encoder_attention_mask)
            # token_type_ids实际上是attention_mask
            outputs = self.ernie(input_ids,
                                attention_mask=attention_mask,
                                encoder_attention_mask=encoder_attention_mask,
                                position_ids=position_ids)
            output_batch.append(outputs[0])

        output = torch.cat(output_batch, 2)
        mask = torch.cat(mask_batch, 2)
        output = self.SelfAttention(output)

        out, _ = self.lstm(output)
        out = self.cat((output, out), 2)
        out = self.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze()
        # out = out.permute(0, 2, 1)
        # out = self.dropouts(out)
        logits = self.sigmoid(self.classifier(out))

        loss = self.criterion(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
        return loss, logits

class ErnieRCNNForMultiLabelSequenceClassificationCatMaxPool(ErniePreTrainedModel):

    def __init__(self, config, base_config):
        super(ErnieRCNNForMultiLabelSequenceClassificationCatMaxPool, self).__init__(config)
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

class ErnieRCNNForMultiLabelSequenceClassificationCatLSTM(ErniePreTrainedModel):

    def __init__(self, config, base_config):
        super(ErnieRCNNForMultiLabelSequenceClassificationCatLSTM, self).__init__(config)
        # 句子个数
        self.sentence_num = base_config.sentence_num
        self.num_labels = config.num_labels
        self.ernie = ErnieModel(config, add_pooling_layer=False)
        for param in self.ernie.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output_layer_1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.rnn_hidden = 256
        self.num_layers = 2
        self.dropout_rnn = 0.2
        self.pad_size = base_config.max_seq_length  # 每句话处理成的长度(短填长切)
        self.pooling = base_config.pooling

        # 在rnn维度拼接
        self.lstm = nn.LSTM(
            config.hidden_size * self.sentence_num, self.rnn_hidden, self.num_layers,
            bidirectional=True, batch_first=True, dropout=self.dropout_rnn
        )
        self.classifier = nn.Linear(self.rnn_hidden * 2 + config.hidden_size * self.sentence_num, self.config.num_labels)
        self.cat = torch.cat
        self.relu = F.relu
        self.maxpool = nn.MaxPool1d(self.pad_size)
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()
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
        # out = out.permute(0, 2, 1)
        # out = self.dropouts(out)
        logits = self.sigmoid(self.classifier(out))

        loss = self.criterion(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
        return loss, logits
    
class ErnieRCNNForMultiLabelSequenceClassificationCatLSTMWide(ErniePreTrainedModel):

    def __init__(self, config, base_config):
        super(ErnieRCNNForMultiLabelSequenceClassificationCatLSTMWide, self).__init__(config)
        # 句子个数
        self.sentence_num = base_config.sentence_num
        self.num_labels = config.num_labels
        self.ernie = ErnieModel(config, add_pooling_layer=False)
        for param in self.ernie.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output_layer_1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.rnn_hidden = 256 * self.sentence_num
        self.num_layers = 2
        self.dropout_rnn = 0.2
        self.pad_size = base_config.max_seq_length  # 每句话处理成的长度(短填长切)
        self.pooling = base_config.pooling
        self.lstm = nn.LSTM(
            config.hidden_size * self.sentence_num, self.rnn_hidden, self.num_layers,
            bidirectional=True, batch_first=True, dropout=self.dropout_rnn
        )
        self.classifier = nn.Linear(self.rnn_hidden * 2 + config.hidden_size * self.sentence_num, self.config.num_labels)
        self.cat = torch.cat
        self.relu = F.relu
        self.maxpool = nn.MaxPool1d(self.pad_size)
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()
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
        logits = self.sigmoid(self.classifier(out))

        # loss = self.criterion(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
        # return loss, logits
    
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = nn.BCELoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            outputs = (loss,) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)


class ErnieRNNForMultiLabelSequenceClassification(ErniePreTrainedModel):

    def __init__(self, config, base_config):
        super(ErnieRNNForMultiLabelSequenceClassification, self).__init__(config)
        # 句子个数
        self.sentence_num = base_config.sentence_num
        self.num_labels = config.num_labels
        self.ernie = ErnieModel(config, add_pooling_layer=False)
        for param in self.ernie.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 相应扩大rnn隐藏层大小
        self.rnn_hidden = 768 * self.sentence_num
        self.num_layers = 2
        self.dropout_rnn = 0.2
        self.pad_size = base_config.max_seq_length  # 每句话处理成的长度(短填长切)

        # 在rnn维度拼接
        self.lstm = nn.LSTM(
            config.hidden_size * self.sentence_num, self.rnn_hidden, self.num_layers,
            bidirectional=True, batch_first=True, dropout=self.dropout_rnn
        )

        self.lstm_dropout = nn.Dropout(self.dropout_rnn)
        self.classifier = nn.Linear(self.rnn_hidden * 2, self.config.num_labels)
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()
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
        out = self.lstm_dropout(out)
        logits = self.sigmoid(self.classifier(out[:, -1, :]))

        loss = self.criterion(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
        return loss, logits


class ErnieCNNForMultiLabelSequenceClassification(ErniePreTrainedModel):

    def __init__(self, config, base_config):
        super(ErnieCNNForMultiLabelSequenceClassification, self).__init__(config)
        # 句子个数
        self.sentence_num = base_config.sentence_num
        self.num_labels = config.num_labels
        self.ernie = ErnieModel(config, add_pooling_layer=False)
        for param in self.ernie.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.filter_sizes = (2, 3, 4)  # 卷积核尺寸
        self.num_filters = 256  # 卷积核数量(channels数)

        # 将隐藏层信息拼接
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (k, config.hidden_size * self.sentence_num)) for k in self.filter_sizes])
        self.dropout_cnn = 0.1
        self.cnn_dropout = nn.Dropout(self.dropout_cnn)
        self.classifier = nn.Linear(self.num_filters * len(self.filter_sizes), self.config.num_labels)
        self.cat = torch.cat
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()
        self.use_vm = False if base_config.no_vm or base_config.no_kg else True
        print("[BertClassifier] use visible_matrix: {}".format(self.use_vm))
        self.init_weights()

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

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


        out = output.unsqueeze(1)
        out = self.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.cnn_dropout(out)
        logits = self.sigmoid(self.classifier(out))

        loss = self.criterion(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
        return loss, logits


