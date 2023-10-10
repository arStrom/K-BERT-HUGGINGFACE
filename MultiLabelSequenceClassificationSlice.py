import torch
from transformers import BertPreTrainedModel, \
    BertModel, \
    ErniePreTrainedModel, \
    ErnieModel
from torch import nn
import torch.nn.functional as F


class BertForMultiLabelSequenceClassification(BertPreTrainedModel):

    def __init__(self, config, args):
        super(BertForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output_layer_1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.output_layer_2 = nn.Linear(config.hidden_size, config.num_labels)
        self.pooling = args.pooling
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss(reduction='none')
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
                            position_ids=position_ids,
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
        logits = self.sigmoid(self.output_layer_2(output))
        loss = self.criterion(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
        return loss, logits


class BertRCNNForMultiLabelSequenceClassification(BertPreTrainedModel):

    def __init__(self, config, args):
        super(BertRCNNForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output_layer_1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.rnn_hidden = 256
        self.num_layers = 2
        self.dropout_rnn = 0.2
        self.pad_size = args.seq_length  # 每句话处理成的长度(短填长切)
        self.pooling = args.pooling
        self.lstm = nn.LSTM(
            config.hidden_size, self.rnn_hidden, self.num_layers,
            bidirectional=True, batch_first=True, dropout=self.dropout_rnn
        )
        self.classifier = nn.Linear(self.rnn_hidden * 2 + config.hidden_size, self.config.num_labels)
        self.cat = torch.cat
        self.relu = F.relu
        self.maxpool = nn.MaxPool1d(self.pad_size)
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss(reduction='none')
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
                            position_ids=position_ids,
                            head_mask=head_mask)


        output = outputs[0]

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

    def __init__(self, config, args):
        super(BertRNNForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.rnn_hidden = 768
        self.num_layers = 2
        self.dropout_rnn = 0.2
        self.pad_size = args.seq_length  # 每句话处理成的长度(短填长切)
        self.lstm = nn.LSTM(
            config.hidden_size, self.rnn_hidden, self.num_layers,
            bidirectional=True, batch_first=True, dropout=self.dropout_rnn
        )
        self.lstm_dropout = nn.Dropout(self.dropout_rnn)
        self.classifier = nn.Linear(self.rnn_hidden * 2, self.config.num_labels)
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss(reduction='none')
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
                            position_ids=position_ids,
                            head_mask=head_mask)

        out, _ = self.lstm(outputs[0])
        out = self.lstm_dropout(out)
        logits = self.sigmoid(self.classifier(out[:, -1, :]))

        loss = self.criterion(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
        return loss, logits


class BertCNNForMultiLabelSequenceClassification(BertPreTrainedModel):

    def __init__(self, config, args):
        super(BertCNNForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.filter_sizes = (2, 3, 4)  # 卷积核尺寸
        self.num_filters = 256  # 卷积核数量(channels数)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (k, config.hidden_size)) for k in self.filter_sizes])
        self.dropout_cnn = 0.1
        self.cnn_dropout = nn.Dropout(self.dropout_cnn)
        self.classifier = nn.Linear(self.num_filters * len(self.filter_sizes), self.config.num_labels)
        self.cat = torch.cat
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss(reduction='none')
        self.use_vm = False if args.no_vm or args.no_kg else True
        print("[BertClassifier] use visible_matrix: {}".format(self.use_vm))
        self.init_weights()

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

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
                            position_ids=position_ids,
                            head_mask=head_mask)

        out = outputs[0].unsqueeze(1)
        out = self.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.cnn_dropout(out)
        logits = self.sigmoid(self.classifier(out))

        loss = self.criterion(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
        return loss, logits


class ErnieForMultiLabelSequenceClassification(ErniePreTrainedModel):

    def __init__(self, config, args):
        super(ErnieForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.ernie = ErnieModel(config, add_pooling_layer=False)
        for param in self.ernie.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output_layer_1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.output_layer_2 = nn.Linear(config.hidden_size, config.num_labels)
        self.pooling = args.pooling
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss(reduction='none')
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
        outputs = self.ernie(input_ids,
                            attention_mask=token_type_ids,
                            encoder_attention_mask=encoder_attention_mask,
                            position_ids=position_ids,
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
        logits = self.sigmoid(self.output_layer_2(output))
        loss = self.criterion(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
        return loss, logits

class ErnieRCNNForMultiLabelSequenceClassification(ErniePreTrainedModel):

    def __init__(self, config, args):
        super(ErnieRCNNForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.ernie = ErnieModel(config, add_pooling_layer=False)
        for param in self.ernie.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output_layer_1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.rnn_hidden = 256
        self.num_layers = 2
        self.dropout_rnn = 0.2
        self.pad_size = args.seq_length  # 每句话处理成的长度(短填长切)
        self.pooling = args.pooling
        self.lstm = nn.LSTM(
            config.hidden_size, self.rnn_hidden, self.num_layers,
            bidirectional=True, batch_first=True, dropout=self.dropout_rnn
        )
        self.classifier = nn.Linear(self.rnn_hidden * 2 + config.hidden_size, self.config.num_labels)
        self.cat = torch.cat
        self.relu = F.relu
        self.maxpool = nn.MaxPool1d(self.pad_size)
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss(reduction='none')
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
        outputs = self.ernie(input_ids,
                            attention_mask=token_type_ids,
                            encoder_attention_mask=encoder_attention_mask,
                            position_ids=position_ids,
                            head_mask=head_mask)


        output = outputs[0]

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


class ErnieRNNForMultiLabelSequenceClassification(ErniePreTrainedModel):

    def __init__(self, config, args):
        super(ErnieRNNForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.ernie = ErnieModel(config, add_pooling_layer=False)
        for param in self.ernie.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.rnn_hidden = 768
        self.num_layers = 2
        self.dropout_rnn = 0.2
        self.pad_size = args.seq_length  # 每句话处理成的长度(短填长切)
        self.lstm = nn.LSTM(
            config.hidden_size, self.rnn_hidden, self.num_layers,
            bidirectional=True, batch_first=True, dropout=self.dropout_rnn
        )
        self.lstm_dropout = nn.Dropout(self.dropout_rnn)
        self.classifier = nn.Linear(self.rnn_hidden * 2, self.config.num_labels)
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss(reduction='none')
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
        outputs = self.ernie(input_ids,
                            attention_mask=token_type_ids,
                            encoder_attention_mask=encoder_attention_mask,
                            position_ids=position_ids,
                            head_mask=head_mask)

        out, _ = self.lstm(outputs[0])
        out = self.lstm_dropout(out)
        logits = self.sigmoid(self.classifier(out[:, -1, :]))

        loss = self.criterion(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
        return loss, logits


class ErnieCNNForMultiLabelSequenceClassification(ErniePreTrainedModel):

    def __init__(self, config, args):
        super(ErnieCNNForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.ernie = ErnieModel(config, add_pooling_layer=False)
        for param in self.ernie.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.filter_sizes = (2, 3, 4)  # 卷积核尺寸
        self.num_filters = 256  # 卷积核数量(channels数)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (k, config.hidden_size)) for k in self.filter_sizes])
        self.dropout_cnn = 0.1
        self.cnn_dropout = nn.Dropout(self.dropout_cnn)
        self.classifier = nn.Linear(self.num_filters * len(self.filter_sizes), self.config.num_labels)
        self.cat = torch.cat
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss(reduction='none')
        self.use_vm = False if args.no_vm or args.no_kg else True
        print("[BertClassifier] use visible_matrix: {}".format(self.use_vm))
        self.init_weights()

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

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
        outputs = self.ernie(input_ids,
                            attention_mask=token_type_ids,
                            encoder_attention_mask=encoder_attention_mask,
                            position_ids=position_ids,
                            head_mask=head_mask)

        out = outputs[0].unsqueeze(1)
        out = self.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.cnn_dropout(out)
        logits = self.sigmoid(self.classifier(out))

        loss = self.criterion(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
        return loss, logits

