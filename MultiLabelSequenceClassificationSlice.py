import torch
from transformers import BertPreTrainedModel, \
    BertModel, \
    ErniePreTrainedModel, \
    ErnieModel
from torch import nn
import torch.nn.functional as F


class BertForMultiLabelSequenceClassificationSlice(BertPreTrainedModel):

    def __init__(self, config, args):
        super(BertForMultiLabelSequenceClassificationSlice, self).__init__(config)
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

    def forward(self, input_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch, labels):
        output_batch = []
        for i in range(3):
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

        output = (output_batch[0] + output_batch[1] + output_batch[2]) / 3

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


class BertRCNNForMultiLabelSequenceClassificationSlice(BertPreTrainedModel):

    def __init__(self, config, args):
        super(BertRCNNForMultiLabelSequenceClassificationSlice, self).__init__(config)
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

    def forward(self, input_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch, labels):
        
        output_batch = []
        for i in range(3):
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

        output = (output_batch[0] + output_batch[1] + output_batch[2]) / 3


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


class BertRNNForMultiLabelSequenceClassificationSlice(BertPreTrainedModel):

    def __init__(self, config, args):
        super(BertRNNForMultiLabelSequenceClassificationSlice, self).__init__(config)
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

    def forward(self, input_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch, labels):
        
        output_batch = []
        for i in range(3):
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

        output = (output_batch[0] + output_batch[1] + output_batch[2]) / 3


        out, _ = self.lstm(output)
        out = self.lstm_dropout(out)
        logits = self.sigmoid(self.classifier(out[:, -1, :]))

        loss = self.criterion(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
        return loss, logits


class BertCNNForMultiLabelSequenceClassificationSlice(BertPreTrainedModel):

    def __init__(self, config, args):
        super(BertCNNForMultiLabelSequenceClassificationSlice, self).__init__(config)
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

    def forward(self, input_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch, labels):
        
        output_batch = []
        for i in range(3):
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

        output = (output_batch[0] + output_batch[1] + output_batch[2]) / 3

        out = output.unsqueeze(1)
        out = self.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.cnn_dropout(out)
        logits = self.sigmoid(self.classifier(out))

        loss = self.criterion(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
        return loss, logits


class ErnieForMultiLabelSequenceClassificationSlice(ErniePreTrainedModel):

    def __init__(self, config, args):
        super(ErnieForMultiLabelSequenceClassificationSlice, self).__init__(config)
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

    def forward(self, input_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch, labels):
        
        output_batch = []
        for i in range(3):
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

        output = (output_batch[0] + output_batch[1] + output_batch[2]) / 3

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

class ErnieRCNNForMultiLabelSequenceClassificationSlice(ErniePreTrainedModel):

    def __init__(self, config, args):
        super(ErnieRCNNForMultiLabelSequenceClassificationSlice, self).__init__(config)
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

    def forward(self, input_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch, labels):
        
        output_batch = []
        for i in range(3):
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

        output = (output_batch[0] + output_batch[1] + output_batch[2]) / 3

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

class ErnieRCNNForMultiLabelSequenceClassificationSliceCatMaxPool(ErniePreTrainedModel):

    def __init__(self, config, args):
        super(ErnieRCNNForMultiLabelSequenceClassificationSliceCatMaxPool, self).__init__(config)
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
        self.maxpool = nn.MaxPool1d(self.pad_size * 3)
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss(reduction='none')
        self.use_vm = False if args.no_vm or args.no_kg else True
        print("[BertClassifier] use visible_matrix: {}".format(self.use_vm))
        self.init_weights()

    def forward(self, input_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch, labels):
        
        output_batch = []
        for i in range(3):
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

        output = torch.cat((output_batch[0], output_batch[1], output_batch[2]),1)

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

class ErnieRCNNForMultiLabelSequenceClassificationSliceCatLSTM(ErniePreTrainedModel):

    def __init__(self, config, args):
        super(ErnieRCNNForMultiLabelSequenceClassificationSliceCatLSTM, self).__init__(config)
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
            config.hidden_size * 3, self.rnn_hidden, self.num_layers,
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

    def forward(self, input_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch, labels):
        
        output_batch = []
        for i in range(3):
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

        output = torch.cat((output_batch[0], output_batch[1], output_batch[2]), 2)

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


class ErnieRNNForMultiLabelSequenceClassificationSlice(ErniePreTrainedModel):

    def __init__(self, config, args):
        super(ErnieRNNForMultiLabelSequenceClassificationSlice, self).__init__(config)
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

    def forward(self, input_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch, labels):
        
        output_batch = []
        for i in range(3):
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

        output = (output_batch[0] + output_batch[1] + output_batch[2]) / 3

        out, _ = self.lstm(output)
        out = self.lstm_dropout(out)
        logits = self.sigmoid(self.classifier(out[:, -1, :]))

        loss = self.criterion(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
        return loss, logits


class ErnieCNNForMultiLabelSequenceClassificationSlice(ErniePreTrainedModel):

    def __init__(self, config, args):
        super(ErnieCNNForMultiLabelSequenceClassificationSlice, self).__init__(config)
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

    def forward(self, input_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch, labels):
        
        output_batch = []
        for i in range(3):
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

        output = (output_batch[0] + output_batch[1] + output_batch[2]) / 3


        out = output.unsqueeze(1)
        out = self.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.cnn_dropout(out)
        logits = self.sigmoid(self.classifier(out))

        loss = self.criterion(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
        return loss, logits


