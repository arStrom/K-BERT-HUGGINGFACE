import torch
from transformers import BertPreTrainedModel, \
    BertModel, \
    ErniePreTrainedModel, \
    ErnieModel
from torch import nn
import torch.nn.functional as F


class BertForSequenceClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
        Labels for computing the sequence classification/regression loss.
        Indices should be in ``[0, ..., config.num_labels - 1]``.
        If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
        If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """

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
    
        # outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        # if labels is not None:
        #     loss_fct = nn.BCELoss()
        #     loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
        #     outputs = (loss,) + outputs
        # return outputs  # (loss), logits, (hidden_states), (attentions)