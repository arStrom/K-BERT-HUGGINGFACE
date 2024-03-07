
import sys
import torch
import torch.nn as nn
import time
from datetime import timedelta
from evaluate import evaluate
import transformers
from torch.optim import Adam, AdamW
from utils.optimizers import BertAdam
# from transformers import AdamW
import sklearn.metrics as metrics

def get_time_dif(start_time):
    """
    获取时间间隔
    """
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def normalize_prediction(predictions, acc_percent):
    zero = torch.zeros_like(predictions)
    one = torch.ones_like(predictions)
    predictions = torch.where(predictions > acc_percent, one, zero)
    return predictions


def train(model, train_batch, eval_batch, test_batch, config, task):

    device = config.device
    batch_size = train_batch.batch_size
    num_train_optimization_steps = len(train_batch) * config.epochs_num
    train_steps = int(len(train_batch.dataset) * config.epochs_num / batch_size) + 1
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    # model_parameters = model.module if hasattr(model, 'module') else model
    # pretrained_model_params = list(model_parameters.bert.named_parameters())
    # multi_text_attention_params = list(model_parameters.multi_text_attention.named_parameters())
    # lstm_params = list(model_parameters.lstm.named_parameters())
    # pooler_params = list(model_parameters.pooler.named_parameters())
    # classifier_params = list(model_parameters.classifier.named_parameters())

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # optimizer_grouped_parameters = [
    #     {
    #         "params": [p for n, p in pretrained_model_params if not any(nd in n for nd in no_decay)],
    #         "weight_decay": 0.0,
    #         "lr": config.pretrained_learning_rate  # pretrained_model_params层的学习率
    #     },
    #     {
    #         "params": [p for n, p in pretrained_model_params if any(nd in n for nd in no_decay)],
    #         "weight_decay": 0.0,
    #         "lr": config.pretrained_learning_rate  # pretrained_model_params层的学习率
    #     },
    #     {
    #         "params": [p for n, p in multi_text_attention_params if not any(nd in n for nd in no_decay)],
    #         "weight_decay": 0.01,
    #         "lr": config.learning_rate  # multi_text_attention层的学习率
    #     },
    #     {
    #         "params": [p for n, p in multi_text_attention_params if any(nd in n for nd in no_decay)],
    #         "weight_decay": 0.0,
    #         "lr": config.learning_rate  # multi_text_attention层的学习率
    #     },
    #     {
    #         "params": [p for n, p in lstm_params if not any(nd in n for nd in no_decay)],
    #         "weight_decay": 0.01,
    #         "lr": config.learning_rate  # LSTM层的学习率
    #     },
    #     {
    #         "params": [p for n, p in lstm_params if any(nd in n for nd in no_decay)],
    #         "weight_decay": 0.0,
    #         "lr": config.learning_rate  # LSTM层的学习率
    #     },
    #     {
    #         "params": [p for n, p in pooler_params if not any(nd in n for nd in no_decay)],
    #         "weight_decay": 0.01,
    #         "lr": config.learning_rate  # pooler的学习率
    #     },
    #     {
    #         "params": [p for n, p in pooler_params if any(nd in n for nd in no_decay)],
    #         "weight_decay": 0.0,
    #         "lr": config.learning_rate  # pooler的学习率
    #     },
    #     {
    #         "params": [p for n, p in classifier_params if not any(nd in n for nd in no_decay)],
    #         "weight_decay": 0.01,
    #         "lr": config.learning_rate  # 全连接层的学习率
    #     },
    #     {
    #         "params": [p for n, p in classifier_params if any(nd in n for nd in no_decay)],
    #         "weight_decay": 0.0,
    #         "lr": config.learning_rate
    #     },
    # ]


    # optimizer = BertAdam(optimizer_grouped_parameters, lr=config.learning_rate, warmup=config.warmup, t_total=train_steps)
    # optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate) 
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                            int(num_train_optimization_steps * config.warmup),
                                                            num_train_optimization_steps)
    
    total_loss = 0.0
    result = 0.0
    best_result = 0.0
    best_test_result = 0.0

    start_time = time.time()
    last_improve = 0    # 记录上次验证集loss下降的batch数
    flag = False        # 记录是否很久没有效果提升
    best_avg_loss = float('inf')

    for epoch in range(1, config.epochs_num+1):

        model.train()
        for i, (input_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch, label_ids_batch) in enumerate(train_batch):

            model.zero_grad()

            input_ids_batch = input_ids_batch.to(device)
            mask_ids_batch = mask_ids_batch.to(device)
            pos_ids_batch = pos_ids_batch.to(device)
            vms_batch = vms_batch.to(device)
            label_ids_batch = label_ids_batch.to(device)

            loss, logits = model(input_ids_batch, 
                            mask_ids_batch, 
                            pos_ids_batch, 
                            vms_batch,
                            label_ids_batch)
            
            if torch.cuda.device_count() > 1:
                loss = torch.mean(loss)

            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if (i + 1) % config.report_steps == 0:
                label = label_ids_batch.data.cpu()
                preds = logits.cpu()
                if task == 'SLC':
                    preds = (logits == logits.max(dim=1, keepdim=True)[0]).to(dtype=torch.int32)
                    preds = preds.cpu()
                elif task == 'MLC':
                    preds = normalize_prediction(preds, config.acc_percent)
                preds[preds >= config.acc_percent] = 1
                preds[preds < config.acc_percent] = 0
                train_acc = metrics.accuracy_score(label, preds)
                time_dif = get_time_dif(start_time)
                print("Epoch id: {0:0>2}, Training steps: {1:0>4},  Train Loss: {2:0<6.6f},  Train Acc: {3:0>6.2%},  Train Avg Loss: {4:0<6.6f},  Time: {5}"
                      .format(epoch, i+1, loss.item(), train_acc, total_loss/config.report_steps, time_dif))
                sys.stdout.flush()
                total_loss = 0.0

        print("Start evaluation on dev dataset.")
        acc, prec, recall, f1, dev_loss = evaluate(model, eval_batch, config, task, is_test = False)
        if f1 > best_result:
            best_result = f1
            print("Start evaluation on test dataset.")
            _, _, _, test_f1, _ = evaluate(model, test_batch, config, task, is_test = False)
            if(test_f1 > best_test_result):
                best_test_result = test_f1
                model_to_save = model.module if hasattr(model, 'module') else model
                model_to_save.save_pretrained(config.output_dir)