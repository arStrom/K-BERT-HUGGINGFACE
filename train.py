
import sys
import torch
import torch.nn as nn
import time
from evaluate import evaluate, evaluate_multi_label, evaluate_multi_label_slice
import transformers
from torch.optim import Adam, AdamW
from utils.optimizers import BertAdam
# from transformers import AdamW
from datetime import timedelta
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
    instances_num = len(train_batch.dataset)
    train_steps = int(instances_num * config.epochs_num / batch_size) + 1
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    # optimizer = BertAdam(optimizer_grouped_parameters, lr=config.learning_rate, warmup=config.warmup, t_total=train_steps)
    # optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
    
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

            input_ids_batch = input_ids_batch.transpose(0,1).to(device)
            mask_ids_batch = mask_ids_batch.transpose(0,1).to(device)
            pos_ids_batch = pos_ids_batch.transpose(0,1).to(device)
            vms_batch = vms_batch.transpose(0,1).to(device)
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

            if (i + 1) % config.report_steps == 0:
                avg_loss = total_loss/config.report_steps
                label = label_ids_batch.data.cpu()
                predictions = (logits == logits.max(dim=1, keepdim=True)[0]).to(dtype=torch.int32).cpu()
                train_acc = metrics.accuracy_score(label, predictions)

                if avg_loss < best_avg_loss:
                    best_avg_loss = avg_loss
                    improve = '*'
                    last_improve = i
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                print("Epoch id: {0}, Training steps: {1},  Train Loss: {2:>5.2},  Train Acc: {3:>6.2%},  Train Avg Loss: {4:.3f},  Time: {5} {6}"
                      .format(epoch, i+1, loss.item(), train_acc, total_loss/config.report_steps, time_dif, improve))
                sys.stdout.flush()
                total_loss = 0.0

            if i - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break

        print("Start evaluation on dev dataset.")
        acc, prec, recall, f1, dev_loss = evaluate(model, eval_batch, config, is_test = False)
        if acc > best_result:
            best_result = acc
            print("Start evaluation on test dataset.")
            _, _, _, test_f1, _ = evaluate(model, test_batch, config, is_test = False)
            if(test_f1 > best_test_result):
                best_test_result = test_f1
                model_to_save = model.module if hasattr(model, 'module') else model
                model_to_save.save_pretrained(config.output_dir)



def train_slice(model, train_batch, eval_batch, test_batch, config, task):

    device = config.device
    batch_size = train_batch.batch_size

    num_train_optimization_steps = len(train_batch) * config.epochs_num
    train_steps = int(len(train_batch) * config.epochs_num) + 1
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # optimizer = BertAdam(optimizer_grouped_parameters, lr=config.learning_rate, warmup=config.warmup, t_total=train_steps)
    # optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate) 
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                            int(num_train_optimization_steps * config.warmup),
                                                            num_train_optimization_steps)
    total_loss = 0.0
    best_result = 0.0
    best_test_result = 0.0

    start_time = time.time()

    for epoch in range(1, config.epochs_num+1):
        model.train()
        for i, (input_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch, label_ids_batch) in enumerate(train_batch):

            model.zero_grad()

            input_ids_batch = input_ids_batch.transpose(0,1).to(device)
            mask_ids_batch = mask_ids_batch.transpose(0,1).to(device)
            pos_ids_batch = pos_ids_batch.transpose(0,1).to(device)
            vms_batch = vms_batch.transpose(0,1).to(device)
            label_ids_batch = label_ids_batch.to(device)

            loss, logits = model(input_ids_batch, 
                            mask_ids_batch, 
                            pos_ids_batch, 
                            vms_batch,
                            label_ids_batch)
            
            if torch.cuda.device_count() > 1:
                loss = torch.mean(loss)

            total_loss += loss.item()
            
            if (i + 1) % config.report_steps == 0:
                # 每多少轮输出在训练集和验证集上的效果
                label = label_ids_batch.data.cpu()
                predictions = logits.cpu()
                predictions = normalize_prediction(predictions, config.acc_percent)
                train_acc = metrics.accuracy_score(label, predictions)
                time_dif = get_time_dif(start_time)
                print("Epoch id: {0}, Training steps: {1}, Train Acc: {2:>6.2%}, Train Loss: {3:.6f}, Train Avg Loss: {4:.6f}, Time: {5}"
                      .format(epoch, i+1, train_acc, loss.item(), total_loss/config.report_steps, time_dif))
                sys.stdout.flush()
                total_loss = 0.

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        print("Start evaluation on dev dataset.")
        acc, prec, f1, dev_loss = evaluate_multi_label_slice(model, eval_batch, config, is_test = False)
        if acc > best_result:
            best_result = acc
            print("Start evaluation on test dataset.")
            _, _, test_f1, _ = evaluate_multi_label_slice(model, test_batch, config, is_test = False)
            if(test_f1 > best_test_result):
                best_test_result = test_f1
                model_to_save = model.module if hasattr(model, 'module') else model
                model_to_save.save_pretrained(config.output_dir)


