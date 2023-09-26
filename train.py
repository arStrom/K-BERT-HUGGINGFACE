
import sys
import torch
import random
import time
from evaluate import evaluate
from utils.optimizers import BertAdam
from datetime import timedelta


def get_time_dif(start_time):
    """
    获取时间间隔
    """
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def train(model, train_batch, eval_batch, test_batch, config):

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
    optimizer = BertAdam(optimizer_grouped_parameters, lr=config.learning_rate, warmup=config.warmup, t_total=train_steps)

    total_loss = 0.0
    result = 0.0
    best_result = 0.0

    start_time = time.time()

    for epoch in range(1, config.epochs_num+1):
        model.train()
        for i, (input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch) in enumerate(train_batch):
            model.zero_grad()

            input_ids_batch = input_ids_batch.to(device)
            label_ids_batch = label_ids_batch.to(device)
            mask_ids_batch = mask_ids_batch.to(device)
            pos_ids_batch = pos_ids_batch.to(device)
            vms_batch = vms_batch.to(device)

            loss, _ = model(input_ids=input_ids_batch, 
                            labels=label_ids_batch, 
                            token_type_ids=mask_ids_batch, 
                            position_ids=pos_ids_batch, 
                            visible_matrix=vms_batch)
            if torch.cuda.device_count() > 1:
                loss = torch.mean(loss)
            total_loss += loss.item()
            if (i + 1) % config.report_steps == 0:
                time_dif = get_time_dif(start_time)
                print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}, Time: {}".format(epoch, i+1, total_loss / config.report_steps, time_dif))
                sys.stdout.flush()
                total_loss = 0.
            loss.backward()
            optimizer.step()

        print("Start evaluation on dev dataset.")
        result = evaluate(model, eval_batch, device, is_test = False)
        if result > best_result:
            best_result = result
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(config.output_dir)
        else:
            continue

        print("Start evaluation on test dataset.")
        evaluate(model, test_batch, config, is_test = True)

