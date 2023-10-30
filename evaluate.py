import torch
import torch.nn as nn
import numpy as np
import sklearn.metrics as metrics

def normalize_prediction(predictions, acc_percent):
    zero = torch.zeros_like(predictions)
    one = torch.ones_like(predictions)
    predictions = torch.where(predictions > acc_percent, one, zero)
    return predictions

# Evaluation function.
def evaluate(model, eval_batch, config, task, is_test):

    predict_all = None
    labels_all = None
    device = config.device
    instances_num = len(eval_batch.dataset)
    if is_test:
        print("The number of evaluation instances: ", instances_num)

    total_loss = 0.0
    model.eval()
    
    with torch.no_grad():
        for i, (input_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch, label_ids_batch) in enumerate(eval_batch):

            input_ids_batch = input_ids_batch.transpose(0,1).to(device)
            mask_ids_batch = mask_ids_batch.transpose(0,1).to(device)
            pos_ids_batch = pos_ids_batch.transpose(0,1).to(device)
            vms_batch = vms_batch.transpose(0,1).to(device)
            label_ids_batch = label_ids_batch.to(device)

            try:
                loss, logits = model(input_ids_batch, 
                                    mask_ids_batch, 
                                    pos_ids_batch, 
                                    vms_batch,
                                    label_ids_batch)
            except Exception as ex:
                print("出现如下异常%s"%ex)
                print(input_ids_batch)
                print(input_ids_batch.size())
                print(vms_batch)
                print(vms_batch.size())
            
            total_loss += loss.item()
            if task == 'SLC':
                preds = (logits == logits.max(dim=1, keepdim=True)[0]).to(dtype=torch.int32)
                preds = preds.cpu().numpy()
            elif task == 'MLC':
                preds = logits.cpu().numpy()
                preds[preds >= config.acc_percent] = 1
                preds[preds < config.acc_percent] = 0
            else:
                raise NameError("任务名称错误")

            labels = label_ids_batch.cpu().numpy()
            if len(preds.shape) == 1:
                preds = np.expand_dims(preds,axis=0)

            if predict_all is None:
                predict_all = preds
                labels_all = labels
            else:
                labels_all = np.append(labels_all, labels, axis=0)
                predict_all = np.append(predict_all, preds, axis=0)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if task == 'SLC':
        prec = metrics.precision_score(y_true=labels_all, y_pred=predict_all, average='weighted', zero_division=0)
        recall = metrics.recall_score(labels_all, predict_all, average='weighted', zero_division=0)
        f1 = metrics.f1_score(labels_all, predict_all, average='weighted', zero_division=0)
    elif task == 'MLC':
        prec = metrics.precision_score(y_true=labels_all, y_pred=predict_all, average='micro', zero_division=0)
        recall = metrics.recall_score(labels_all, predict_all, average='micro', zero_division=0)
        f1 = metrics.f1_score(labels_all, predict_all, average='micro', zero_division=0)

    print("Acc. (Correct/Total): {:>6.2%} micro-Prec: {:.4f} micro-Recall:{:.4f} micro-F1: {:.4f} dev loss: {:.6f}"
        .format(acc, prec, recall, f1, total_loss / len(eval_batch))) 
    if is_test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4, zero_division=0)
        print(report)
        return acc, prec, recall, f1, total_loss / len(eval_batch), report
    
    return acc, prec, recall, f1, total_loss / len(eval_batch)
