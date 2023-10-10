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
def evaluate(model, eval_batch, config, is_test):

    predict_all = None
    labels_all = None
    device = config.device
    instances_num = len(eval_batch.dataset)
    if is_test:
        print("The number of evaluation instances: ", instances_num)

    correct = 0
    # Confusion matrix.
    confusion = torch.zeros(config.label_number, config.label_number, dtype=torch.long).to(device)

    model.eval()
    
    for i, (input_ids_batch, label_ids_batch,  mask_ids_batch, pos_ids_batch, vms_batch) in enumerate(eval_batch):

        input_ids_batch = input_ids_batch.to(device)
        label_ids_batch = label_ids_batch.to(device)
        mask_ids_batch = mask_ids_batch.to(device)
        pos_ids_batch = pos_ids_batch.to(device)
        vms_batch = vms_batch.to(device)

        with torch.no_grad():
            # try:
            loss, logits = model(input_ids=input_ids_batch, 
                labels=label_ids_batch, 
                token_type_ids=mask_ids_batch, 
                position_ids=pos_ids_batch, 
                visible_matrix=vms_batch)
            # except:
            #     print(input_ids_batch)
            #     print(input_ids_batch.size())
            #     print(vms_batch)
            #     print(vms_batch.size())

        logits = nn.Softmax(dim=1)(logits)
        pred = torch.argmax(logits, dim=1).unsqueeze(1)
        gold = label_ids_batch
        for j in range(pred.size()[0]):
            confusion[pred[j], gold[j]] += 1
        correct += torch.sum(pred == gold).item()

    if is_test:
        print("Confusion matrix:")
        print(confusion)
        print("Report precision, recall, and f1:")
    
    for i in range(confusion.size()[0]):
        p = confusion[i,i].item()/confusion[i,:].sum().item()
        r = confusion[i,i].item()/confusion[:,i].sum().item()
        f1 = 2*p*r / (p+r)
        if i == 1:
            label_1_f1 = f1
        print("Label {}: {:.3f}, {:.3f}, {:.3f}".format(i,p,r,f1))
    print("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct/instances_num, correct, instances_num))

    return correct/len(eval_batch)




# Evaluation function.
def evaluate_multi_label(model, eval_batch, config, is_test):

    predict_all = None
    labels_all = None
    device = config.device
    instances_num = len(eval_batch.dataset)
    if is_test:
        print("The number of evaluation instances: ", instances_num)

    correct = 0

    model.eval()
    loss_total = 0
    pred_all = None
    labels_all = None
    with torch.no_grad():
        for i, (input_ids_batch, label_ids_batch,  mask_ids_batch, pos_ids_batch, vms_batch) in enumerate(eval_batch):

            input_ids_batch = input_ids_batch.to(device)
            label_ids_batch = label_ids_batch.to(device)
            mask_ids_batch = mask_ids_batch.to(device)
            pos_ids_batch = pos_ids_batch.to(device)
            vms_batch = vms_batch.to(device)
            # try:
            loss, logits = model(input_ids=input_ids_batch, 
                labels=label_ids_batch, 
                token_type_ids=mask_ids_batch, 
                position_ids=pos_ids_batch, 
                visible_matrix=vms_batch)
            # except:
            #     print(input_ids_batch)
            #     print(input_ids_batch.size())
            #     print(vms_batch)
            #     print(vms_batch.size())
            loss_total += loss.mean()
            labels = label_ids_batch.data.cpu().numpy()
            pred = logits.cpu().numpy()
            pred[pred >= config.acc_percent] = 1
            pred[pred < config.acc_percent] = 0
            if len(pred.shape) == 1:
                pred = np.expand_dims(pred,axis=0)
            correct += np.sum(pred == labels)
            if predict_all is None:
                predict_all = pred
                labels_all = labels
            else:
                labels_all = np.append(labels_all, labels, axis=0)
                predict_all = np.append(predict_all, pred, axis=0)
    acc = metrics.accuracy_score(labels_all, predict_all)
    prec = metrics.precision_score(y_true=labels_all, y_pred=predict_all, average='samples')
    f1 = metrics.f1_score(labels_all, predict_all, average='samples')

    if is_test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        print("Acc. (Correct/Total): {:>6.2%} Prec: {:.4f} F1: {:.4f}".format(acc, prec, f1))
        print("Precision, Recall and F1-Score...")
        print(report)
        return acc, prec, f1, loss_total / len(eval_batch), report
    

    print("Acc. (Correct/Total): {:>6.2%}% Prec: {:.4f} F1: {:.4f}".format(acc, prec, f1))
    return acc, prec, f1, loss_total / len(eval_batch)


# Evaluation function.
def evaluate_multi_label_slice(model, eval_batch, config, is_test):
    
    device = config.device
    instances_num = len(eval_batch.dataset)
    if is_test:
        print("The number of evaluation instances: ", instances_num)

    correct = 0

    model.eval()
    loss_total = 0
    predict_all = None
    labels_all = None
    with torch.no_grad():
        for i, (input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, label_ids_batch) in enumerate(eval_batch):

            input_ids_batch = input_ids_batch.transpose(0,1).to(device)
            mask_ids_batch = mask_ids_batch.transpose(0,1).to(device)
            pos_ids_batch = pos_ids_batch.transpose(0,1).to(device)
            vms_batch = vms_batch.transpose(0,1).to(device)
            label_ids_batch = label_ids_batch.to(device)

            # try:
            loss, logits = model(input_ids_batch, 
                                mask_ids_batch, 
                                pos_ids_batch, 
                                vms_batch,
                                label_ids_batch)
            # except:
            #     print(input_ids_batch)
            #     print(input_ids_batch.size())
            #     print(vms_batch)
            #     print(vms_batch.size())
            loss_total += loss.mean()
            labels = label_ids_batch.data.cpu().numpy()
            pred = logits.cpu().numpy()
            pred[pred >= config.acc_percent] = 1
            pred[pred < config.acc_percent] = 0
            if len(pred.shape) == 1:
                pred = np.expand_dims(pred,axis=0)
            correct += np.sum(pred == labels)
            if predict_all is None:
                predict_all = pred
                labels_all = labels
            else:
                labels_all = np.append(labels_all, labels, axis=0)
                predict_all = np.append(predict_all, pred, axis=0)
    acc = metrics.accuracy_score(labels_all, predict_all)
    prec = metrics.precision_score(y_true=labels_all, y_pred=predict_all, average='samples')
    f1 = metrics.f1_score(labels_all, predict_all, average='samples')

    if is_test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        print("Acc. (Correct/Total): {:>6.2%} Prec: {:.4f} F1: {:.4f}".format(acc, prec, f1))
        print("Precision, Recall and F1-Score...")
        print(report)
        return acc, prec, f1, loss_total / len(eval_batch), report
    

    print("Acc. (Correct/Total): {:>6.2%}% Prec: {:.4f} F1: {:.4f}".format(acc, prec, f1))
    return acc, prec, f1, loss_total / len(eval_batch)