import torch
from dataloader import read_dataset
import torch.nn as nn

# Evaluation function.
def evaluate(model, eval_batch, config, is_test, metrics='Acc'):

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
            try:
                loss, logits = model(input_ids=input_ids_batch, 
                    labels=label_ids_batch, 
                    token_type_ids=mask_ids_batch, 
                    position_ids=pos_ids_batch, 
                    visible_matrix=vms_batch)
            except:
                print(input_ids_batch)
                print(input_ids_batch.size())
                print(vms_batch)
                print(vms_batch.size())

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
    if metrics == 'Acc':
        return correct/instances_num
    elif metrics == 'f1':
        return label_1_f1
    else:
        return correct/instances_num