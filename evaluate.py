import torch
from dataloader import read_dataset, batch_loader
import torch.nn as nn

# Evaluation function.
def evaluate(args, model, device, is_test, metrics='Acc'):
    if is_test:
        dataset = read_dataset(args.test_path, workers_num=args.workers_num)
    else:
        dataset = read_dataset(args.dev_path, workers_num=args.workers_num)

    input_ids = torch.LongTensor([sample[0] for sample in dataset])
    label_ids = torch.LongTensor([sample[1] for sample in dataset])
    mask_ids = torch.LongTensor([sample[2] for sample in dataset])
    pos_ids = torch.LongTensor([example[3] for example in dataset])
    vms = [example[4] for example in dataset]

    batch_size = args.batch_size
    instances_num = input_ids.size()[0]
    if is_test:
        print("The number of evaluation instances: ", instances_num)

    correct = 0
    # Confusion matrix.
    confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)

    model.eval()
    
    if not args.mean_reciprocal_rank:
        for i, (input_ids_batch, label_ids_batch,  mask_ids_batch, pos_ids_batch, vms_batch) in enumerate(batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vms)):

            # vms_batch = vms_batch.long()
            vms_batch = torch.LongTensor(vms_batch)

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
            pred = torch.argmax(logits, dim=1)
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
        print("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct/len(dataset), correct, len(dataset)))
        if metrics == 'Acc':
            return correct/len(dataset)
        elif metrics == 'f1':
            return label_1_f1
        else:
            return correct/len(dataset)
    else:
        for i, (input_ids_batch, label_ids_batch,  mask_ids_batch, pos_ids_batch, vms_batch) in enumerate(batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vms)):

            vms_batch = torch.LongTensor(vms_batch)

            input_ids_batch = input_ids_batch.to(device)
            label_ids_batch = label_ids_batch.to(device)
            mask_ids_batch = mask_ids_batch.to(device)
            pos_ids_batch = pos_ids_batch.to(device)
            vms_batch = vms_batch.to(device)

            with torch.no_grad():
                loss, _ = model(input_ids=input_ids_batch, 
                    labels=label_ids_batch, 
                    token_type_ids=mask_ids_batch, 
                    position_ids=pos_ids_batch, 
                    visible_matrix=vms_batch)
            logits = nn.Softmax(dim=1)(logits)
            if i == 0:
                logits_all=logits
            if i >= 1:
                logits_all=torch.cat((logits_all,logits),0)
    
        order = -1
        gold = []
        for i in range(len(dataset)):
            qid = dataset[i][-1]
            label = dataset[i][1]
            if qid == order:
                j += 1
                if label == 1:
                    gold.append((qid,j))
            else:
                order = qid
                j = 0
                if label == 1:
                    gold.append((qid,j))

        label_order = []
        order = -1
        for i in range(len(gold)):
            if gold[i][0] == order:
                templist.append(gold[i][1])
            elif gold[i][0] != order:
                order=gold[i][0]
                if i > 0:
                    label_order.append(templist)
                templist = []
                templist.append(gold[i][1])
        label_order.append(templist)

        order = -1
        score_list = []
        for i in range(len(logits_all)):
            score = float(logits_all[i][1])
            qid=int(dataset[i][-1])
            if qid == order:
                templist.append(score)
            else:
                order = qid
                if i > 0:
                    score_list.append(templist)
                templist = []
                templist.append(score)
        score_list.append(templist)

        rank = []
        pred = []
        print(len(score_list))
        print(len(label_order))
        for i in range(len(score_list)):
            if len(label_order[i])==1:
                if label_order[i][0] < len(score_list[i]):
                    true_score = score_list[i][label_order[i][0]]
                    score_list[i].sort(reverse=True)
                    for j in range(len(score_list[i])):
                        if score_list[i][j] == true_score:
                            rank.append(1 / (j + 1))
                else:
                    rank.append(0)

            else:
                true_rank = len(score_list[i])
                for k in range(len(label_order[i])):
                    if label_order[i][k] < len(score_list[i]):
                        true_score = score_list[i][label_order[i][k]]
                        temp = sorted(score_list[i],reverse=True)
                        for j in range(len(temp)):
                            if temp[j] == true_score:
                                if j < true_rank:
                                    true_rank = j
                if true_rank < len(score_list[i]):
                    rank.append(1 / (true_rank + 1))
                else:
                    rank.append(0)
        MRR = sum(rank) / len(rank)
        print("MRR", MRR)
        return MRR