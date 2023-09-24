
import torch
from dataloader import read_dataset


def train():
    # Training phase.
    print("Start training.")
    trainset = read_dataset(args.train_path, workers_num=args.workers_num)
    print("Shuffling dataset")
    random.shuffle(trainset)
    instances_num = len(trainset)
    batch_size = args.batch_size

    print("Trans data to tensor.")
    print("input_ids")
    input_ids = torch.LongTensor([example[0] for example in trainset])
    print("label_ids")
    label_ids = torch.LongTensor([example[1] for example in trainset])
    print("mask_ids")
    mask_ids = torch.LongTensor([example[2] for example in trainset])
    print("pos_ids")
    pos_ids = torch.LongTensor([example[3] for example in trainset])
    print("vms")
    vms = [example[4] for example in trainset]

    train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    print("Batch size: ", batch_size)
    print("The number of training instances:", instances_num)

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
    for epoch in range(1, args.epochs_num+1):
        model.train()
        for i, (input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch) in enumerate(batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vms)):
            model.zero_grad()

            vms_batch = torch.LongTensor(vms_batch)

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
            if (i + 1) % args.report_steps == 0:
                print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i+1, total_loss / args.report_steps))
                sys.stdout.flush()
                total_loss = 0.
            loss.backward()
            optimizer.step()

        print("Start evaluation on dev dataset.")
        result = evaluate(args, False)
        if result > best_result:
            best_result = result
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(config.output_dir)
        else:
            continue

        print("Start evaluation on test dataset.")
        evaluate(args, True)

    # Evaluation phase.
    print("Final evaluation on the test dataset.")

    if torch.cuda.device_count() > 1:
        model.module.load_state_dict(torch.load(config.output_dir + '/pytorch_model.bin'))
    else:
        model.load_state_dict(torch.load(config.output_dir + '/pytorch_model.bin'))
    evaluate(args, True)