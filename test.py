import torch
from evaluate import evaluate

def test(model,test_batch,config,task):
    # path = 'outputs/bert-rcnn_pretrained=bert_num_epochs=20_batch_size=16_learning_rate=2e-05_max_seq_length=512_no_kg=True_no_vm=False'
    if torch.cuda.device_count() > 1:
        model.module.load_state_dict(torch.load(config.output_dir + '/pytorch_model.bin'))
        # model.module.load_state_dict(torch.load(path + '/pytorch_model.bin'))
    else:
        model.load_state_dict(torch.load(config.output_dir + '/pytorch_model.bin'))
        # model.load_state_dict(torch.load(path + '/pytorch_model.bin'))

    evaluate(model, test_batch, config, task, is_test=True)