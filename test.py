import torch
from evaluate import evaluate, evaluate_multi_label

def test(model,test_batch,config):
    
    if torch.cuda.device_count() > 1:
        model.module.load_state_dict(torch.load(config.output_dir + '/pytorch_model.bin'))
    else:
        model.load_state_dict(torch.load(config.output_dir + '/pytorch_model.bin'))
    evaluate_multi_label(model, test_batch, config, is_test = True)