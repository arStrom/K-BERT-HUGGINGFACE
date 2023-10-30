import torch
from evaluate import evaluate

def test(model,test_batch,config,task):

    if torch.cuda.device_count() > 1:
        model.module.load_state_dict(torch.load(config.output_dir + '/pytorch_model.bin'))
    else:
        model.load_state_dict(torch.load(config.output_dir + '/pytorch_model.bin'))


    evaluate(model, test_batch, config, task, is_test=True)

if __name__ == "__main__":
    test()
