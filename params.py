import torch


# No. of trianable parameters
def count_parameters(model):
    param_count = sum(v.numel() for k, v in model.items())
    return param_count


model = torch.load('models/BERT_FIRE.bin', map_location=torch.device('cpu'))

print(count_parameters(model))
