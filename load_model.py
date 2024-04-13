from collections import OrderedDict
import torch
from project1_model import project1_model

def load_checkpoint(filepath):
    map_location = torch.device('cpu') if torch.cuda.is_available() == False else torch.device('gpu')
    checkpoint = torch.load(filepath, map_location=map_location)
    print(checkpoint.keys())
    print(checkpoint['epoch'])
    model, total_params = project1_model(config=checkpoint['config'])
    print(checkpoint['acc'])
    state_dict = checkpoint['net']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return model
