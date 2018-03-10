from pytorch_utils import * 
import scipy.io as sio 
import torch 
import sys 
from custom_models import *

model_name = sys.argv[1]
layer_name = sys.argv[2]
model = torch.load(model_name)

#model = LenetZhang()   
#model.load_state_dict(torch.load(model_name))
for i, (name, layer) in enumerate(model.named_modules()): 
    if layer_name in name: 
        print('Dumping weights of layer: ' + name)
        dump_layer_weights(layer)
 
        
