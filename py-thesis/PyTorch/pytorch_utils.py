'''
A personal collection of PyTorch utils for Deep Learning.
A. Salman 
'''


from torch.nn.modules.module import _addindent
import torch
from torch.autograd import Variable
# Utils 
import numpy as np
from logger import Logger
import scipy.io as sio
import os 
import time 

def to_np(x): 
    return x.data.cpu().numpy()

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def log_csv(step, acc, loss, file='cifar10.csv'):
    with open(file, 'a') as out:
        out.write("%d,%.3f,%.3f\n" % (step, acc, loss))
        out.close()

def get_layer_weights(layer, numpy=True):
    '''
        Return weights of the given layer. 
    Args:
        numpy: bool. Defalt true. If false return weights as torch array. 
    '''
    print('Retrieving weights of size: ' + str(layer.weight.data.numpy().shape))
    if numpy: 
        return to_np(layer.weight.data)
    else: 
        return layer.weight.data

def set_layer_weights(layer, tensor): 
    if not(layer.weight.data.numpy().shape == tensor.shape):
        raise Exception('Size mismatch! Cannot asssign weights')

    layer.weight.data = tensor
        
# Summary of a model, as in Keras .summary() method
def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""

    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and
        # weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr += ', parameters={}'.format(params)
        tmpstr += '\n'

    tmpstr = tmpstr + ')'
    return tmpstr

 # ============ TensorBoard logging ============#

def tensorboard_log(steps, model, info, dir='./logs'):
    logger = Logger(dir)

    for tag, value in info.items():
        logger.scalar_summary(tag, value, steps)

    # (2) Log values and gradients of the parameters (histogram)    
    for tag, value in model.named_parameters():
        # print(str(tag)+"  "+str(value))
        tag = tag.replace('.', '/')
        logger.histo_summary(tag, to_np(value), steps)
        if 'bn' not in tag:
            logger.histo_summary(
                tag + '/grad', to_np(value.grad), steps)
   


# Helper function to save weights in MAT format 
def save_weigths_to_mat(allweights, save_dir):
    for idx, weights in enumerate(allweights):
        name = os.path.join(save_dir, "conv" + str(
            idx) + ".mat")  # conv1.mat, conv2.mat, ...
        sio.savemat(name,  {'weights': weights})


def dump_model_weights(model, save_dir='./dumps'):
    '''
    Dump weights for all Conv2D layers and saves it as .mat files
    TODO: Add check if file exists
    '''
    save_dir = os.path.join(os.getcwd(), save_dir)
    # create dir if not exists
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    allweights = []   
    for layer in model.children():
        if isinstance(layer, torch.nn.modules.conv.Conv2d):
            print('Saving layer: ' + str(layer) + ' to ' + save_dir)
            tmp = []      
            tmp.append(layer.weight)
            tmp.append(layer.bias)
            allweights.append(tmp) 
    
    save_weigths_to_mat(allweights, save_dir)

    '''
    # For Sequential Nets 
     allweights = []                              
     for child in model.children():   
         for layer in child.children():
             if isinstance(layer, torch.nn.modules.conv.Conv2d):
                 print(layer)
                 tmp = []      
                 tmp.append(layer.weight)
                 tmp.append(layer.bias)
                 allweights.append(tmp) 
    '''                                    




# def save_best_model(best_avg, current_avg, )
