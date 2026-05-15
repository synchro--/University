****## Tensor Decomposition Workspace
Testing methods to decompose convolutional layer into 3/4 optimized sub-layers which squeeze the model into a much lighter one. 

## Goal
Release a library to perform tensor decomposition on every CNNs. 

## Usage 
There are sevearal ways to use perform layer decomposition with standard or more fine-grained control depending on your requirements. 

The easiest way is to let the module decide which compression could be the best for your layer (there's still research ongoing on this topic), like so: 

```python
from decomposer import pytorch_cp_decomposition

# define your model 
model = models.alexnet()
params_original = sum([param.nelement() for param in model.parameters()])

# get a decomposed layer (e.g. model.conv1)
compressed_layer = pytorch_cp_decomposition(model.conv1)
# assign the layer to where it belongs in your model 
model.conv1 = compressed_layer

# Print the comparison between the two models
params_compressed = sum([param.nelement() for param in model.parameters()])
print('Number of trainable params before decomposition:', params_before)
print('Number of trainable params after decomposition:', params_compressed)
print('Compression: {.3f}x'.format(params_before/params_compressed))




```

## TODO
- MobileNets decomposed 
- Add procedure to train on CIFAR100 
- Test the decomposer class 

Conviene fare un metodo _make_block che crea un modulo diviso per 4 oppure creare direttamente la rete in maniera normale e poi usare il decomposer con xavier init? cosi uno definisce la sua rete in maniera normale e poi passa tutti i layer di convoluzione al decomposer che restituisce un modello totalmente decomposto con xavier init (con xavier init solo per design, cioè non per i layer che sono pretrainati)

Probabilmente entrambi.

Others: 


- remove logger functions from pytorch_utils and update all the files to use the Logger class 
- update summarize function in all files 
- add function to save model weights and model to the logger, by calling it only on the model. 

### Decomposer 
All of this must be done regardless of the decomposition method, so the method can be an option for all of them

1. create function to decompose a specified layer ==> the layer is specified by the user. 
    - this function must have different signatures according to the control the user wants to have regarding to rank estimation 
    - the standard method tries to do the best way, i.e. estimate ranks in the best possible way fast/accuracy 
    - another method takes in the 'desired compression' and performs the decomposition 
    - 
2. create function to decompose all layers, looping on the previous one 
3. create function to decompose_and_finetune that decomposes the layer and finetune the whole network (but fine-tuning should be an indepentent task)
    - same versions as 1. 
- create function to dec
