## Tensor Decomposition Workspace
Testing methods to decompose convolutional layer into 3/4 optimized sub-layers which squeeze the model into a much lighter one. 

## Goal
Release a library to perform tensor decomposition on every CNNs. 

## TODO
- ResNet decomposed 
- MobileNets decomposed 
- Add procedure to train on CIFAR100 
- Test the decomposer class 

Conviene fare un metodo _make_block che crea un modulo diviso per 4 oppure creare direttamente la rete in maniera normale e poi usare il decomposer con xavier init? 

cosi uno definisce la sua rete in maniera normale e poi passa tutti i layer di convoluzione al decomposer che restituisce un modello totalmente decomposto con xavier init. 