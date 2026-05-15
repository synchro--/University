# CPD Workplace 

### Names
CPD-Arch: all those models built with the 4-layers decomposition of the conv layer but without the actual CPD weights
          these are all trained from scratch. Resulting in a lot less parameters overall. Thus, it isn't doable to
          to apply these to pre-trained model. 

CPD:      Pre-trained models whose weights have been substituted with those computed by the CPD in TensorLab. 

Top or Class: Models whose last classifier (i.e. FC layer which holds most of the parameters) has been substituted
              with a Conv layer which has been decomposed itself in 4 smaller conv layers according to the paper of
              Lebedev.


These pattern repeats in the names of the logs & weights, e.g. weights.cifar.cpd.arch.hdf5. Sometimes even the number of parameters is specified, like 120K, 4K, or 1.2M. 
