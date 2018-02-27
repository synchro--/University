import cpd
import cpd_training
import keras

from keras.models import load_model, save_model
from cpd import load_cpd_weights, make_conv2D_from_weights
from cpd_training import build_mnist_cpd_model, finetune_cifar_cpd

import sys


def main(argv):
    model_name = "saved_models/cifar10.h5"
    cpd_filename = "dumps-cifar/cpd1_32.mat"
    log_file = "logs_cifar.txt"
    l_index = 2

    if(len(sys.argv) == 1):
        print("[Usage]: train_mnist.py model_name|load model_name")
        sys.exit(-1)

    # if we specify a model_name to load, then it'll also be saved with the
    # same name
    if(len(sys.argv) == 3 and sys.argv[1] == "load"):
        model = load_model(sys.argv[2])
        model_to_save = sys.argv[2]
    else:
        model_to_save = sys.argv[1]
        model = build_mnist_cpd_model(model_name, cpd_filename, layer_index=l_index)

    finetune_cifar_cpd(model, lr=1e-6, epochs=15, layer_index=l_index, freeze_below=True, freeze_above=True, log_file=log_file)
    
    finetune_cifar_cpd(model, lr=1e-4, epochs=6, layer_index=l_index, freeze_below=True, freeze_above=False, log_file=log_file)
    
    finetune_cifar_cpd(model, lr=1e-4, epochs=6, optimizer="adam", layer_index=l_index, freeze_below=False, freeze_above=False, log_file=log_file)
    model.save(model_to_save)

if __name__ == "__main__":
    main(sys.argv[1:])
