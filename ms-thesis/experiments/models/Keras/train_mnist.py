import cpd 
import cpd_training 
import keras 

from keras.models import load_model, save_model
from cpd import load_cpd_weights, make_conv2D_from_weights
from cpd_training import build_mnist_cpd_model, finetune_mnist_cpd

import sys

def main(argv):
    model_name = "saved_models/mnist.h5"
    cpd_filename = "dumps-mnist/cpd2_5.mat"
    log_file = "logs_mnist.txt"

    if(len(sys.argv) == 1): 
        print("[Usage]: train_mnist.py model_name|load model_name")
        sys.exit(-1)

    # if we specify a model_name to load, then it'll also be saved with the same name
    if(len(sys.argv) == 3 and sys.argv[1] == "load"):
        model = load_model(sys.argv[2])
        model_to_save = sys.argv[2]
    else: 
        model_to_save = sys.argv[1]
        model = build_mnist_cpd_model(model_name, cpd_filename, layer_index=1)
    
    finetune_mnist_cpd(model, lr=1e-3, epochs=6, layer_index=1, freeze_below=True, freeze_above=True, log_file=log_file)
    finetune_mnist_cpd(model, lr=1e-5, epochs=6, layer_index=1, freeze_below=True, freeze_above=False, log_file=log_file)
    
    finetune_mnist_cpd(model, lr=1e-5, epochs=6, optimizer="adam", layer_index=1, freeze_below=True, freeze_above=False, log_file=log_file)
    model.save(model_to_save)

if __name__ == "__main__":
    main(sys.argv[1:])
