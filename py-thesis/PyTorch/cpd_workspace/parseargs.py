import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type = str, default = "cifar")
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--decompose", dest="decompose", action="store_true")
    parser.add_argument("--layer",type=str, default="conv1")
    parser.add_argument("--fine-tune", dest="fine_tune", action="store_true")
    parser.add_argument("--train_path", type = str, default = "train")
    parser.add_argument("--test_path", type = str, default = "test")
    parser.add_argument("--cp", dest="cp", action="store_true", \
        help="Use cp decomposition. uses tucker by default")
    parser.add_argument('--layers', nargs='+', dest="layers", default = ['conv4', 'conv3', 'conv2', 'conv1'])
    parser.set_defaults(train=False)
    parser.set_defaults(decompose=False)
    parser.set_defaults(fine_tune=False)
    parser.set_defaults(cp=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    print(args.layers)
    