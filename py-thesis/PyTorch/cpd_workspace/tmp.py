def get_rec_layers_helper(layer_dict, idx, layer):
    for i, (name, layer) in enumerate(layer.named_children()):
        # if layer has children, go recursive
        if list(layer.children()) != []:
            print('rec')
            idx += get_rec_layers_helper(layer_dict, idx, layer)
        # otherwise, get index and layer
        else:
            print(str(idx) + ':' + str(layer))
            layer_dict[idx] = layer
            idx += 1
    return idx


def get_rec_layers(model):
    layer_dict = dict()
    idx = 0
    get_rec_layers_helper(layer_dict, idx, model)
    return layer_dict

##################################################
##################################################


def get_list_rec_layers_helper(layer_list, layer):
    for (name, layer) in layer.named_children():
        if type(layer) == list(layer.children()) != []:
            print('rec')
            get_list_rec_layers_helper(layer_list, layer)
        else:
            layer_list.append(name)


def get_list_rec_layers(model):
    layer_list = []
    get_list_rec_layers_helper(layer_list, model)
    return layer_list
