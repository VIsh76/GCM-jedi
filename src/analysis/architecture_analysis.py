def number_of_parameters(model):
    tot = 0
    for t in model.parameters():
        tot_param = 1
        for dim in range(len(t.size())):
            tot_param *= t.size(dim)
        tot += tot_param
    return tot

def get_all_weights_name(model):
    name_list = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            name_list.append(name)
    return name_list

def get_all_layers_name(model):
    layer_list = []
    weight_list = get_all_weights_name(model) 
    for name in weight_list:
        layer_name = '.'.join(name.split('.')[:-1]) # drop the last element (which is the weight name)
        if not (layer_name in layer_list):
            layer_list.append(layer_name)
    return layer_list  

def get_activation(name, activation):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


