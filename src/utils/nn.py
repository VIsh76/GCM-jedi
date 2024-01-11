def number_of_parameters(model):
    tot = 0
    for t in model.parameters():
        tot_param = 1
        for dim in range(len(t.size())):
            tot_param *= t.size(dim)
        tot += tot_param
    return tot



