def fill_missing_values(parameters):
    """
    Fill the missing values of the parameters of the architectures
    Using the graph of the NN (embedding -> encoder -> process -> decoder) and
    the list of inputs
    """
    forced_input_variables = 0
    for f in parameters['variables']['forced']:
        forced_input_variables += len(parameters['variables']['forced'][f])

    # Embedding
    parameters['architecture']['surface_embedding']['input_size'] = len(parameters['variables']['pred']['surface']) + forced_input_variables
    parameters['architecture']['column_embedding']['input_size'] = len(parameters['variables']['pred']['column'])

    # Encoder
    parameters['architecture']['encoder']['surface_vars'] = parameters['architecture']['surface_embedding']['output_size']
    parameters['architecture']['encoder']['column_vars']  = parameters['architecture']['column_embedding']['output_size']
    parameters['architecture']['encoder']['n_levels']  = parameters['variables']['n_levels']

    #Processor    
    parameters['architecture']['process']['input_size'] =  parameters['architecture']['encoder']['output_size']

    # Decoder
    parameters['architecture']['decoder']['input_size'] =  parameters['architecture']['process']['output_size']
    parameters['architecture']['decoder']['surface_vars'] = len(parameters['variables']['pred']['surface'])
    parameters['architecture']['decoder']['column_vars'] = len(parameters['variables']['pred']['column'])
    parameters['architecture']['decoder']['n_levels']  = parameters['variables']['n_levels']

    # kernel dimension 1or3 handling
    return parameters
