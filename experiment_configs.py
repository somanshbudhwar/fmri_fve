FVE = [0.0,0.2,0.4,0.6,0.8,1.0]
COLS = [100,]
LAYERS = [3,]
HIDDEN_LAYERS_FACTORS = [1,]
ACTIVATIONS = ["leaky_relu"]
OUTPUT_SIZE = 1
SCALING_CONSTANT = [3.0]

# Create a list of all possible combinations of hyperparameters
def create_experiment_configs():
    experiment_configs = []
    for fve in FVE:
        for cols in COLS:
            for layers in LAYERS:
                for hidden_layers_factor in HIDDEN_LAYERS_FACTORS:
                    for activation in ACTIVATIONS:
                        for scaling_constant in SCALING_CONSTANT:
                            experiment_configs.append({
                                "fve": fve,
                                "cols": cols,
                                "layers": layers,
                                "hidden_layers_factor": hidden_layers_factor,
                                "hidden_layers": [int(cols/(hidden_layers_factor**(i+1))) for i in range(layers)],
                                "activation": activation,
                                "output_size": OUTPUT_SIZE,
                                "scaling_constant": scaling_constant
                            })
    return experiment_configs
