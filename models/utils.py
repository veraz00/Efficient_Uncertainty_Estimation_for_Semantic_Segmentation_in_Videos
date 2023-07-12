import torch
def init_all(model):
    init_funcs = {
        1: lambda x: torch.nn.init.normal_(x, mean=0.0, std=1.0),  # can be bias
        2: lambda x: torch.nn.init.kaiming_uniform_(x, nonlinearity='relu'),  # can be weight
        3: lambda x: torch.nn.init.kaiming_uniform_(x, nonlinearity='relu'),  # can be conv1D filter
        4: lambda x: torch.nn.init.kaiming_uniform_(x, nonlinearity='relu'),  # can be conv2D filter
        5: lambda x: torch.nn.init.kaiming_normal_(x, mode="fan_out"),
        "default": lambda x: torch.nn.init.constant(x, 1.0),  # everything else
    }

    for p in model.parameters():
        init_func = init_funcs.get(len(p.shape), init_funcs["default"])
        init_func(p)
    return model
