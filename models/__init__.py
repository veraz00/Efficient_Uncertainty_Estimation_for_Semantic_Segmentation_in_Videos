from .tiramisu import Tiramisu103
from .utils import * 

def get_model(name, classes):
    model = _get_model_instance(name)
    if name == 'bayesian_tiramisu':
        return model(num_classes= classes)
    else:
        raise f'model {name} not available'


def _get_model_instance(name):
    try:
        return {
            'bayesian_tiramisu': Tiramisu103,
        }[name]
    except:
        raise f'model {name} not available'